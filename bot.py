"""Telegram-бот на GLM (Z.ai) для деплоя в amvera.ru.

Память диалога в памяти процесса, доступ только по chat_id, стриминг ответа.
"""

import asyncio
import logging
import os
import time
from collections import defaultdict

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message
from openai import AsyncOpenAI

# Локальный запуск читает .env; в amvera переменные приходят из панели.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
GLM_API_KEY = os.environ.get("GLM_API_KEY", "")
GLM_BASE_URL = os.environ.get("GLM_BASE_URL", "https://api.z.ai/api/coding/paas/v4")
DEFAULT_MODEL = os.environ.get("GLM_MODEL", "glm-5.2")

ALLOWED_IDS = {
    int(x)
    for x in os.environ.get("ALLOWED_CHAT_IDS", "").replace(" ", "").split(",")
    if x
}

ALLOWED_MODELS = {"glm-5.2", "glm-4.6", "glm-4.5", "glm-4.5-air", "glm-4.5-flash"}

HISTORY_LIMIT = 20        # последних пар (user+assistant) в контексте
TG_MSG_LIMIT = 4000       # запас от лимита Telegram в 4096
UPDATE_INTERVAL = 1.0     # сек между правками сообщения при стриминге

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("glm-bot")

bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
)
dp = Dispatcher()

glm = AsyncOpenAI(api_key=GLM_API_KEY, base_url=GLM_BASE_URL)

# chat_id -> список сообщений [{"role","content"}, ...]
history: dict[int, list[dict]] = defaultdict(list)
# chat_id -> выбранная модель
user_model: dict[int, str] = {}

ALLOWED_FILTER = F.chat.id.in_(ALLOWED_IDS)


@dp.message(ALLOWED_FILTER, Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "Привет! Я бот на GLM. Пиши — отвечу.\n"
        "<b>/clear</b> — сбросить контекст\n"
        "<b>/model &lt;name&gt;</b> — сменить модель"
    )


@dp.message(ALLOWED_FILTER, Command("clear"))
async def cmd_clear(message: Message):
    history[message.chat.id].clear()
    await message.answer("🧹 Контекст очищен.")


@dp.message(ALLOWED_FILTER, Command("model"))
async def cmd_model(message: Message):
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        cur = user_model.get(message.chat.id, DEFAULT_MODEL)
        await message.answer(
            f"Текущая модель: <code>{cur}</code>\n"
            f"Доступные: {', '.join(sorted(ALLOWED_MODELS))}\n"
            "Пример: <code>/model glm-4.5-flash</code>"
        )
        return
    name = parts[1].strip().lower()
    if name not in ALLOWED_MODELS:
        await message.answer(
            f"Не знаю такую модель. Доступные: {', '.join(sorted(ALLOWED_MODELS))}"
        )
        return
    user_model[message.chat.id] = name
    await message.answer(f"Модель сменена на <code>{name}</code>.")


@dp.message(ALLOWED_FILTER, F.text)
async def handle_text(message: Message):
    chat_id = message.chat.id
    model = user_model.get(chat_id, DEFAULT_MODEL)

    history[chat_id].append({"role": "user", "content": message.text})
    # Держим только последние HISTORY_LIMIT пар сообщений.
    limit = HISTORY_LIMIT * 2
    if len(history[chat_id]) > limit:
        history[chat_id] = history[chat_id][-limit:]

    placeholder = await message.answer("…")
    buf = ""
    last_update = time.monotonic()

    try:
        stream = await glm.chat.completions.create(
            model=model,
            messages=history[chat_id],
            stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if not delta:
                continue
            buf += delta
            now = time.monotonic()
            if now - last_update >= UPDATE_INTERVAL:
                await _safe_edit(placeholder, buf[:TG_MSG_LIMIT])
                last_update = now
    except Exception as e:
        log.exception("GLM error")
        # Ответа нет — убираем пользовательское сообщение из истории.
        if history[chat_id] and history[chat_id][-1]["role"] == "user":
            history[chat_id].pop()
        await _safe_edit(placeholder, f"⚠️ Ошибка GLM: {e}")
        return

    full = buf or "(пустой ответ)"
    history[chat_id].append({"role": "assistant", "content": full})

    chunks = [full[i:i + TG_MSG_LIMIT] for i in range(0, len(full), TG_MSG_LIMIT)]
    await _safe_edit(placeholder, chunks[0])
    for c in chunks[1:]:
        try:
            await bot.send_message(chat_id, c)
        except Exception:
            log.exception("send_message chunk failed")


async def _safe_edit(message: Message, text: str):
    """Редактируем сообщение, игнорируя 'not modified' и rate-limit."""
    try:
        await bot.edit_message_text(
            text=text,
            chat_id=message.chat.id,
            message_id=message.message_id,
        )
    except Exception as e:
        msg = str(e).lower()
        if "not modified" in msg or "too many requests" in msg:
            return
        log.debug("edit_message_text skipped: %s", e)


async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    log.info("Bot started. Allowed chat_ids: %s", sorted(ALLOWED_IDS) or "(none)")
    await dp.start_polling(bot)


if __name__ == "__main__":
    # Рекомендация amvera: задержка против конфликта getUpdates при рестарте.
    time.sleep(2)
    asyncio.run(main())
