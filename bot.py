"""Telegram-бот на GLM (Z.ai) для деплоя в amvera.ru.

Память диалога в памяти процесса, доступ только по chat_id, стриминг ответа.
"""

import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message
from openai import AsyncOpenAI

try:
    from ddgs import DDGS
except ImportError:  # duckduckgo-search <7 экспортирует модуль иначе
    from duckduckgo_search import DDGS

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
# Модель для web-режима: glm-4.6 — подтверждённый function calling, дешевле.
WEB_MODEL = os.environ.get("WEB_MODEL", "glm-4.6")

ALLOWED_IDS = {
    int(x)
    for x in os.environ.get("ALLOWED_CHAT_IDS", "").replace(" ", "").split(",")
    if x
}

ALLOWED_MODELS = {"glm-5.2", "glm-4.6", "glm-4.5", "glm-4.5-air", "glm-4.5-flash"}

HISTORY_LIMIT = 20        # последних пар (user+assistant) в контексте
TG_MSG_LIMIT = 4000       # запас от лимита Telegram в 4096
UPDATE_INTERVAL = 1.0     # сек между правками сообщения при стриминге
WEB_RESULTS = 5           # сколько результатов поиска отдавать в модель
WEB_SNIPPET = 300         # макс. длина сниппета в символах
WEB_TIMEOUT = 5.0         # сек — таймаут на запрос к DuckDuckGo

def _today_str() -> str:
    """Текущая дата с сервера (формат: 28.07.2026 (Monday))."""
    return datetime.now().strftime("%d.%m.%Y (%A)")


def build_system_prompt(web_on: bool) -> str:
    """Системный промпт с актуальной датой.

    В web-режиме — жёсткое указание ОБЯЗАНО использовать поиск и НИКОГДА не
    писать про отсутствие интернета / «знания ограничены 2023 годом».
    """
    date_str = _today_str()
    base = (
        f"Ты — русскоязычный ассистент в Telegram. Отвечай кратко и по делу. "
        f"Сегодняшняя дата: {date_str}."
    )
    if web_on:
        return base + (
            " У тебя ЕСТЬ доступ в интернет через функцию поиска. "
            "Если пользователь спрашивает про погоду, актуальные события, "
            "курсы, цены, версии, даты или любые свежие факты — ты ОБЯЗАН "
            "использовать web_search и отвечать на основе найденного, "
            "указывая источники. НИКОГДА не пиши, что у тебя нет доступа к "
            "интернету или что твои знания ограничены каким-либо годом: "
            "всегда сначала ищи, потом отвечай."
        )
    return base + (
        " Отвечай из своих знаний. Если вопрос требует свежих данных, "
        "которых у тебя нет, честно скажи об этом."
    )

# OpenAI-style схема инструмента веб-поиска.
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Веб-поиск через DuckDuckGo. Возвращает список результатов: "
            "заголовок, краткое описание, ссылка. Используй, когда нужен "
            "свежий или точный факт из интернета."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Поисковый запрос, лучше на языке ответа.",
                }
            },
            "required": ["query"],
        },
    },
}

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
# chat_id -> включён ли web-режим
web_mode: dict[int, bool] = defaultdict(bool)

ALLOWED_FILTER = F.chat.id.in_(ALLOWED_IDS)


def _ensure_system(chat_id: int, web_on: bool) -> None:
    """Гарантирует, что история начинается с актуального system-сообщения.

    Содержимое пересобирается под текущий web-режим при каждом сообщении,
    чтобы переключение /web сразу меняло инструкции модели.
    """
    prompt = build_system_prompt(web_on)
    msgs = history[chat_id]
    if msgs and msgs[0].get("role") == "system":
        msgs[0]["content"] = prompt
    else:
        msgs.insert(0, {"role": "system", "content": prompt})


def _ddg_search(query: str) -> str:
    """Синхронный поиск через DuckDuckGo. Возвращает текст для tool-сообщения.

    Вызывается через asyncio.to_thread, чтобы не блокировать event loop.
    ВАЖНО: backend='duckduckgo' — единственный движок. Без него ddgs (метапоиск)
    гоняет цепочку из 7 бэкендов (Wikipedia/Grokipedia/Startpage/Mojeek/Google/
    Yandex/Brave), что даёт задержки 60-70с и роняет Telegram по таймауту.
    Жёсткий timeout=5 сек ограничивает весь поиск.
    """
    try:
        with DDGS(timeout=WEB_TIMEOUT) as ddgs:
            results = list(
                ddgs.text(
                    query,
                    backend="duckduckgo",
                    max_results=WEB_RESULTS,
                )
            )
    except Exception as e:
        log.warning("web_search failed: %s", e)
        return f"Поиск недоступен: {e}"

    if not results:
        return "По вашему запросу ничего не найдено."

    lines = []
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "").strip()
        snippet = (r.get("body") or r.get("snippet") or "").strip()
        url = (r.get("href") or r.get("url") or "").strip()
        if len(snippet) > WEB_SNIPPET:
            snippet = snippet[:WEB_SNIPPET].rstrip() + "…"
        lines.append(f"{i}. {title}\n{snippet}\n{url}")
    return "\n\n".join(lines)


async def _web_search(query: str) -> str:
    """Асинхронная обёртка: синхронный поиск в фоновом потоке, не блокирует loop.

    asyncio.to_thread гарантирует, что блокирующий I/O DuckDuckGo уходит из
    основного потока event loop. Таймаут — через asyncio.wait_for.
    """
    return await asyncio.wait_for(
        asyncio.to_thread(_ddg_search, query),
        timeout=WEB_TIMEOUT,
    )


@dp.message(ALLOWED_FILTER, Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "Привет! Я бот на GLM. Пиши — отвечу.\n"
        "<b>/clear</b> — сбросить контекст\n"
        "<b>/model &lt;name&gt;</b> — сменить модель\n"
        "<b>/web on|off</b> — доступ в интернет (поиск)"
    )


@dp.message(ALLOWED_FILTER, Command("clear"))
async def cmd_clear(message: Message):
    history[message.chat.id].clear()
    await message.answer("🧹 Контекст очищен.")


@dp.message(ALLOWED_FILTER, Command("web"))
async def cmd_web(message: Message):
    parts = (message.text or "").split(maxsplit=1)
    chat_id = message.chat.id
    if len(parts) < 2:
        cur = "вкл" if web_mode[chat_id] else "выкл"
        await message.answer(
            f"🌐 Web-режим сейчас: <b>{cur}</b>.\n"
            "<code>/web on</code> — включить\n<code>/web off</code> — выключить"
        )
        return
    arg = parts[1].strip().lower()
    if arg in {"on", "1", "да", "вкл", "включить"}:
        web_mode[chat_id] = True
        await message.answer("🌐 Web-режим включён. Буду искать через DuckDuckGo.")
    elif arg in {"off", "0", "нет", "выкл", "выключить"}:
        web_mode[chat_id] = False
        await message.answer("🕸 Web-режим выключен. Отвечаю из своих знаний.")
    else:
        await message.answer(
            "Использование: <code>/web on</code> или <code>/web off</code>"
        )


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
    web_on = web_mode[chat_id]
    # В web-режиме используем WEB_MODEL (подтверждённый function calling),
    # иначе — выбранную пользователем модель чата.
    model = WEB_MODEL if web_on else user_model.get(chat_id, DEFAULT_MODEL)

    _ensure_system(chat_id, web_on)
    history[chat_id].append({"role": "user", "content": message.text})
    _trim_history(chat_id)

    # Плашка ожидания — в САМОМ НАЧАЛЕ, до любых вызовов GLM и веб-поиска.
    # Позже она редактируется: на «🔍 Ищу...» во время поиска и на финальный
    # ответ через edit_text в _stream_answer.
    placeholder = await message.answer("⏳ Ищу информацию..." if web_on else "…")

    # В web-режиме сначала просим модель решить: нужен ли поиск.
    if web_on:
        tool_calls = await _maybe_search(chat_id, model, placeholder)
        if tool_calls is None:
            # Уже ответили пользователю в _maybe_search (ошибка tools-не-поддержки).
            return

    # Стриминг финального ответа (всегда — и в обычном, и в web-режиме после tool).
    await _stream_answer(chat_id, model, placeholder)


def _trim_history(chat_id: int) -> None:
    """Держит system + последние HISTORY_LIMIT пар сообщений."""
    msgs = history[chat_id]
    # system не должен вылетать при обрезке.
    sys_part = msgs[:1] if msgs and msgs[0].get("role") == "system" else []
    rest = msgs[1:] if sys_part else msgs
    limit = HISTORY_LIMIT * 2
    if len(rest) > limit:
        rest = rest[-limit:]
    history[chat_id] = sys_part + rest


# Ключевые слова, при которых поиск обязателен (в web-режиме).
_FRESH_KEYWORDS = (
    "погод", "новост", "курс", "цен", "сегодн", "вчер", "сейчас", "актуальн",
    "последн", "недавн", "свеж", "верси", "релиз", "итог", "счёт", "счет",
    "расписан", "результат", "катастроф", "выбор", "событ", "произош",
    "latest", "current", "today", "now", "news", "weather", "price",
)


def _needs_search(text: str) -> bool:
    """Эвристика: вопрос явно про свежее/актуальное → форсировать поиск."""
    t = (text or "").lower()
    return any(k in t for k in _FRESH_KEYWORDS)


async def _maybe_search(chat_id, model, placeholder) -> "list | None":
    """Не-стриминговый запрос с tool web_search.

    Возвращает список tool_calls (возможно пустой) или None, если нужно прервать
    (ошибка, на которую уже ответили пользователю).
    """
    user_text = message_text(chat_id)
    # Если вопрос явно про свежее — заставляем модель вызвать инструмент.
    force = _needs_search(user_text)
    tool_choice = {"type": "function", "function": {"name": "web_search"}} if force else "auto"
    try:
        log.info("GLM web-decision request: model=%s tool_choice=%s", model,
                 "any(web_search)" if force else "auto")
        resp = await glm.chat.completions.create(
            model=model,
            messages=history[chat_id],
            tools=[WEB_SEARCH_TOOL],
            tool_choice=tool_choice,
        )
    except Exception as e:
        log.exception("GLM web-decision error")
        if history[chat_id] and history[chat_id][-1]["role"] == "user":
            history[chat_id].pop()
        msg = str(e)
        hint = ""
        if "tool" in msg.lower() or "function" in msg.lower():
            hint = "\n💡 Похоже, эта модель не поддерживает web-поиск. " \
                   "Попробуй /model glm-4.6"
        await _safe_edit(placeholder, f"⚠️ Ошибка GLM: {e}{hint}")
        return None

    choice = resp.choices[0]
    if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
        # Модель решила искать не нужно — отвечаем из знаний (стримингом ниже).
        log.info("no tool_calls; finish_reason=%s", choice.finish_reason)
        return []

    # Сохраняем ассистент-сообщение с tool_calls, как требует схема OpenAI.
    history[chat_id].append(choice.message.model_dump(exclude_none=True))

    for call in choice.message.tool_calls:
        if call.function.name != "web_search":
            continue
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        query = (args.get("query") or "").strip() or message_text(chat_id)
        await _safe_edit(placeholder, f"🔍 Ищу: {query[:100]}…")
        # Явно отдаём управление в event loop, чтобы правка статуса улетела
        # в Telegram ДО запуска долгого поиска в фоновом потоке.
        await asyncio.sleep(0)
        result = await _web_search(query)
        history[chat_id].append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            }
        )
        log.info("web_search '%s' -> %d chars", query, len(result))

    _trim_history(chat_id)
    return choice.message.tool_calls


def message_text(chat_id: int) -> str:
    """Последний текст пользователя из истории (фоллбэк для query)."""
    for m in reversed(history[chat_id]):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


async def _stream_answer(chat_id, model, placeholder) -> None:
    """Стримит финальный ответ модели и обновляет сообщение в Telegram."""
    buf = ""
    last_update = time.monotonic()
    actual_model = None

    try:
        log.info("GLM request: requested model=%s", model)
        stream = await glm.chat.completions.create(
            model=model,
            messages=history[chat_id],
            stream=True,
        )
        async for chunk in stream:
            # chunk.model — истинная модель, которую вернул сервер z.ai.
            if actual_model is None and getattr(chunk, "model", None):
                actual_model = chunk.model
                log.info("GLM response: actual model=%s", actual_model)
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


def _mask(value: str) -> str:
    """Маскированное значение для логов: длина + первые 4/последние 4 символа."""
    if not value:
        return f"<empty> (len=0)"
    if len(value) <= 8:
        return f"<hidden> (len={len(value)})"
    return f"{value[:4]}…{value[-4:]} (len={len(value)})"


async def main():
    # Диагностика env ПЕРВЫМ делом — до любых сетевых вызовов к Telegram/GLM.
    # Если токен невалиден, delete_webhook упадёт и роняет весь процесс,
    # но эти строки уже успеют попасть в лог amvera.
    log.info("Bot starting. Allowed chat_ids: %s", sorted(ALLOWED_IDS) or "(none)")
    log.info("ENV BOT_TOKEN=%s", _mask(BOT_TOKEN))
    log.info("ENV GLM_API_KEY=%s", _mask(GLM_API_KEY))
    log.info("ENV GLM_BASE_URL=%s", GLM_BASE_URL)
    log.info("ENV GLM_MODEL=%s", DEFAULT_MODEL)
    log.info("ENV WEB_MODEL=%s", WEB_MODEL)
    log.info("ENV ALLOWED_CHAT_IDS=%s", sorted(ALLOWED_IDS))
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    # Рекомендация amvera: задержка против конфликта getUpdates при рестарте.
    time.sleep(2)
    asyncio.run(main())
