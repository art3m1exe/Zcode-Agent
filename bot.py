"""Telegram-бот на GLM (Z.ai) для деплоя в amvera.ru.

Память диалога в памяти процесса, доступ только по chat_id, стриминг ответа.
"""

import asyncio
import json
import logging
import os
import re
import time
from collections import defaultdict
from datetime import datetime

import aiohttp
from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message
from bs4 import BeautifulSoup
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
WEB_TIMEOUT = 5.0         # сек — жёсткий таймаут на весь веб-поиск
WEB_FETCH_TIMEOUT = 4.0   # сек — таймаут на скачивание страницы для глубокого парсинга
WEB_FETCH_CHARS = 2000    # макс. символов текста, извлекаемого со страницы
# Хосты, с которых имеет смысл тянуть полный текст (расписания/билеты/события):
# сниппеты поисковика там слишком скудные, нужны точные время/номера.
WEB_DEEP_HOSTS = (
    "rasp.yandex.ru", "tutu.ru", "rzd.ru", "pass.rzd.ru",
    "busfor.ru", "aviasales.ru", "flyone", "aeroport",
    "kassir.ru", "radario", "timepad.ru", "qtickets",
    "onely", "ponominalu", "spb.kassir", "msk.kassir",
)
# ddgs v9 — метапоиск: при backend='auto' перебирает до 7 движков, часть падает
# (DuckDuckGo/Wikipedia/Google часто отдают DDGSException), и в сумме поиск длится
# 30-60с, что рвёт keep-alive Telegram. Фиксируем один стабильный быстрый бэкенд.
WEB_BACKEND = os.environ.get("WEB_BACKEND", "yandex")

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
            "всегда сначала ищи, потом отвечай. "
            "При ответе на запросы о расписаниях, билетах и мероприятиях ты "
            "ОБЯЗАН указывать точные данные из найденного текста: номера "
            "поездов/рейсов, точное время отправления и прибытия, станции, "
            "цены. Никогда не отправляй пользователя искать информацию "
            "самостоятельно, если данные можно извлечь из поиска. "
            "НИКОГДА не генерируй теги <tool_call> или XML-структуры в тексте "
            "ответа: для вызова функции используй только штатный механизм "
            "function-calling. В тексте ответа для пользователя не должно быть "
            "никаких <tool_call>, <arg_key>, <arg_value> и подобных тегов."
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
# chat_id -> блокировка, чтобы не гонять параллельные ответы в одном чате
chat_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
# Ссылки на фоновые задачи (чтобы GC не убил). Чистятся в done-callback.
_bg_tasks: set = set()

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


def _ddg_search(query: str) -> list[dict]:
    """Синхронный поиск. Возвращает список результатов [{title, snippet, url}].

    Вызывается через asyncio.to_thread, чтобы не блокировать event loop.
    Один движок (WEB_BACKEND, по умолчанию 'yandex'): ddgs v9 — это метапоиск,
    при backend='auto' он перебирает до 7 движков; упавшие (DuckDuckGo/Wikipedia/
    Google часто отдают DDGSException) заставляют идти дальше, поиск растягивается
    на 30-60с и рвёт keep-alive Telegram. Жёсткий timeout=WEB_TIMEOUT.
    """
    try:
        with DDGS(timeout=WEB_TIMEOUT) as ddgs:
            raw = list(
                ddgs.text(
                    query,
                    backend=WEB_BACKEND,
                    max_results=WEB_RESULTS,
                )
            )
    except Exception as e:
        log.warning("web_search failed (backend=%s): %s", WEB_BACKEND, e)
        return []

    results = []
    for r in raw:
        url = (r.get("href") or r.get("url") or "").strip()
        snippet = (r.get("body") or r.get("snippet") or "").strip()
        if len(snippet) > WEB_SNIPPET:
            snippet = snippet[:WEB_SNIPPET].rstrip() + "…"
        results.append(
            {
                "title": (r.get("title") or "").strip(),
                "snippet": snippet,
                "url": url,
            }
        )
    return results


def _is_deep_host(url: str) -> bool:
    """Стоит ли тянуть полный текст с этой страницы (расписания/билеты)."""
    u = (url or "").lower()
    return any(h in u for h in WEB_DEEP_HOSTS)


async def _fetch_page_text(url: str) -> str:
    """Скачивает страницу и извлекает видимый текст (BeautifulSoup).

    Возвращает '' при любой ошибке/таймауте — глубокий парсинг факультативен.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept-Language": "ru-Ru,ru;q=0.9",
    }
    try:
        timeout = aiohttp.ClientTimeout(total=WEB_FETCH_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers, allow_redirects=True) as resp:
                if resp.status != 200:
                    return ""
                # Решаем кодировку: если сервер не дал, пробуем utf-8 затем cp1251.
                raw = await resp.read()
                enc = resp.charset or "utf-8"
                try:
                    html = raw.decode(enc, errors="ignore")
                except (LookupError, UnicodeDecodeError):
                    html = raw.decode("utf-8", errors="ignore")
        soup = BeautifulSoup(html, "lxml")
        # Убираем шум.
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        # Нормализуем: убираем длинные серии пробелов/пустых строк.
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        text = "\n".join(lines)
        return text[:WEB_FETCH_CHARS]
    except Exception as e:
        log.debug("deep fetch failed %s: %s", url, e)
        return ""


def _format_search_results(results: list[dict], deep: dict[str, str]) -> str:
    """Собирает итоговый текст для модели: сниппеты + (опц.) полный текст страницы."""
    if not results:
        return "По вашему запросу ничего не найдено."
    lines = []
    for i, r in enumerate(results, 1):
        block = f"{i}. {r['title']}\n{r['snippet']}\n{r['url']}"
        page_text = deep.get(r["url"])
        if page_text:
            block += f"\n[ТЕКСТ СТРАНИЦЫ]:\n{page_text}"
        lines.append(block)
    return "\n\n".join(lines)


async def _web_search(query: str) -> str:
    """Веб-поиск + глубокий парсинг релевантных страниц, с жёсткими таймаутами.

    1) ddgs в фоне (to_thread) под wait_for(WEB_TIMEOUT).
    2) Если среди ссылок есть сайты расписаний/билетов (WEB_DEEP_HOSTS),
       тянем и парсим первую такую страницу — в ней точные время/номера.
    Все сетевые операции — неблокирующие; на любом сбое fallback на сниппеты.
    """
    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(_ddg_search, query),
            timeout=WEB_TIMEOUT,
        )
    except (asyncio.TimeoutError, Exception) as e:
        log.error("Search failed or timed out: %s", e)
        return "Информация из поиска недоступна."

    if not results:
        return "Информация из поиска недоступна."

    # Глубокий парсинг: первая релевантная страница из разрешённых хостов.
    deep: dict[str, str] = {}
    for r in results:
        if _is_deep_host(r["url"]):
            try:
                page_text = await asyncio.wait_for(
                    _fetch_page_text(r["url"]),
                    timeout=WEB_FETCH_TIMEOUT + 1.0,
                )
            except (asyncio.TimeoutError, Exception) as e:
                log.debug("deep fetch timed out %s: %s", r["url"], e)
                page_text = ""
            if page_text:
                deep[r["url"]] = page_text
                break  # достаточно одной страницы с точными данными

    return _format_search_results(results, deep)


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
    """Первая (и единственная быстроя) операция в polling-корутине.

    Шлём плашку и сразу возвращаем управление — вся тяжёлая работа (GLM,
    веб-поиск, стриминг) уходит в фоновую задачу. Пока она бежит, aiogram
    свободен отвечать на keep-alive Telegram, соединение не рвётся.
    """
    chat_id = message.chat.id
    web_on = web_mode[chat_id]
    # Плашка — в САМОМ НАЧАЛЕ, до любых вызовов GLM и веб-поиска.
    status_msg = await _send_status(message, "⏳ Ищу информацию..." if web_on else "…")
    # Тяжёлая обработка — в фон, polling не блокируется.
    task = asyncio.create_task(_process_message(message, status_msg))
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


async def _process_message(message: Message, status_msg: Message) -> None:
    """Фоновая обработка: web-decision → поиск → финальный стрим.

    Работает вне polling-корутины, под per-chat локом (защита от спама).
    ВСЁ тело обёрнуто в try/except: фоновая таска падает тихо, и если не
    обновить плашку, «⏳ Ищу информацию...» зависнет навсегда.
    """
    chat_id = message.chat.id
    try:
        web_on = web_mode[chat_id]
        model = WEB_MODEL if web_on else user_model.get(chat_id, DEFAULT_MODEL)

        lock = chat_locks[chat_id]
        async with lock:
            _ensure_system(chat_id, web_on)
            history[chat_id].append({"role": "user", "content": message.text})
            _trim_history(chat_id)

            # В web-режиме сначала просим модель решить: нужен ли поиск.
            if web_on:
                tool_calls = await _maybe_search(chat_id, model, status_msg)
                if tool_calls is None:
                    # Уже ответили пользователю в _maybe_search.
                    return

            # Стриминг финального ответа (всегда — и в обычном, и в web-режиме после tool).
            await _stream_answer(chat_id, model, status_msg)
    except Exception as e:
        log.exception("Ошибка в фоновой таске: %s", e)
        # Плашка не должна висеть «⏳ Ищу...» бесконечно — сообщаем об ошибке.
        await _safe_edit(status_msg, "❌ Произошла ошибка при поиске ответа.")


async def _send_status(message: Message, text: str):
    """Шлём плашку с ретраями: кратковременный разрыв connection pool к Telegram
    не должен валить весь апдейт."""
    last = None
    for attempt in range(3):
        try:
            return await message.answer(text)
        except Exception as e:
            last = e
            log.warning("answer attempt %d failed: %s", attempt + 1, e)
            await asyncio.sleep(0.5)
    raise last


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


async def _stream_answer(chat_id, model, placeholder, _tool_attempts: int = 0) -> None:
    """Стримит финальный ответ модели и обновляет сообщение в Telegram.

    Если модель (glm-4.5-flash иногда так делает) проигнорировала штатный
    function-calling и выписала вызов tool прямо текстом (<tool_call>...),
    перехватываем: парсим query из <arg_value>, выполняем веб-поиск, кладём
    результат в историю и перезапрашиваем модель — уже без tool-тегов.
    _tool_attempts ограничивает число таких перехватов (защита от зацикливания).
    """
    MAX_TOOL_ATTEMPTS = 2
    buf = ""
    last_update = time.monotonic()
    actual_model = None
    started_tool_tag = False  # True, если в потоке замечен '<tool_call>' —
                              # такие куски юзеру не показываем.

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
            # Появился текстовый tool_call — не показываем его юзеру, копим в buf.
            if "<tool_call>" in buf:
                started_tool_tag = True
                continue
            if not started_tool_tag:
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

    # Перехват текстового tool_call: модель не использовала function-calling.
    query = _parse_text_tool_call(full)
    if query and _tool_attempts < MAX_TOOL_ATTEMPTS:
        log.info("text tool_call intercepted (attempt %d), query=%r",
                 _tool_attempts + 1, query)
        await _safe_edit(placeholder, f"🔍 Ищу: {query[:100]}…")
        result = await _web_search(query)
        history[chat_id].append({"role": "assistant", "content": full})
        history[chat_id].append(
            {
                "role": "user",
                "content": (
                    f"Результаты веб-поиска по запросу «{query}»:\n{result}\n\n"
                    "На основе этих результатов дай пользователю краткий "
                    "человеческий ответ. НЕ используй теги <tool_call>."
                ),
            }
        )
        # Рекурсивный вызов — теперь модель должна ответить нормально.
        await _stream_answer(chat_id, model, placeholder, _tool_attempts + 1)
        return
    if query:
        # Лимит перехватов исчерпан — модель упорно шлёт теги. Чистим и отдаём
        # как есть, вырезав сами теги, чтобы не пугать юзера разметкой.
        log.warning("tool_call attempts exhausted, stripping tags")
        full = re.sub(r"<tool_call>.*?</tool_call>", "", full, flags=re.DOTALL).strip()
        if not full:
            full = "Не удалось получить ответ."

    # Чистый человеческий ответ — обычный путь.
    history[chat_id].append({"role": "assistant", "content": full})

    chunks = [full[i:i + TG_MSG_LIMIT] for i in range(0, len(full), TG_MSG_LIMIT)]
    # Финальный ответ — с fallback по разметке: сырые символы в ответе/ссылках
    # поиска ломают HTML/Markdown (TelegramBadRequest). Сначала Markdown,
    # при ошибке — чистый текст без parse_mode.
    await _safe_edit_fallback(placeholder, chunks[0])
    for c in chunks[1:]:
        try:
            await _send_fallback(chat_id, c)
        except Exception:
            log.exception("send_message chunk failed")


def _parse_text_tool_call(text: str) -> str:
    """Достаёт запрос из текстового <tool_call>...<arg_value>query</arg_value>.

    Возвращает '' если это не текстовый вызов web_search.
    """
    if "<tool_call>" not in text:
        return ""
    # Сначала ищем web_search по имени функции, потом arg_value рядом.
    has_web = "web_search" in text.lower()
    m = re.search(r"<arg_value>(.*?)</arg_value>", text, re.DOTALL | re.IGNORECASE)
    if has_web and m:
        return m.group(1).strip()
    return ""


async def _safe_edit(message: Message, text: str):
    """Редактируем сообщение, игнорируя 'not modified' и rate-limit.

    На сетевых ошибках (connection pool к Telegram порвался) — пара ретраев
    с короткой паузой: разрыв обычно кратковременный.
    """
    for attempt in range(3):
        try:
            await bot.edit_message_text(
                text=text,
                chat_id=message.chat.id,
                message_id=message.message_id,
            )
            return
        except Exception as e:
            msg = str(e).lower()
            if "not modified" in msg or "too many requests" in msg:
                return
            if attempt < 2 and ("timeout" in msg or "network" in msg or "connection" in msg):
                log.warning("edit retry %d: %s", attempt + 1, e)
                await asyncio.sleep(0.5)
                continue
            log.debug("edit_message_text skipped: %s", e)
            return


async def _safe_edit_fallback(message: Message, text: str):
    """Финальная правка ответа с fallback по разметке.

    Сначала Markdown; если Telegram ругается на разметку (TelegramBadRequest
    из-за сырых символов/ссылок поиска) — повторяем как чистый текст.
    """
    try:
        await message.edit_text(text, parse_mode="Markdown")
    except Exception as e:
        log.debug("Markdown edit failed, retry as plain text: %s", e)
        try:
            await message.edit_text(text, parse_mode=None)
        except Exception as e2:
            # Не даём плашке зависеть: хотя бы чем-то её обновим.
            log.debug("plain edit also failed: %s", e2)


async def _send_fallback(chat_id: int, text: str):
    """Досылка доп. чанков с fallback по разметке (Markdown → plain)."""
    try:
        await bot.send_message(chat_id, text, parse_mode="Markdown")
    except Exception as e:
        log.debug("Markdown send failed, retry as plain text: %s", e)
        await bot.send_message(chat_id, text, parse_mode=None)


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
