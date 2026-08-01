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
from datetime import datetime, timedelta

import aiohttp
from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramNetworkError
from aiogram.filters import Command
from aiogram.types import ErrorEvent, Message, TelegramObject, Update
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
# Обычный план Z.ai (бесплатный glm-4.5-flash). Для Coding Plan: /api/coding/paas/v4.
GLM_BASE_URL = os.environ.get("GLM_BASE_URL", "https://api.z.ai/api/paas/v4")
DEFAULT_MODEL = os.environ.get("GLM_MODEL", "glm-4.5-flash")
# Модель для web-режима (function calling). glm-4.5-flash поддерживает его и бесплатна.
WEB_MODEL = os.environ.get("WEB_MODEL", "glm-4.5-flash")
# Ключ Яндекс.Расписаний (https://yandex.ru/dev/rasp). Без него интеграция
# с реальными расписаниями поездов выключена — бот честно об этом скажет.
YANDEX_RASP_KEY = os.getenv("YANDEX_RASP_KEY", "")
# Оба домена отвечают на запрос (проверено: HTTP 400 на фейковый ключ = живой
# эндпоинт). Перебираем оба — если первый упадёт/зависнет, второй подхватит.
YANDEX_RASP_URLS = (
    "https://api.rasp.yandex.net/v3.0/search/",
    "https://api.rasp.yandex-net.ru/v3.0/search/",
)

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
            " ГЛАВНОЕ ПРАВИЛО: точность важнее полноты. "
            "У тебя ЕСТЬ доступ в интернет через функцию поиска: если "
            "пользователь спрашивает про погоду, актуальные события, курсы, "
            "цены, версии, даты или любые свежие факты — сначала ищи через "
            "web_search и отвечай на основе найденного, указывая источники. "
            "НИКОГДА не пиши, что у тебя нет доступа к интернету или что твои "
            "знания ограничены каким-либо годом: всегда сначала ищи, потом "
            "отвечай. "
            "По расписаниям, билетам и мероприятиям указывай номера "
            "поездов/рейсов, время отправления/прибытия, станции, цены "
            "ТОЛЬКО если они реально есть в результатах поиска. Если точных "
            "данных нет — прямо скажи, что динамическое расписание доступно "
            "только на официальных сайтах (Яндекс.Расписания, Tutu, РЖД), "
            "и дай ссылки. НИКОГДА не выдумывай время, номера рейсов или "
            "цены: лучше честно написать «точное время не нашлось», чем "
            "назвать придуманное. Это правило сильнее, чем «ответить любой "
            "ценой». "
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

# AiohttpSession с увеличенным таймаутом: amvera иногда держит сетевые задержки
# дольше дефолта aiogram (30с), после чего вылетает 'Request timeout error' и
# роняет весь процесс. 120с дают запросу дойти.
_session = AiohttpSession(timeout=120)
bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    session=_session,
)
dp = Dispatcher()


@dp.error()
async def on_error(event: ErrorEvent):
    """Глобальный перехват ошибок: сетевой таймаут Telegram не должен ронять бот.

    aiogram 3 передаёт ОДИН ErrorEvent (update + exception в нём), а не два
    позиционных аргумента. TelegramNetworkError ('Request timeout error')
    возникает при задержках сети в amvera — логируем и глушим. Прочие
    исключения логируем с traceback, polling не рвём.
    """
    exception = event.exception
    update = event.update
    uid = update.update_id if update else "?"
    if isinstance(exception, TelegramNetworkError):
        log.warning("TelegramNetworkError on update %s (suppressed): %s", uid, exception)
        return True  # обработано, не падать
    log.exception("Unhandled error on update %s: %s", uid, exception)
    return True

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
    чтобы переключение /web сразу меняло инструкции модели. При выключении
    web-режима tool-связки фильтруются — иначе role=tool без tools= ломают
    запрос к модели.
    """
    prompt = build_system_prompt(web_on)
    msgs = history[chat_id]
    if not web_on and any(m.get("role") in ("tool",) or
                          (m.get("role") == "assistant" and m.get("tool_calls"))
                          for m in msgs):
        history[chat_id] = _strip_tool_messages(msgs)
        msgs = history[chat_id]
    if msgs and msgs[0].get("role") == "system":
        msgs[0]["content"] = prompt
    else:
        msgs.insert(0, {"role": "system", "content": prompt})


# Коды станций Яндекс.Расписаний для популярных городов России.
# Пополнить при необходимости через /v3.0/suggest/.
YANDEX_STATIONS = {
    "москва": "c213", "мск": "c213", "moscow": "c213",
    "санкт-петербург": "c2", "спб": "c2", "питер": "c2", "петербург": "c2",
    "санкт петербург": "c2",
    "новосибирск": "c54",
    "екатеринбург": "c207",
    "нижний новгород": "c35",
    "казань": "c86",
    "самара": "c51",
    "омск": "c292",
    "челябинск": "c208",
    "ростов-на-дону": "c40", "ростов на дону": "c40", "ростов": "c40",
    "уфа": "c294",
    "красноярск": "c290",
    "пермь": "c210",
    "волгоград": "c194",
    "воронеж": "c61",
    "краснодар": "c270",
    "сочи": "c142",
    "тюмень": "c296",
    "ярославль": "c108",
    "мурманск": "c282",
    "кондопога": "s9603093",
    "петрозаводск": "s9603421",
    "тверь": "c78",
    "балашиха": "c5888465",
    "подольск": "c5874515",
}


def _normalize_city(word: str) -> str:
    """Сводит слово к словарной форме:剥 типичные падежные окончания.

    Москва/Москвы/Москве/Москву → москва, Казань/Казани → казан,
    Питера/Питере → питер, Уфа/Уфы/Уфе → уф. Корень минимум 3 символа.
    """
    w = word.lower()
    for suf in ("ами", "ях", "ев", "ов", "ах", "ой", "ей", "ом", "ем",
                "е", "ы", "у", "ю", "а", "я", "и", "ь"):
        #剥 только если корень останется >= 3 символов; для совсем коротких
        # (уфа/уфы) опускаем порог до 2, иначе падеж не свернётся.
        limit = 2 if len(w) <= 4 else 3
        if len(w) - len(suf) >= limit and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _find_station_codes(text: str) -> tuple[str, str] | None:
    """Пытается найти в тексте запроса пару городов → коды станций.

    Возвращает (from_code, to_code) или None. Берём первые два распознанных
    города по порядку в тексте. Поддержка падежей (через剥 окончаний),
    составных названий («санкт-петербург», «нижний новгород») и коротких имён.
    """
    t = (text or "").lower()
    norm = " " + t + " "

    # (позиция, код): сканируем словарь, приоритет — составным (длинным) ключам.
    found: list[tuple[int, str]] = []
    # Сортируем ключи по убыванию длины, чтобы «санкт-петербург» матчился
    # раньше «петербург», а «ростов-на-дону» — раньше «ростов».
    items = sorted(YANDEX_STATIONS.items(), key=lambda kv: -len(kv[0]))
    matched_spans: list[tuple[int, int]] = []  # (start, end) занятых позиций

    for name, code in items:
        # 1) Составные/с дефисом — ищем подстрокой (нормализуем дефисы).
        needle = name.replace("-", " ")
        hay = norm.replace("-", " ")
        idx = hay.find(needle)
        while idx != -1:
            start, end = idx, idx + len(needle)
            if not any(not (end <= s or start >= e) for s, e in matched_spans):
                matched_spans.append((start, end))
                found.append((start, code))
                break
            idx = hay.find(needle, idx + 1)

    # 2) Одиночные слова (в т.ч. короткие) через剥 окончаний.
    tokens = [(m.start(), m.group()) for m in re.finditer(r"[а-яёa-z]+", t)]
    for pos, tok in tokens:
        # Пропускаем токены, уже покрытые составным матчем.
        if any(s <= pos < e for s, e in matched_spans):
            continue
        norm_tok = _normalize_city(tok)
        for name, code in items:
            if " " in name or "-" in name:
                continue  # составные уже обработаны выше
            norm_name = _normalize_city(name)
            if norm_tok == norm_name and norm_name:
                found.append((pos, code))
                break

    if len(found) < 2:
        return None
    found.sort()  # по позиции в тексте
    # Берём два разных кода (город может встретиться дважды).
    codes = []
    for _, code in found:
        if code not in codes:
            codes.append(code)
        if len(codes) == 2:
            break
    return (codes[0], codes[1]) if len(codes) == 2 else None


def _format_yandex_rasp(data: dict) -> str:
    """Превращает JSON ответ Яндекс.Расписаний в человекочитаемый текст."""
    segments = data.get("segments") or []
    if not segments:
        return ""
    lines = []
    for s in segments[:8]:  # первые 8 рейсов, чтобы не раздувать контекст
        thread = s.get("thread", {}) or {}
        dep_raw = s.get("departure") or ""
        arr_raw = s.get("arrival") or ""
        dep = dep_raw[11:16] if len(dep_raw) >= 16 else "—"
        arr = arr_raw[11:16] if len(arr_raw) >= 16 else "—"
        number = thread.get("number", "")
        title = thread.get("title", "")
        ttype = {"train": "поезд", "suburban": "электричка",
                 "bus": "автобус", "plane": "самолёт"}.get(
            thread.get("transport_type"), thread.get("transport_type", ""))
        dur = s.get("duration")
        dur_h = f"{dur // 3600:.0f}ч {dur % 3600 // 60:02d}м" if dur else ""
        price = ""
        # Современный формат: prices.whole.price. Фоллбэк: tickets_info[].price.
        prices = s.get("prices") or {}
        if isinstance(prices, dict):
            pwhole = prices.get("whole") or {}
            if isinstance(pwhole, dict) and pwhole.get("price"):
                price = f", от {pwhole['price']}₽"
        if not price:
            ti = s.get("tickets_info") or []
            for t in ti:
                tp = (t.get("price") or {}) if isinstance(t, dict) else {}
                place = tp.get("price") if isinstance(tp, dict) else None
                if place:
                    price = f", от {place}₽"
                    break
        lines.append(
            f"• {ttype} {number} {title}: отпр {dep} → приб {arr}"
            f"{(' (' + dur_h + ')') if dur_h else ''}{price}"
        )
    return "Реальное расписание (Яндекс.Расписания API):\n" + "\n".join(lines)


_MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
    "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11,
    "декабря": 12,
    "янв": 1, "фев": 2, "мар": 3, "апр": 4, "июн": 6, "июл": 7, "авг": 8,
    "сен": 9, "окт": 10, "ноя": 11, "дек": 12,
}


def _extract_date(text: str) -> str:
    """Достаёт дату из запроса в формате YYYY-MM-DD, иначе пусто.

    Понимает числовые форматы (17.08.2026, 17/08/26), текстовые («17 августа»)
    и относительные («завтра», «послезавтра»). Невалидные даты (99.99) → ''.
    """
    t = (text or "").lower()

    # Относительные даты.
    if "послезавтра" in t:
        return (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
    if "завтра" in t:
        return (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    if "сегодня" in t:
        return datetime.now().strftime("%Y-%m-%d")

    # Числовой формат DD.MM.YYYY.
    m = re.search(r"(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{2,4})", t)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
        if len(y) == 2:
            y = "20" + y
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y}-{mo:02d}-{d:02d}"
        return ""

    # Текстовый формат: «17 августа», «3 авг».
    m = re.search(r"(\d{1,2})\s+([а-яё]+)", t)
    if m:
        mo = _MONTHS.get(m.group(2))
        if mo:
            d = int(m.group(1))
            if 1 <= d <= 31:
                return f"{datetime.now().year}-{mo:02d}-{d:02d}"
    return ""


async def _yandex_rasp_search(text: str) -> str:
    """Прямой запрос к Яндекс.Расписаниям для поездов между станциями.

    Возвращает '' если интеграция выключена (нет ключа), не распознаны города
    или нет рейсов. Все причины логируются на INFO, чтобы в логах было видно
    yandex=yes/no и почему. Вызов идёт из _web_search с fallback на DDG.
    """
    if not YANDEX_RASP_KEY:
        log.info("yandex=no (no YANDEX_RASP_KEY)")
        return ""
    codes = _find_station_codes(text)
    if not codes:
        log.info("yandex=no (cities not recognized in %r)", text)
        return ""
    from_code, to_code = codes
    date = _extract_date(text)
    params = {
        "apikey": YANDEX_RASP_KEY,
        "from": from_code,
        "to": to_code,
        "format": "json",
        "lang": "ru_RU",
        "transport_types": "train,suburban",
    }
    if date:
        params["date"] = date
    log.info("yandex API: from=%s to=%s date=%s", from_code, to_code, date or "(today)")

    # Перебираем домены; у каждого свой независимый таймаут WEB_TIMEOUT,
    # чтобы зависший первый домен не съел бюджет второго.
    async with aiohttp.ClientSession() as session:
        for url in YANDEX_RASP_URLS:
            try:
                # wait_for — это корутина, а не async-context-manager:
                # берём await, затем response используем как контекст.
                resp = await asyncio.wait_for(
                    session.get(url, params=params),
                    timeout=WEB_TIMEOUT,
                )
                async with resp:
                    if resp.status != 200:
                        log.warning("yandex rasp %s -> status %s", url, resp.status)
                        continue
                    data = await resp.json(content_type=None)
                    # API может вернуть 200 с ошибкой в теле.
                    if data.get("error"):
                        log.warning("yandex rasp error body: %s", data.get("error"))
                        continue
                    out = _format_yandex_rasp(data)
                    if out:
                        log.info("yandex=yes (%d chars via %s)", len(out), url)
                        return out
                    log.info("yandex=no (200 OK but no segments)")
                    return ""
            except (asyncio.TimeoutError, Exception) as e:
                log.debug("yandex rasp %s exc: %s", url, e)
                continue
    log.info("yandex=no (all domains failed)")
    return ""


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
        # lxml быстрее, но если его нет — падает FeatureNotFound; fallback на
        # встроенный html.parser, чтобы глубокий парсинг не отключался молча.
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            soup = BeautifulSoup(html, "html.parser")
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


def _answer_from_snippets(query: str, search_result: str) -> str:
    """Собирает человекочитаемый ответ из сниппетов, если модель не справилась
    (упорно шлёт теги <tool_call>). Показываем найденное, а не «Не удалось».
    """
    if not search_result or "недоступна" in search_result or "ничего не найдено" in search_result:
        return (
            "К сожалению, подробных данных по этому запросу найти не удалось.\n"
            "Для точного расписания/билетов проверьте:\n"
            "• Яндекс.Расписания — rasp.yandex.ru\n"
            "• Tutu.ru — tutu.ru\n"
            "• РЖД — rzd.ru"
        )
    # Обрезаем слишком длинную простыню сниппетов — оставляем читаемое.
    text = search_result
    if len(text) > TG_MSG_LIMIT:
        text = text[:TG_MSG_LIMIT].rsplit("\n", 1)[0] + "\n…"
    return f"Вот что удалось найти по запросу «{query}»:\n\n{text}"


async def _web_search(query: str) -> str:
    """Веб-поиск + (опц.) реальное расписание + глубокий парсинг.

    Порядок источников (каждый факультативен, на сбое — fallback дальше):
    0) Яндекс.Расписания API: если есть ключ и распознались два города —
       получаем точные поезда/время/цены. Это первичный источник для расписаний.
    1) Упрощаем запрос (_simplify_query): убираем даты в кавычках и лишние слова.
    2) ddgs в фоне (to_thread) под wait_for(WEB_TIMEOUT).
    3) Если среди ссылок есть сайты расписаний/билетов (WEB_DEEP_HOSTS),
       тянем и парсим первую такую страницу.
    Все сетевые операции — неблокирующие.
    """
    # 0) Реальное расписание из Яндекс.Расписаний (если настроен ключ).
    yandex_text = ""
    try:
        yandex_text = await asyncio.wait_for(
            _yandex_rasp_search(query), timeout=WEB_TIMEOUT
        )
    except (asyncio.TimeoutError, Exception) as e:
        log.debug("yandex rasp timed out: %s", e)
    if yandex_text:
        log.info("web_search: yandex rasp data found (%d chars)", len(yandex_text))

    # 1-3) Обычный веб-поиск + глубокий парсинг.
    simple = _simplify_query(query)
    log.info("web_search: original=%r simplified=%r yandex=%s",
             query, simple, "yes" if yandex_text else "no")
    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(_ddg_search, simple or query),
            timeout=WEB_TIMEOUT,
        )
    except (asyncio.TimeoutError, Exception) as e:
        log.error("Search failed or timed out: %s", e)
        # Если есть данные Яндекс.Расписаний — отдаём хотя бы их.
        return yandex_text or "Информация из поиска недоступна."

    # Если упрощённый запрос ничего не дал — пробуем исходный (на всякий).
    if not results and simple and simple != query:
        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(_ddg_search, query),
                timeout=WEB_TIMEOUT,
            )
        except (asyncio.TimeoutError, Exception):
            results = []

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

    snippets = _format_search_results(results, deep) if results else ""
    if yandex_text and snippets:
        return yandex_text + "\n\n---\n\nДополнительно из веб-поиска:\n" + snippets
    if yandex_text:
        return yandex_text
    return snippets or "Информация из поиска недоступна."


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
    # Под тем же локом, что и фоновая _process_message — иначе /clear режет
    # историю прямо под идущей обработкой ответа (race).
    async with chat_locks[message.chat.id]:
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
    task.add_done_callback(_on_bg_task_done)


def _on_bg_task_done(task: asyncio.Task) -> None:
    """Чистим задачу из набора и логируем потерянное исключение.

    Фоновые задачи выпадают из dp.error(), поэтому без явного извлечения
    исключение утекло бы с «Task exception was never retrieved».
    """
    _bg_tasks.discard(task)
    exc = task.exception()
    if exc is not None:
        log.error("Фоновая задача упала: %s", exc, exc_info=exc)


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
    """Держит system + последние сообщения в пределах HISTORY_LIMIT*2.

    Транзакционная обрезка: не оставляем role=tool без предшествующего
    assistant(tool_calls) — иначе API вернёт 400 на следующем запросе.
    """
    msgs = history[chat_id]
    sys_part = msgs[:1] if msgs and msgs[0].get("role") == "system" else []
    rest = msgs[1:] if sys_part else msgs[:]
    limit = HISTORY_LIMIT * 2
    if len(rest) <= limit:
        return
    keep = rest[-limit:]
    # Если срез начинается с role=tool — отрезаем «висячие» tool-сообщения,
    # у которых нет парного assistant(tool_calls) в начале keep.
    while keep and keep[0].get("role") == "tool":
        keep = keep[1:]
    history[chat_id] = sys_part + keep


def _strip_tool_messages(msgs: list[dict]) -> list[dict]:
    """Удаляет tool-связки (assistant с tool_calls + последующие role=tool).

    Нужно при выключении web-режима: role=tool / tool_calls в истории без
    объявленных tools= ломают запрос к модели.
    """
    out: list[dict] = []
    for m in msgs:
        if m.get("role") == "tool":
            continue
        if m.get("role") == "assistant" and m.get("tool_calls"):
            continue
        out.append(m)
    return out


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
        # Поиск не нужен. Контент уже готов — отдадим его напрямую, без
        # повторной генерации через _stream_answer (иначе платим дважды и
        # рискуем рассинхронизацией ответов).
        content = (choice.message.content or "").strip()
        if content:
            history[chat_id].append({"role": "assistant", "content": content})
            await _safe_edit_fallback(placeholder, content[:TG_MSG_LIMIT])
            for c in [content[i:i + TG_MSG_LIMIT] for i in range(TG_MSG_LIMIT, len(content), TG_MSG_LIMIT)]:
                try:
                    await _send_fallback(chat_id, c)
                except Exception:
                    log.exception("send_message chunk failed")
            return None  # ответ уже отправлен, _stream_answer не нужен
        log.info("no tool_calls and no content; finish_reason=%s", choice.finish_reason)
        return []

    # Сохраняем ассистент-сообщение с tool_calls, как требует схема OpenAI.
    history[chat_id].append(choice.message.model_dump(exclude_none=True))

    for call in choice.message.tool_calls:
        if call.function.name != "web_search":
            # Чужая функция — кладём заглушку role:tool, иначе API ждёт ответ
            # для этого tool_call_id и следующий запрос упадёт с 400.
            history[chat_id].append(
                {"role": "tool", "tool_call_id": call.id,
                 "content": "не поддерживается"}
            )
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
    if query:
        # Упрощаем запрос: модель часто формирует перегруженный
        # ('"маршрут" 17.08.2026 билеты'), от которого DDG пуст.
        simple_q = _simplify_query(query) or query
        if _tool_attempts < MAX_TOOL_ATTEMPTS:
            log.info("text tool_call intercepted (attempt %d), query=%r -> %r",
                     _tool_attempts + 1, query, simple_q)
            await _safe_edit(placeholder, f"🔍 Ищу: {simple_q[:100]}…")
            result = await _web_search(simple_q)
            # ВАЖНО: в историю кладём НЕ сырой <tool_call> (он учит модель
            # плохому), а чистую ассистент-пометку + результат поиска.
            history[chat_id].append({"role": "assistant", "content": "[вызов web_search]"})
            history[chat_id].append(
                {
                    "role": "user",
                    "content": (
                        f"Результаты веб-поиска по запросу «{simple_q}»:\n{result}\n\n"
                        "На основе этих результатов дай пользователю краткий "
                        "человеческий ответ обычным текстом. НЕ используй теги "
                        "<tool_call> и не вызывай функцию снова — просто ответь."
                    ),
                }
            )
            # Рекурсивный вызов — теперь модель должна ответить нормально.
            await _stream_answer(chat_id, model, placeholder, _tool_attempts + 1)
            return
        # Лимит перехватов исчерпан: модель упорно шлёт теги. Не сдаёмся с
        # «Не удалось» — сами ищем и собираем ответ из сниппетов.
        log.warning("tool_call attempts exhausted, building answer from search")
        await _safe_edit(placeholder, f"🔍 Ищу: {simple_q[:100]}…")
        result = await _web_search(simple_q)
        full = _answer_from_snippets(simple_q, result)
        history[chat_id].append({"role": "assistant", "content": full})
    else:
        # Чистый человеческий ответ — обычный путь. Но если в стриме был
        # замечен <tool_call>, а парсер не вытащил запрос (тег оборвался /
        # нет <arg_value>), не отдавать юзеру сырую разметку: вырезаем всё
        # с <tool_call> до конца и при пустом остатке ищем сами.
        if started_tool_tag:
            cut = full.split("<tool_call>")[0].strip()
            if cut:
                full = cut
            else:
                last_q = message_text(chat_id) or ""
                if last_q:
                    log.warning("incomplete tool_call, fallback search by user msg")
                    await _safe_edit(placeholder, "🔍 Ищу…")
                    result = await _web_search(last_q)
                    full = _answer_from_snippets(_simplify_query(last_q), result)
                else:
                    full = "Не удалось получить ответ."
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


def _simplify_query(query: str) -> str:
    """Упрощает поисковый запрос для DuckDuckGo.

    Модель часто формирует перегруженный запрос с точными датами в кавычках
    ('"поезд Санкт-Петербург Кондопога" 17.08.2026 расписание билеты'),
    из-за чего DDG ничего не находит. Убираем кавычки, даты, лишние слова и
    приводим к простому виду: 'поезд Санкт Петербург Кондопога расписание'.
    """
    q = (query or "").strip()
    # Убираем кавычки (любые).
    q = q.replace("«", " ").replace("»", " ").replace('"', " ").replace("'", " ")
    # Убираем даты: 17.08.2026, 17-08-2026, 2026-08-17, 17 августа и т.п.
    q = re.sub(r"\b\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}\b", " ", q)
    q = re.sub(r"\b\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2}\b", " ", q)
    q = re.sub(r"\b\d{1,2}\s+(?:янв|фев|мар|апр|ма[йя]|июн|июл|авг|сен|окт|ноя|дек)\w*", " ", q, flags=re.IGNORECASE)
    # Убираем года отдельно.
    q = re.sub(r"\b(19|20)\d{2}\b", " ", q)
    # Лишние служебные слова, которые мешают поиску.
    q = re.sub(r"\b(билеты?|купить|цена|стоимость|заказать)\b", " ", q, flags=re.IGNORECASE)
    # Схлопываем повторы пробелов.
    q = re.sub(r"\s+", " ", q).strip()
    # Отрезаем висячие предлоги/союзы в конце ('... на', '... и').
    q = re.sub(r"\s+(на|и|или|с|от|до|в|к|по|для)$", "", q, flags=re.IGNORECASE)
    return q


async def _safe_edit(message: Message, text: str):
    """Редактируем сообщение plain-текстом, игнорируя 'not modified' и rate-limit.

    Явный parse_mode=None: ответы модели и сниппеты поиска содержат сырые
    <>[]_*, которые ломают HTML/Markdown-парсинг Telegram (TelegramBadRequest),
    и плашка «зависает». На сетевых ошибках — пара ретраев с паузой.
    """
    for attempt in range(3):
        try:
            await bot.edit_message_text(
                text=text,
                chat_id=message.chat.id,
                message_id=message.message_id,
                parse_mode=None,
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
    """Финальная правка ответа plain-текстом.

    Plain (parse_mode=None) единообразно со стримом: сырые символы ссылок/HTML
    больше не ломают рендер. Совпадает с поведением _safe_edit.
    """
    try:
        await message.edit_text(text, parse_mode=None)
    except Exception as e:
        log.debug("plain edit failed: %s", e)


async def _send_fallback(chat_id: int, text: str):
    """Досылка доп. чанков plain-текстом."""
    try:
        await bot.send_message(chat_id, text, parse_mode=None)
    except Exception as e:
        log.debug("send failed: %s", e)


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
    log.info("ENV YANDEX_RASP_KEY=%s", "set" if YANDEX_RASP_KEY else "not set")
    log.info("ENV ALLOWED_CHAT_IDS=%s", sorted(ALLOWED_IDS))
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    # Рекомендация amvera: задержка против конфликта getUpdates при рестарте.
    time.sleep(2)
    asyncio.run(main())
