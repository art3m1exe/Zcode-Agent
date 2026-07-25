# GLM Telegram-бот для amvera.ru

Бот на Python (aiogram 3), который вечно крутится на **amvera.ru** и ходит в **GLM API (Z.ai)**. Пишешь боту из Telegram с телефона — ноутбук можно закрыть.

Память диалога в памяти процесса, доступ только по `chat_id`, стриминг длинных ответов по частям.

## Что нужно получить заранее

1. **BOT_TOKEN** — у [@BotFather](https://t.me/BotFather) в Telegram: `/newbot` → придумай имя и юзернейм (заканчивается на `bot`) → получишь токен вида `1234:abcd…`.
2. **GLM_API_KEY** — на [z.ai](https://z.ai): регистрация → API Keys → создать ключ. Формат `{id}.{secret}`. Международный хост `api.z.ai`.
3. **Твой chat_id** — напиши [@userinfobot](https://t.me/userinfobot) любое сообщение → он пришлёт твой `Id`.

## Локальный запуск (для проверки)

```bash
cd Zcode-Agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# заполни .env: BOT_TOKEN, GLM_API_KEY, ALLOWED_CHAT_IDS (твой chat_id)
python bot.py
```

Открой бота в Telegram, проверь:
- `/start` — приветствие
- обычное сообщение — ответ от GLM со стримингом
- `/clear` — сброс контекста
- `/model glm-4.5-flash` — смена модели

## Деплой в amvera.ru

Amvera поддерживает фоновые long-running процессы (бот на long-polling), публичный порт/домен **не нужен**. Конфиг деплоя уже описан в `amvera.yml`.

### Шаги

1. **Создай проект** на [amvera.ru](https://amvera.ru):
   - «Создать проект» → тип **Python**
   - Подключи GitHub/GitLab-репозиторий **или** загрузи файлы ZIP'ом (все файлы из этой папки: `bot.py`, `requirements.txt`, `amvera.yml`).
2. **Проверь конфиг** (`amvera.yml` уже в репо): окружение `python:3.11`, установка из `requirements.txt`, запуск `python bot.py`.
3. **Пропиши секреты** — вкладка **«Переменные»** в настройках проекта, каждая через «Добавить секрет»:

   | Ключ | Значение |
   |------|----------|
   | `BOT_TOKEN` | токен от BotFather |
   | `GLM_API_KEY` | ключ с z.ai |
   | `ALLOWED_CHAT_IDS` | твой chat_id (можно несколько через запятую) |
   | `GLM_MODEL` | `glm-5.2` (по умолчанию, GLM Coding Plan) |
   | `GLM_BASE_URL` | `https://api.z.ai/api/coding/paas/v4` |

   ⚠️ После добавления секретов — **перезапусти контейнер**.

4. **Запусти проект**. В логах должно появиться: `Bot started. Allowed chat_ids: [...]`.
5. Напиши боту из Telegram — должен ответить.

## Как это работает

- **Long-polling**, не webhook → домен/порт не требуется, работает из закрытого контейнера.
- Amvera **автоперезапускает** процесс при падении. В коде есть стартовая задержка 2 сек — против конфликта `getUpdates` при одновременном старте двух экземпляров в момент рестарта.
- Память диалога — **в памяти процесса** (первые 20 пар сообщений на chat_id). При рестарте контейнера история сбрасывается — это нормально для первой версии.
- Доступ ограничен списком `ALLOWED_CHAT_IDS`: чужие сообщения игнорируются, чтобы не тратить GLM-лимит.

## Структура

```
bot.py            # ядро бота (~180 строк)
requirements.txt  # aiogram, openai, python-dotenv
amvera.yml        # конфиг деплоя
.env.example      # шаблон переменных окружения
.gitignore
README.md
```

## Если что-то не работает

- В логах amvera нет строки `Bot started` → проверь, что `BOT_TOKEN` задан и валиден.
- Бот молчит в Telegram → проверь, что твой chat_id есть в `ALLOWED_CHAT_IDS`.
- Ошибка GLM в ответе → проверь `GLM_API_KEY` и баланс/лимиты на z.ai.
- Для китайского хоста `open.bigmodel.cn` поменяй `GLM_BASE_URL` на `https://open.bigmodel.cn/api/paas/v4`.
