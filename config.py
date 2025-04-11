# config.py
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Загружаем переменные окружения из файла .env
# Ищет .env в текущей директории или родительских
load_dotenv()

# --- Секреты --- 
# Получаем значения из переменных окружения
TELEGRAM_API_ID = os.getenv("TELEGRAM_API_ID")
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH")
TELEGRAM_PHONE_NUMBER = os.getenv("TELEGRAM_PHONE_NUMBER")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
POSTGRES_ASYNC_URL = os.getenv("POSTGRES_ASYNC_URL")

# --- Проверки наличия переменных --- 
# Важно убедиться, что все необходимые переменные загружены
required_env_vars = [
    "TELEGRAM_API_ID",
    "TELEGRAM_API_HASH",
    "TELEGRAM_PHONE_NUMBER",
    "BOT_TOKEN",
    "GEMINI_API_KEY",
    "POSTGRES_ASYNC_URL",
]

missing_vars = [var for var in required_env_vars if not globals().get(var)]
if missing_vars:
    raise ValueError(f"Отсутствуют переменные окружения: {', '.join(missing_vars)}. Убедитесь, что файл .env существует и заполнен.")

# Попытка преобразовать API ID в int (добавлено из-за возможных ошибок)
try:
    TELEGRAM_API_ID = int(TELEGRAM_API_ID)
except (ValueError, TypeError):
    raise ValueError("Переменная окружения TELEGRAM_API_ID должна быть числом.")

# --- Другие настройки (если есть) --- 
# Например, можно оставить здесь несекретные настройки
ADMIN_TELEGRAM_ID = 123456789 # Пример

# --- Вывод для проверки (можно убрать в production) ---
print("Конфигурация загружена:")
print(f"  BOT_TOKEN: {'*' * (len(BOT_TOKEN) - 4) + BOT_TOKEN[-4:] if BOT_TOKEN else 'Не найден'}")
print(f"  GEMINI_API_KEY: {'*' * (len(GEMINI_API_KEY) - 4) + GEMINI_API_KEY[-4:] if GEMINI_API_KEY else 'Не найден'}")
print(f"  POSTGRES_ASYNC_URL: {POSTGRES_ASYNC_URL is not None}")

# Database settings
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASS = "12345"

# Формируем строки подключения
POSTGRES_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
POSTGRES_ASYNC_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Остальные настройки
#BOT_TOKEN = os.getenv("BOT_TOKEN")
#PHONE_NUMBER = os.getenv("PHONE_NUMBER")
PHONE_NUMBER = "+79254323035"
#GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY = "AIzaSyAR3IRvu_WIrMPfbnyL5wyhcgXBW2UCGcU"
GEMINI_EMBEDDING_MODEL_NAME = "models/text-embedding-004"

# Дополнительные настройки (если понадобятся)
# DEBUG_MODE = os.getenv("DEBUG_MODE", False)

genai.configure(api_key=GOOGLE_API_KEY)