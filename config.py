# config.py
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

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
BOT_TOKEN = "7690195228:AAFemqlhmZ1v0lpr8znmsfVakVCzGJYi9wg"
API_ID = os.getenv("API_ID")
API_HASH = os.getenv("API_HASH")
#PHONE_NUMBER = os.getenv("PHONE_NUMBER")
PHONE_NUMBER = "+79254323035"
#GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY = "AIzaSyAR3IRvu_WIrMPfbnyL5wyhcgXBW2UCGcU"
GEMINI_EMBEDDING_MODEL_NAME = "models/text-embedding-004"

# Дополнительные настройки (если понадобятся)
# DEBUG_MODE = os.getenv("DEBUG_MODE", False)

genai.configure(api_key=GOOGLE_API_KEY)