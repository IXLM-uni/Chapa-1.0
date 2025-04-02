import asyncio
import logging
import os
import sys
from aiogram import Bot, Dispatcher, Router, F
from aiogram.fsm.storage.memory import MemoryStorage
from langchain_community.chat_message_histories import SQLChatMessageHistory
from datetime import datetime, timedelta
import numpy as np
from sqlalchemy import select

# Импортируем handler-ы
from bot.handlers import start_handlers, channel_add_handler, channel_delete_handler, chat_handlers
from bot.middlewares.user_middleware import UserMiddleware
from database.models import Base, ChannelPosts
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from config import POSTGRES_ASYNC_URL, BOT_TOKEN
from database.requests import DatabaseRequests
from bot.llm_processor import SimpleGeminiProcessor, set_db_instance
from bot.parser import TelegramParser, main as parser_main
from telethon import TelegramClient
from database.requests import DatabaseRequests
from services.text_processing import TextProcessor
from services.vector_search import FaissIndexManager, init_globals as vs_init_globals

# Настройка логирования
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# Создаем engine для базы данных
engine = create_async_engine(
    url=POSTGRES_ASYNC_URL,
    echo=False,
    pool_size=100,
    max_overflow=200
)

# Создаем фабрику сессий
async_session = async_sessionmaker(engine, expire_on_commit=False)

# Закрытие соединений при завершении
async def close_db_connection():
    await engine.dispose()

# Импортируем необходимые библиотеки для LangChain FAISS
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings as LCEmbeddings
from langchain_core.documents import Document

# Создаем класс-адаптер для совместимости с LangChain
class SentenceTransformerEmbeddings(LCEmbeddings):
    def __init__(self, model):
        self.model = model
    
    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()
    
    def embed_query(self, text):
        return self.model.encode(text, normalize_embeddings=True).tolist()

async def run(dp):
    """Автоматически добавляет заданные каналы для пользователя"""
    # Получаем необходимые объекты из диспетчера
    db = dp["db"]
    client = dp["client"]
    
    # ID пользователя
    telegram_id = 1310172407
    
    # Список каналов для добавления
    channels_to_add = [
        {"url": "https://t.me/renatageorge"},
        {"url": "https://t.me/boris_again"},
        {"url": "https://t.me/seeallochnaya"},
        {"url": "https://t.me/ai_newz"},
        {"url": "https://t.me/htech_plus"},
        {"url": "https://t.me/sakhalinonetwo"},
        {"url": "https://t.me/MyHomeKidss"},
        {"url": "https://t.me/codecamp"},
        {"url": "https://t.me/technodeus2023"},
        {"url": "https://t.me/cgplugin"},
        {"url": "https://t.me/wtfont"},
        {"url": "https://t.me/RKadyrov_95"},
        {"url": "https://t.me/artgallery"},
        {"url": "https://t.me/darkmgimo"},
        {"url": "https://t.me/people_theatre"},
        {"url": "https://t.me/vs_court"},
        {"url": "https://t.me/voenpravoru"},
        {"url": "https://t.me/minobrnaukiofficial"},
        {"url": "https://t.me/pfccskamoscow"},
        {"url": "https://t.me/theyseeku"},
        {"url": "https://t.me/DrButriy"},
        {"url": "https://t.me/DrAnshina"},
        {"url": "https://t.me/prostoy_retsept"},
        {"url": "https://t.me/Kulinaria_Receptii"},
        {"url": "https://t.me/dachaogrod"},
        {"url": "https://t.me/National_Travell"},
        {"url": "https://t.me/mudotalida"},
        {"url": "https://t.me/yandexmusic_live"},
        {"url": "https://t.me/cybers"},
        {"url": "https://t.me/kinopoisk"},
        {"url": "https://t.me/blumcrypto"},
        {"url": "https://t.me/fintopionews"},
        {"url": "https://t.me/investingcorp"},
        {"url": "https://t.me/startups"},
        {"url": "https://t.me/sberbusiness"},
        {"url": "https://t.me/poltorapomidora"},
        {"url": "https://t.me/mashablinovafit"},
        {"url": "https://t.me/Alinamorozovaofficial"}
    ]
    
    # Добавляем каналы по очереди
    for channel in channels_to_add:
        success = await db.add_channel(
            channel_id=channel["url"],
            client=client,
            telegram_id=telegram_id,
        )
        if success:
            print(f"Канал {channel['url']} успешно добавлен для пользователя {telegram_id}")
        else:
            print(f"Не удалось добавить канал {channel['url']} для пользователя {telegram_id}")
    
    print("Автоматическое добавление каналов завершено")

async def main():
    # Создаем клиент Telethon с корректными настройками (как в парсере)
    client = TelegramClient('session_name', 24520702, '4873fd31ae3a9a93f77fdea2e88ef738')
    phone_number = '+79254323035'
    
    # Инициализация клиента по правильному алгоритму из парсера
    await client.start()
    if not await client.is_user_authorized():
        await client.send_code_request(phone_number)
        try:
            print("Пожалуйста, введите код подтверждения, отправленный в Telegram:")
            code = input()
            await client.sign_in(phone_number, code)
        except Exception as e:
            print(f"Ошибка при входе: {str(e)}")
            return
    
    # Создаем экземпляр DatabaseRequests с передачей session_maker
    db = DatabaseRequests(session_maker=async_session)
    bot = Bot(token=BOT_TOKEN)
    llm_processor = SimpleGeminiProcessor(bot=bot)
    
    dp = Dispatcher(storage=MemoryStorage())
    
    # Сохраняем инициализированный клиент и другие объекты в диспетчере
    dp["client"] = client
    dp["db"] = db
    
    # Загружаем модели NLP и эмбеддингов один раз и сохраняем в диспетчере
    import spacy
    from sentence_transformers import SentenceTransformer
    
    print("Загрузка моделей NLP и эмбеддингов...")
    nlp_model = spacy.load("ru_core_news_md")
    embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')
    print("Модели успешно загружены!")
    
    # Добавляем модели в диспетчер
    dp["nlp_model"] = nlp_model
    dp["embedding_model"] = embedding_model
    
    # Создаем и инициализируем FAISS индекс
    vector_search = FaissIndexManager(omp_threads=4)
    dp["vector_search"] = vector_search
    vs_init_globals(vs=vector_search, emb_model=embedding_model)

    # Передаем нужные объекты в диспетчер
    dp["llm_processor"] = llm_processor
    
    # СНАЧАЛА вызываем run() для добавления каналов перед запуском парсера
    #await run(dp)
    
    # Создаем парсер, передавая ему УЖЕ ИНИЦИАЛИЗИРОВАННЫЙ клиент
    parser = TelegramParser(db=db, client=client)
    
    # Инициализируем парсер без повторной инициализации клиента
    if await parser.init():
        print("Парсер Telegram инициализирован, запускаем...")
        # Просто запускаем парсер в фоновом режиме
        asyncio.create_task(parser.run())
        
        # Получаем эмбеддинги напрямую из базы данных
        print("Получение эмбеддингов для создания FAISS индекса...")
        message_embeddings = await db.get_all_message_embeddings()
        
        # Добавьте подробное логирование
        if message_embeddings:
            print(f"Загружено {len(message_embeddings)} эмбеддингов, создаем FAISS индекс...")
            # Используем менеджер для создания индекса
            success = await vector_search.create_index(message_embeddings)
            if success:
                print("FAISS индекс успешно создан!")
            else:
                print("ОШИБКА: Не удалось создать FAISS индекс!")
        else:
            print("ПРЕДУПРЕЖДЕНИЕ: Нет эмбеддингов для создания FAISS индекса на старте.")
        
    else:
        print("Не удалось инициализировать парсер Telegram")
        
    db.set_parser(parser)
    
    # Запускаем обновление метаданных каналов
    print("Запуск проверки и обновления метаданных каналов...")
    # Обновляем только каналы с пустыми метаданными, максимум 10 каналов за раз
    #asyncio.create_task(db.update_all_empty_channel_metadata(force_update=False, max_channels=10))
    
    # Затем регистрируем роутеры и запускаем бота
    dp.include_router(start_handlers.router)
    dp.include_router(channel_add_handler.router)
    dp.include_router(channel_delete_handler.router)
    dp.include_router(chat_handlers.router)

    # После создания DatabaseRequests
    set_db_instance(db)  # Передаем готовый экземпляр

    try:
        await dp.start_polling(bot)
    except Exception as e:
        logging.error(f"Возникла ошибка: {e}")
        import traceback
        print(traceback.format_exc())
    finally:
        # Закрываем соединения с БД
        await close_db_connection()
        await bot.session.close()

# Вызываем функцию после инициализации text_processor
async def initialize_langchain_faiss(dp, text_processor):
    try:
        # Получаем все сообщения с эмбеддингами
        message_embeddings = await text_processor.get_all_message_embeddings()
        
        if not message_embeddings:
            logging.warning("Нет эмбеддингов для создания LangChain FAISS индекса")
            return False
            
        # Адаптируем нашу модель эмбеддингов для LangChain
        lc_embeddings = SentenceTransformerEmbeddings(dp["embedding_model"])
        
        # Получаем тексты документов из базы данных
        documents = []
        db = dp["db"]
        
        async with db.session() as session:
            for msg_id, _ in message_embeddings:
                query = select(ChannelPosts).where(ChannelPosts.id == msg_id)
                result = await session.execute(query)
                post = result.scalars().first()
                
                if post:
                    channel_info = f"(канал: {post.peer_id})" if post.peer_id else ""
                    documents.append(
                        Document(
                            page_content=post.message,
                            metadata={
                                "id": post.id,
                                "message_id": post.message_id,
                                "date": post.date.strftime("%Y-%m-%d %H:%M") if post.date else "неизвестно",
                                "channel": channel_info
                            }
                        )
                    )
        
        if not documents:
            logging.warning("Не удалось получить документы для индексации")
            return False
            
        # Создаем индекс FAISS через LangChain
        #faiss_index = await FAISS.afrom_documents(documents, lc_embeddings)
        
        # Сохраняем индекс в диспетчере
        #dp["langchain_faiss"] = faiss_index
        logging.info(f"LangChain FAISS индекс успешно создан с {len(documents)} документами")
        return True
        
    except Exception as e:
        logging.error(f"Ошибка при создании LangChain FAISS индекса: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    asyncio.run(main())