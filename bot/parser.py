from telethon import TelegramClient, events
from telethon.tl.types import Channel
import asyncio
import csv
from datetime import datetime
import os
from telethon.errors import FloodWaitError, ServerError
from telethon.errors.rpcerrorlist import PeerIdInvalidError
from requests.exceptions import ConnectionError
from database.requests import DatabaseRequests
from sqlalchemy import insert, select
from database.models import ChannelPosts
from typing import List
import logging # Добавляем логирование
# Импортируем переменные из config
from config import TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_PHONE_NUMBER

# Предполагаем, что используется spaCy, импортируем тип для type hinting
import spacy

logger = logging.getLogger(__name__) # Настраиваем логгер для парсера

class TelegramParser():
    def __init__(self, db, nlp_model: spacy.language.Language, client=None):
        # Используем значения из config
        self.api_id = TELEGRAM_API_ID
        self.api_hash = TELEGRAM_API_HASH
        self.phone_number = TELEGRAM_PHONE_NUMBER
        # Используем переданный клиент или создаем новый
        self.client = client or TelegramClient('session_name', self.api_id, self.api_hash)
        # Добавляем флаг, указывающий, был ли клиент предоставлен извне
        self.client_provided = client is not None
        self.db = db
        self.nlp_model = nlp_model # Сохраняем NLP модель
        
        # Добавляем семафор для ограничения конкурентных запросов
        self.semaphore = asyncio.Semaphore(3)  # Максимум 3 одновременных запроса
        
        # Добавляем счетчики для статистики
        self.total_requests = 0
        self.failed_requests = 0
        self.retry_delay = 30  # Задержка между повторными попытками в секундах
        self.max_retries = 80   # Максимальное количество попыток
        
        self.is_parsing = False  # Флаг состояния парсера
        
    async def init(self):
        # Если клиент был предоставлен извне, пропускаем его инициализацию
        if not self.client_provided:
            # Инициализируем клиент только если он не был предоставлен извне
            await self.client.start()
            if not await self.client.is_user_authorized():
                await self.client.send_code_request(self.phone_number)
                try:
                    print("Пожалуйста, введите код подтверждения, отправленный в Telegram:")
                    code = input()
                    await self.client.sign_in(self.phone_number, code)
                except Exception as e:
                    print(f"Ошибка при входе: {str(e)}")
                    return False
        else:
            # Проверяем, что предоставленный клиент авторизован
            if not await self.client.is_user_authorized():
                print("Ошибка: Предоставленный клиент Telethon не авторизован")
                return False

        # Загружаем каналы и ID последних сообщений из БД
        if self.db:
            self.last_message_ids = await self.db.get_all_channels_with_last_messages()
            print(f"Загружены ID последних сообщений для {len(self.last_message_ids)} каналов")
        
        return True

    async def _make_request_with_retry(self, func, *args, **kwargs):
        """Выполняет запрос с повторными попытками при ошибках"""
        for attempt in range(self.max_retries):
            try:
                async with self.semaphore:  # Используем семафор для ограничения конкурентности
                    self.total_requests += 1
                    return await func(*args, **kwargs)
            except FloodWaitError as e:
                print(f"Превышен лимит запросов, ожидаем {e.seconds} секунд")
                await asyncio.sleep(e.seconds)
            except (ServerError, ConnectionError) as e:
                self.failed_requests += 1
                wait_time = self.retry_delay * (attempt + 1)
                print(f"Ошибка сервера: {e}. Повторная попытка через {wait_time} секунд")
                await asyncio.sleep(wait_time)
            except Exception as e:
                # Проверяем, является ли это ошибкой доступа к приватному каналу
                if "The channel specified is private and you lack permission to access it" in str(e):
                    # Это специальный случай, пропускаем канал
                    self.failed_requests += 1
                    print(f"Канал недоступен (приватный или заблокирован): {args[0] if args else 'неизвестный'}")
                    # Генерируем специальное исключение, которое будет перехвачено выше
                    raise ValueError("PrivateChannelError") from e
                else:
                    # Для других ошибок сохраняем прежнее поведение
                    self.failed_requests += 1
                    print(f"Неожиданная ошибка: {e}")
                    if attempt == self.max_retries - 1:
                        raise
                    await asyncio.sleep(self.retry_delay)
        raise Exception("Превышено максимальное количество попыток")

    async def _parse_single_channel(self, channel_id, limit):
        try:
            print(f"\nПопытка парсинга канала с ID: {channel_id}")
            
            # Проверяем статус канала в БД
            channel = await self.db.get_channel_by_tg_id(channel_id)
            if channel and channel.is_unavailable:
                print(f"Канал {channel_id} уже помечен как недоступный в БД, пропускаем")
                return []
            
            # Пробуем получить канал через ссылку, если она есть
            if channel and channel.telegram_link:
                print(f"Пробуем получить канал через ссылку: {channel.telegram_link}")
                channel_entity = await self.client.get_entity(channel.telegram_link)
            else:
                print(f"У канала {channel_id} нет ссылки в БД, пробуем получить через ID")
                channel_entity = await self.client.get_entity(channel_id)
            
            if not isinstance(channel_entity, Channel):
                print(f"ID {channel_id} не является каналом, помечаем как недоступный")
                if channel:
                    channel.is_unavailable = True
                    await self.db.session().commit()
                return []
            
            messages = []
            last_id = self.last_message_ids.get(channel_id)
            
            async for message in self.client.iter_messages(channel_entity, limit=limit):
                if last_id and message.id <= last_id:
                    break
                messages.append(message)
            
            print(f"Собрано {len(messages)} новых сообщений из канала {channel_entity.title}")
            return messages
            
        except Exception as e:
            print(f"Ошибка при парсинге канала {channel_id}: {e}")
            if channel:
                channel.is_unavailable = True
                await self.db.session().commit()
            return []

    async def parse_channels(self, limit_per_channel=1000000):
        all_messages = []
        
        # Создаем список задач с ограничением конкурентности
        semaphore = asyncio.Semaphore(3)  # Максимум 3 канала одновременно
        tasks = []
        
        for channel_id in self.last_message_ids.keys():
            task = asyncio.create_task(self._parse_channel_with_semaphore(
                channel_id, 
                limit_per_channel, 
                semaphore
            ))
            tasks.append(task)
        
        # Выполняем все задачи и собираем результаты
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Обрабатываем результаты и ошибки
        for result in results:
            if isinstance(result, Exception):
                print(f"Ошибка при парсинге: {result}")
            elif isinstance(result, list):
                all_messages.extend(result)
        
        # Выводим статистику
        print(f"\nСтатистика парсинга:")
        print(f"Всего запросов: {self.total_requests}")
        print(f"Неудачных запросов: {self.failed_requests}")
        print(f"Успешность: {((self.total_requests - self.failed_requests) / self.total_requests * 100):.2f}%")
        
        return all_messages

    def _sync_extract_lemmas(self, text: str) -> List[str]:
        """Синхронная функция для извлечения лемм с помощью spaCy."""
        if not text or not self.nlp_model:
            return []
        
        try:
            doc = self.nlp_model(text)
            lemmas = [
                token.lemma_.lower()
                for token in doc
                if not token.is_stop
                and not token.is_punct
                and not token.is_space
                and len(token.lemma_) > 2
                and token.pos_ in {'NOUN', 'PROPN', 'ADJ', 'VERB'}
            ]
            return list(set(lemmas))
        except Exception as e:
            # Логируем ошибку синхронной части
            logger.error(f"Sync Error in _sync_extract_lemmas for text: '{text[:50]}...': {e}", exc_info=True)
            return []

    async def _extract_lemmas(self, text: str) -> List[str]:
        """Асинхронно извлекает леммы, запуская CPU-bound spaCy в потоке."""
        if not text or not self.nlp_model:
            return []
        try:
            # Запускаем синхронную CPU-bound функцию в отдельном потоке
            lemmas = await asyncio.to_thread(self._sync_extract_lemmas, text)
            return lemmas
        except Exception as e:
            # Логируем ошибку асинхронной обертки
            logger.error(f"Async Error in _extract_lemmas wrapper for text: '{text[:50]}...': {e}", exc_info=True)
            return []

    async def _parse_channel_with_semaphore(self, channel_id, limit, semaphore):
        """Обёртка для парсинга канала с использованием семафора"""
        async with semaphore:
            try:
                messages = await self._parse_single_channel(channel_id, limit)
                if messages:
                    saved_count = await self.save_to_db(messages)
                    print(f"Сохранено {saved_count} сообщений из канала {channel_id} в базу данных")
                    
                    # --- Начало: Удаление извлечения сущностей ---
                    # Убираем извлечение сущностей отсюда
                    # --- Конец: Удаление извлечения сущностей ---
                    
                return messages
            except Exception as e:
                print(f"Ошибка при парсинге канала {channel_id}: {e}")
                return []

    async def run(self):
        try:
            self.is_parsing = True  # Устанавливаем флаг в начале работы
            print("Начинаем парсинг каналов...")
            if not self.last_message_ids:
                print("Нет каналов для парсинга с указанными последними ID")
                return
                
            total_saved = 0
            messages = await self.parse_channels()
            
            # Убираем сохранение в БД здесь, так как оно теперь происходит для каждого канала
            total_saved = sum(len(msg_list) for msg_list in messages if isinstance(msg_list, list))
            print(f"Парсинг завершен. Собрано и сохранено {total_saved} новых сообщений.")
            
        except Exception as e:
            print(f"Произошла ошибка: {str(e)}")
        finally:
            self.is_parsing = False  # Сбрасываем флаг при завершении

    def _message_to_db_format(self, message):
        """Преобразует объект Message Telethon в формат для БД"""
        
        # Конвертируем дату из "offset-aware" в "offset-naive" datetime
        msg_date = None
        if hasattr(message, 'date') and message.date:
            # Приводим datetime с timezone к datetime без timezone
            # для совместимости с типом TIMESTAMP WITHOUT TIME ZONE
            msg_date = message.date.replace(tzinfo=None)
        
        # Получаем ID канала
        peer_id = message.peer_id.channel_id if hasattr(message, 'peer_id') else \
                 (message.chat.id if hasattr(message, 'chat') and message.chat else None)
        
        # Формируем ссылку на сообщение
        message_link = None
        if peer_id:
            # Для чатов по ID - используем стандартный формат
            message_link = f"https://t.me/c/{peer_id}/{message.id}"
            
            # Для каналов с username - проверяем есть ли username
            if hasattr(message.chat, 'username') and message.chat.username:
                message_link = f"https://t.me/{message.chat.username}/{message.id}"
        
        return {
            "message_id": message.id,  # ID сообщения
            "peer_id": peer_id,
            "date": msg_date,  # datetime уже без timezone
            "message": message.message if hasattr(message, 'message') else None,  # текст сообщения
            "views": message.views if hasattr(message, 'views') else 0,
            "forwards": message.forwards if hasattr(message, 'forwards') else 0,
            "post_author": message.post_author if hasattr(message, 'post_author') else None,
            "embedding": None,
            "key_words": None,
            "message_link": message_link  # Добавляем ссылку на сообщение
        }

    async def save_to_db(self, messages):
        """Сохраняет сообщения в БД без использования цикла for"""
        
        # Используем функциональный подход map
        db_objects = list(map(self._message_to_db_format, messages))
        
        # Один запрос для массовой вставки
        async with self.db.session() as session:
            await session.execute(insert(ChannelPosts), db_objects)
            await session.commit()
        
        return len(db_objects)

async def main(db):
    # При вызове main нужно будет передать nlp_model
    # Пример: nlp = spacy.load("ru_core_news_md")
    # parser = TelegramParser(db, nlp)
    parser = TelegramParser(db) # Пока оставляем так, т.к. main не используется напрямую в боте
    if await parser.init():
        await parser.run()
    else:
        print("Не удалось авторизоваться")

if __name__ == '__main__':
    asyncio.run(main())

