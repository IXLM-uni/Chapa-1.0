from telethon import TelegramClient, events
from telethon.tl.types import Channel
import asyncio
import csv
from datetime import datetime
import os
from telethon.errors import FloodWaitError, ServerError
from requests.exceptions import ConnectionError
from database.requests import DatabaseRequests
from sqlalchemy import insert
from database.models import ChannelPosts

class TelegramParser():
    def __init__(self, db, client=None):
        # Данные из .env файла
        self.api_id = 24520702
        self.api_hash = '4873fd31ae3a9a93f77fdea2e88ef738'
        self.phone_number = '+79254323035'
        # Используем переданный клиент или создаем новый
        self.client = client or TelegramClient('session_name', self.api_id, self.api_hash)
        # Добавляем флаг, указывающий, был ли клиент предоставлен извне
        self.client_provided = client is not None
        self.db = db
        
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
            try:
                channel = await self._make_request_with_retry(
                    self.client.get_entity,
                    channel_id
                )
            except ValueError as e:
                # Проверяем, является ли это нашей специальной ошибкой приватного канала
                if str(e) == "PrivateChannelError":
                    print(f"Канал {channel_id} приватный или недоступен. Пропускаем.")
                    return []
                else:
                    raise  # Пробрасываем другие ошибки ValueError
            except Exception as e:
                # Для других типов ошибок
                raise
            
            messages = []
            last_id = self.last_message_ids.get(channel_id)
            
            async for message in self.client.iter_messages(channel, limit=limit):
                try:
                    if last_id and message.id <= last_id:
                        break
                    
                    # Просто добавляем сообщение, игнорируя медиа
                    messages.append(message)
                    
                except Exception as msg_error:
                    print(f"Ошибка при обработке сообщения {message.id}: {msg_error}")
                    continue
                
            print(f"Собрано {len(messages)} новых сообщений из канала {channel.title}")
            return messages
            
        except Exception as e:
            print(f"Ошибка при парсинге канала {channel_id}: {str(e)}")
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

    async def _parse_channel_with_semaphore(self, channel_id, limit, semaphore):
        """Обёртка для парсинга канала с использованием семафора"""
        async with semaphore:
            try:
                messages = await self._parse_single_channel(channel_id, limit)
                if messages:
                    await self.save_to_db(messages)
                    print(f"Сохранено {len(messages)} сообщений из канала {channel_id} в базу данных")
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
    parser = TelegramParser(db)
    if await parser.init():
        await parser.run()
    else:
        print("Не удалось авторизоваться")

if __name__ == '__main__':
    asyncio.run(main())

