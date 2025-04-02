from telethon import TelegramClient, events
from telethon.tl.types import Channel
import asyncio
import csv
from datetime import datetime
import os
from telethon.errors import FloodWaitError, ServerError
from requests.exceptions import ConnectionError

class TelegramParser:
    def __init__(self):
        # Данные из .env файла
        self.api_id = 24520702
        self.api_hash = '4873fd31ae3a9a93f77fdea2e88ef738'
        self.phone_number = '+79254323035'
        self.client = TelegramClient('session_name', self.api_id, self.api_hash)
        # Словарь с последними добавленными ID для каждого канала
        self.last_message_ids = {
            -1002091780361: 102,
            -1001764279964: 800,
            -1001993972957: 240,
            -1001511414765:None,
        }
        
        # Добавляем семафор для ограничения конкурентных запросов
        self.semaphore = asyncio.Semaphore(3)  # Максимум 3 одновременных запроса
        
        # Добавляем счетчики для статистики
        self.total_requests = 0
        self.failed_requests = 0
        self.retry_delay = 30  # Задержка между повторными попытками в секундах
        self.max_retries = 3   # Максимальное количество попыток
        
    async def init(self):
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
        return True

    async def save_to_csv(self, messages, filename='telegram_messages.csv', batch_size=1000):
        print(f"Начинаю сохранение {len(messages)} сообщений...")
        
        headers = ['date', 'message_id', 'reply_to_msg_id', 'channel_name', 'channel_id', 
                  'sender', 'sender_id', 'message', 'views', 'comments',
                  'channel_members_count', 'channel_url']
        
        saved_count = 0
        
        with open(filename, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, 
                                  fieldnames=headers, 
                                  delimiter=';',
                                  quoting=csv.QUOTE_ALL,
                                  quotechar='"',
                                  escapechar='\\')
            writer.writeheader()
            
            batch = []
            for message in messages:
                try:
                    clean_text = str(message.text).replace('\n', ' ').replace(';', '.') if message.text else 'MediaMessage'
                    
                    msg_data = {
                        'date': message.date.strftime('%Y-%m-%d %H:%M:%S') if message.date else '',
                        'message_id': str(message.id) if message.id else '',
                        'reply_to_msg_id': str(message.reply_to_msg_id) if message.reply_to_msg_id else '',
                        'channel_name': str(message.chat.title).replace(';', '.') if message.chat and message.chat.title else "Неизвестный канал",
                        'channel_id': str(message.chat.id) if message.chat else '',
                        'sender': str(message.sender.username).replace(';', '.') if message.sender and message.sender.username else '',
                        'sender_id': str(message.sender.id) if message.sender else '',
                        'message': clean_text,
                        'views': str(message.views) if message.views is not None else '0',
                        'comments': str(message.replies) if hasattr(message, 'replies') and message.replies else '0',
                        'channel_members_count': str(message.chat.participants_count) if message.chat and message.chat.participants_count else '0',
                        'channel_url': f"https://t.me/c/{str(message.chat.id)[4:]}/{message.id}" if message.chat else ''
                    }
                    
                    batch.append(msg_data)
                    saved_count += 1
                    
                    if len(batch) >= batch_size:
                        writer.writerows(batch)
                        print(f"Сохранено {saved_count} сообщений...")
                        batch = []
                        
                except Exception as e:
                    print(f"Ошибка при обработке сообщения: {str(e)}")
                    continue
            
            if batch:
                writer.writerows(batch)
        
        print(f"Сохранение завершено. Всего сохранено {saved_count} сообщений.")

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
                self.failed_requests += 1
                print(f"Неожиданная ошибка: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay)
        raise Exception("Превышено максимальное количество попыток")

    async def _parse_single_channel(self, channel_id, limit):
        try:
            channel = await self._make_request_with_retry(
                self.client.get_entity,
                channel_id
            )
            
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
                return await self._parse_single_channel(channel_id, limit)
            except Exception as e:
                print(f"Ошибка при парсинге канала {channel_id}: {e}")
                return []

    async def run(self):
        try:
            print("Начинаем парсинг каналов...")
            if not self.last_message_ids:
                print("Нет каналов для парсинга с указанными последними ID")
                return
                
            messages = await self.parse_channels()
            
            if messages:
                await self.save_to_csv(messages)
                print(f"Парсинг завершен. Собрано {len(messages)} новых сообщений.")
            else:
                print("Новых сообщений не найдено.")
            
        except Exception as e:
            print(f"Произошла ошибка: {str(e)}")
        finally:
            await self.client.disconnect()

async def main():
    parser = TelegramParser()
    if await parser.init():
        await parser.run()
    else:
        print("Не удалось авторизоваться")

if __name__ == '__main__':
    asyncio.run(main())
