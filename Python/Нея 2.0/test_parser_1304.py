from telethon import TelegramClient, events
import asyncio
import csv
from typing import List, Dict
import os
from datetime import datetime
import random
from telethon.tl.types import InputPeerChannel
from telethon.tl.functions.messages import GetHistoryRequest

class TelegramParser:
    def __init__(self):
        self.api_id = 24520702
        self.api_hash = '4873fd31ae3a9a93f77fdea2e88ef738'
        self.phone = '+79254323035'
        session_name = f'session_{random.randint(1, 1000000)}'
        self.client = TelegramClient(session_name, self.api_id, self.api_hash)
        self.chunk_size = 100  # Размер пачки сообщений
        self.max_concurrent_chats = 3  # Количество одновременно обрабатываемых чатов
        
    async def parse_chat(self, chat_id: int) -> List[Dict]:
        messages_data = []
        offset_id = 0
        limit = self.chunk_size
        
        channel = await self.client.get_entity(chat_id)
        
        while True:
            history = await self.client(GetHistoryRequest(
                peer=channel,
                offset_id=offset_id,
                offset_date=None,
                add_offset=0,
                limit=limit,
                max_id=0,
                min_id=0,
                hash=0
            ))
            
            if not history.messages:
                break
                
            messages = history.messages
            for message in messages:
                if not message.text:
                    continue
                    
                data = {
                    'message_id': message.id,
                    'date': message.date.strftime('%Y-%m-%d %H:%M:%S'),
                    'text': message.text,
                    'from_id': message.from_id.user_id if message.from_id else None,
                    'reply_to_msg_id': message.reply_to_msg_id,
                    'forwards': message.forwards
                }
                messages_data.append(data)
            
            offset_id = messages[-1].id
            
        return messages_data

    async def process_chats(self, chat_ids: List[int]):
        # Разбиваем чаты на группы для параллельной обработки
        chunks = [chat_ids[i:i + self.max_concurrent_chats] 
                 for i in range(0, len(chat_ids), self.max_concurrent_chats)]
        
        for chunk in chunks:
            # Создаем задачи для каждого чата в группе
            tasks = [self.process_single_chat(chat_id) for chat_id in chunk]
            # Запускаем их параллельно
            await asyncio.gather(*tasks)

    async def process_single_chat(self, chat_id: int):
        try:
            data = await self.parse_chat(chat_id)
            if data:
                await self.save_to_csv(chat_id, data)
                print(f'Успешно обработан чат {chat_id}')
            else:
                print(f'Нет данных для сохранения из чата {chat_id}')
        except Exception as e:
            print(f'Ошибка при обработке чата {chat_id}: {str(e)}')

    # ... остальные методы остаются без изменений ...