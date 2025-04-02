import os
import sys
import asyncio
import logging
from dotenv import load_dotenv
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import pandas as pd
from typing import List, Dict, Any, Optional
import json

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

# Настройки БД
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASS = "12345"
POSTGRES_ASYNC_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

class DatabaseTester:
    def __init__(self, db_url: str):
        self.engine = create_async_engine(db_url)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
    
    async def list_tables(self):
        """Получает список всех таблиц в базе данных"""
        logger.info("Получение списка таблиц...")
        
        async with self.async_session() as session:
            query = text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                ORDER BY table_name;
            """)
            result = await session.execute(query)
            tables = result.all()
            
            print("\n=== Таблицы в базе данных ===")
            for i, (table,) in enumerate(tables):
                print(f"{i+1}. {table}")
            
            return [table[0] for table in tables]
    
    async def examine_table_structure(self, table_name: str):
        """Изучает структуру указанной таблицы"""
        logger.info(f"Изучение структуры таблицы {table_name}...")
        
        async with self.async_session() as session:
            query = text(f"""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position;
            """)
            result = await session.execute(query)
            columns = result.all()
            
            print(f"\n=== Структура таблицы '{table_name}' ===")
            for i, (column, data_type, is_nullable) in enumerate(columns):
                print(f"{i+1}. {column} ({data_type}) {'NULL' if is_nullable == 'YES' else 'NOT NULL'}")
            
            return columns
    
    async def check_table_data(self, table_name: str, limit: int = 5):
        """Проверяет данные в таблице"""
        logger.info(f"Проверка данных в таблице {table_name}...")
        
        async with self.async_session() as session:
            query = text(f"SELECT * FROM {table_name} LIMIT {limit}")
            result = await session.execute(query)
            rows = result.all()
            
            print(f"\n=== Данные в таблице '{table_name}' (первые {limit} строк) ===")
            
            if not rows:
                print("Таблица пуста")
                return []
            
            # Получаем имена колонок
            keys = result.keys()
            
            # Создаем DataFrame для красивого вывода
            df = pd.DataFrame(rows, columns=keys)
            print(df)
            
            # Для детального изучения выведем текстовые поля
            print("\n=== Детальный просмотр текстовых полей ===")
            for i, row in enumerate(rows):
                print(f"\nСтрока {i+1}:")
                for j, key in enumerate(keys):
                    value = row[j]
                    if isinstance(value, str) and len(value) > 50:
                        print(f"  {key}: {value[:50]}...")
                    elif key == 'embedding' and value:
                        if isinstance(value, list):
                            print(f"  {key}: [размерность: {len(value)}]")
                        else:
                            print(f"  {key}: {type(value)} (не список)")
                    else:
                        print(f"  {key}: {value}")
            
            return rows
    
    async def check_embeddings(self, table_name: str):
        """Проверяет наличие и формат эмбеддингов в таблице"""
        logger.info(f"Проверка эмбеддингов в таблице {table_name}...")
        
        try:
            async with self.async_session() as session:
                # Сначала проверим, есть ли колонка embedding
                query = text(f"""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = '{table_name}' AND column_name = 'embedding';
                """)
                result = await session.execute(query)
                if not result.first():
                    print(f"\nВ таблице '{table_name}' нет колонки 'embedding'")
                    return
                
                # Проверим количество записей с эмбеддингами
                query = text(f"SELECT COUNT(*) FROM {table_name} WHERE embedding IS NOT NULL")
                result = await session.execute(query)
                count = result.scalar()
                print(f"\nКоличество записей с эмбеддингами: {count}")
                
                if count == 0:
                    print("Эмбеддинги отсутствуют в таблице")
                    return
                
                # Получим несколько записей с эмбеддингами
                query = text(f"""
                    SELECT id, embedding 
                    FROM {table_name} 
                    WHERE embedding IS NOT NULL 
                    LIMIT 3
                """)
                result = await session.execute(query)
                rows = result.all()
                
                print("\n=== Примеры эмбеддингов ===")
                for i, (id, embedding) in enumerate(rows):
                    embedding_type = type(embedding)
                    embedding_len = len(embedding) if embedding else 0
                    print(f"ID: {id}, Тип эмбеддинга: {embedding_type}, Размерность: {embedding_len}")
                    
                    # Проверка наличия нулевых значений
                    if embedding and embedding_len > 0:
                        zeros = sum(1 for x in embedding if x == 0)
                        print(f"  Количество нулевых значений: {zeros} ({zeros/embedding_len:.2%})")
                        
                        non_zero_sample = [x for x in embedding[:10] if x != 0]
                        if non_zero_sample:
                            print(f"  Примеры ненулевых значений: {non_zero_sample[:3]}")
                
                # Проверка разной размерности
                query = text(f"""
                    SELECT id, array_length(embedding, 1) as emb_len
                    FROM {table_name}
                    WHERE embedding IS NOT NULL
                    GROUP BY id, array_length(embedding, 1)
                    ORDER BY emb_len DESC
                    LIMIT 10
                """)
                result = await session.execute(query)
                dimensions = result.all()
                
                if dimensions:
                    print("\n=== Проверка размерностей эмбеддингов ===")
                    for id, dim in dimensions:
                        print(f"ID: {id}, Размерность: {dim}")
                
        except Exception as e:
            logger.error(f"Ошибка при проверке эмбеддингов: {e}")
            import traceback
            traceback.print_exc()
    
    async def test_message_retrieval(self, table_name: str, message_ids: List[int]):
        """Тестирует получение сообщений по ID"""
        logger.info(f"Тестирование получения сообщений из {table_name} по ID...")
        
        try:
            async with self.async_session() as session:
                for msg_id in message_ids:
                    query = text(f"""
                        SELECT * FROM {table_name} WHERE id = {msg_id}
                    """)
                    result = await session.execute(query)
                    row = result.first()
                    
                    print(f"\n=== Сообщение с ID {msg_id} ===")
                    if not row:
                        print(f"Сообщение с ID {msg_id} не найдено")
                        continue
                    
                    # Получаем имена колонок
                    keys = result.keys()
                    
                    # Выводим данные
                    for i, key in enumerate(keys):
                        value = row[i]
                        if key == 'embedding' and value:
                            if isinstance(value, list):
                                print(f"{key}: [размерность: {len(value)}]")
                            else:
                                print(f"{key}: {type(value)} (не список)")
                        elif isinstance(value, str) and len(value) > 100:
                            print(f"{key}: {value[:100]}...")
                        else:
                            print(f"{key}: {value}")
        
        except Exception as e:
            logger.error(f"Ошибка при получении сообщений: {e}")
            import traceback
            traceback.print_exc()

async def main():
    print("=== Тест базы данных и FAISS индекса ===")
    
    tester = DatabaseTester(POSTGRES_ASYNC_URL)
    
    # 1. Получаем список таблиц
    tables = await tester.list_tables()
    
    # 2. Изучаем структуру предполагаемых таблиц с сообщениями
    message_tables = [table for table in tables if 'message' in table or 'post' in table or 'channel' in table]
    
    if not message_tables:
        print("\nНе найдено таблиц, которые могут содержать сообщения.")
        table_to_check = input("Введите имя таблицы для проверки: ")
        if table_to_check:
            message_tables = [table_to_check]
    
    # 3. Проверяем структуру и данные для найденных таблиц
    for table in message_tables:
        await tester.examine_table_structure(table)
        await tester.check_table_data(table)
        await tester.check_embeddings(table)
    
    # 4. Тестируем получение сообщений
    chosen_table = message_tables[0] if message_tables else None
    
    if chosen_table:
        print(f"\nВыбрана таблица '{chosen_table}' для тестирования получения сообщений")
        
        # Получим несколько ID для тестирования
        async with tester.async_session() as session:
            query = text(f"""
                SELECT id FROM {chosen_table} 
                WHERE embedding IS NOT NULL 
                LIMIT 5
            """)
            result = await session.execute(query)
            ids = [row[0] for row in result.all()]
            
            if ids:
                await tester.test_message_retrieval(chosen_table, ids)
            else:
                print("Не найдено сообщений с эмбеддингами для тестирования")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nТест прерван пользователем")
    except Exception as e:
        logger.error(f"Непредвиденная ошибка: {e}")
        import traceback
        traceback.print_exc() 