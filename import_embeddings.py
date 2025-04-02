import asyncio
import json
import os
import sys
import time
import gc
import ijson  # Добавляем библиотеку ijson для потоковой обработки JSON
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from database.models import ChannelPosts
from config import POSTGRES_ASYNC_URL

async def import_embeddings_from_json(input_file="exports/texts_with_embeddings.json", batch_size=2000):
    """
    Импортирует эмбеддинги из JSON-файла в базу данных, обрабатывая файл по частям.
    
    Args:
        input_file (str): Путь к входному JSON-файлу с эмбеддингами.
        batch_size (int): Размер пакета для обновления базы данных за один раз.
    """
    print(f"Начинаем импорт эмбеддингов из файла {input_file}")
    
    # Проверяем существование файла
    if not os.path.exists(input_file):
        print(f"Ошибка: файл {input_file} не найден")
        return
    
    # Создаем engine для базы данных
    engine = create_async_engine(
        url=POSTGRES_ASYNC_URL,
        echo=False,
        pool_size=10,
        max_overflow=20
    )
    
    # Создаем фабрику сессий
    async_session = async_sessionmaker(engine, expire_on_commit=False)
    
    try:
        # Начинаем отсчет времени
        start_time = time.time()
        
        # Подготавливаем счетчики
        processed = 0
        updated = 0
        not_found = 0
        current_batch = []
        
        # Открываем файл для потоковой обработки
        print(f"Начинаем потоковую обработку файла {input_file}...")
        
        # Используем контекстный менеджер для файла
        with open(input_file, 'r', encoding='utf-8') as f:
            # Получаем итератор по парам ключ-значение в корневом объекте JSON
            parser = ijson.kvitems(f, '')
            
            for message_id, item_data in parser:
                # Проверяем наличие эмбеддинга
                if isinstance(item_data, dict) and "embedding" in item_data:
                    embedding = item_data.get("embedding")
                    if embedding and isinstance(embedding, list):
                        # Добавляем в текущий пакет
                        current_batch.append((message_id, embedding))
                
                # Если пакет заполнен, обрабатываем его
                if len(current_batch) >= batch_size:
                    await process_batch(current_batch, async_session)
                    
                    # Обновляем счетчики
                    batch_updated, batch_not_found = await process_batch(current_batch, async_session)
                    updated += batch_updated
                    not_found += batch_not_found
                    processed += len(current_batch)
                    
                    # Выводим прогресс
                    elapsed = time.time() - start_time
                    records_per_sec = processed / elapsed if elapsed > 0 else 0
                    
                    print(f"Обработано: {processed}, "
                          f"Обновлено: {updated}, Не найдено: {not_found}, "
                          f"Скорость: {records_per_sec:.2f} записей/сек")
                    
                    # Очищаем пакет и освобождаем память
                    current_batch = []
                    gc.collect()
        
        # Обрабатываем оставшиеся записи
        if current_batch:
            batch_updated, batch_not_found = await process_batch(current_batch, async_session)
            updated += batch_updated
            not_found += batch_not_found
            processed += len(current_batch)
        
        # Выводим итоговую статистику
        total_time = time.time() - start_time
        print(f"\nИмпорт завершен за {total_time:.2f} секунд")
        print(f"Всего обработано: {processed}")
        print(f"Успешно обновлено: {updated}")
        print(f"Не найдено в БД: {not_found}")
        
    except Exception as e:
        print(f"Ошибка при импорте данных: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Закрываем соединение с базой данных
        await engine.dispose()

async def process_batch(batch, async_session):
    """
    Обрабатывает пакет записей и обновляет базу данных.
    
    Args:
        batch: Список кортежей (message_id, embedding)
        async_session: Фабрика сессий SQLAlchemy
        
    Returns:
        tuple: (количество обновленных записей, количество не найденных записей)
    """
    updated = 0
    not_found = 0
    
    async with async_session() as session:
        for message_id, embedding in batch:
            try:
                # Находим запись в базе данных по message_id
                query = select(ChannelPosts).where(ChannelPosts.id == int(message_id))
                result = await session.execute(query)
                post = result.scalars().first()
                
                if post:
                    # Обновляем эмбеддинг
                    post.embedding = embedding
                    updated += 1
                else:
                    not_found += 1
            except ValueError:
                # Если message_id не может быть преобразован в int
                not_found += 1
                print(f"Ошибка: message_id '{message_id}' не является целым числом")
        
        # Сохраняем изменения
        await session.commit()
    
    return updated, not_found

async def main():
    """Основная функция для запуска импорта"""
    # Проверяем наличие библиотеки ijson
    try:
        import ijson
    except ImportError:
        print("Ошибка: библиотека ijson не установлена.")
        print("Установите ее с помощью команды: pip install ijson")
        return
    
    # Путь к входному файлу
    input_file = "exports/texts_with_embeddings.json"
    
    # Запускаем импорт с увеличенным размером пакета для больших файлов
    await import_embeddings_from_json(input_file=input_file, batch_size=2000)

if __name__ == "__main__":
    asyncio.run(main()) 