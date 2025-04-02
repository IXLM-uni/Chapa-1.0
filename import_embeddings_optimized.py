import asyncio
import json
import os
import sys
import time
import gc
import ijson
import psycopg2
import psycopg2.extras
import tempfile
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from database.models import ChannelPosts
from config import POSTGRES_ASYNC_URL, POSTGRES_URL

# Получаем количество ядер процессора для параллельной обработки
CPU_COUNT = max(1, multiprocessing.cpu_count() - 1)

# Размер пакета для обработки
BATCH_SIZE = 1000

# Размер чанка файла для параллельной обработки (в байтах)
CHUNK_SIZE = 100 * 1024 * 1024  # 100 МБ

def extract_db_params(url):
    """Извлекает параметры подключения к БД из URL"""
    # Пример URL: postgresql://username:password@hostname:port/database
    if url.startswith('postgresql://'):
        url = url[len('postgresql://'):]
    elif url.startswith('postgresql+asyncpg://'):
        url = url[len('postgresql+asyncpg://'):]
    
    auth, rest = url.split('@', 1)
    host_port, dbname = rest.split('/', 1)
    
    if ':' in auth:
        user, password = auth.split(':', 1)
    else:
        user, password = auth, ''
    
    if ':' in host_port:
        host, port = host_port.split(':', 1)
        port = int(port)
    else:
        host, port = host_port, 5432
    
    return {
        'dbname': dbname,
        'user': user,
        'password': password,
        'host': host,
        'port': port
    }

def process_json_chunk(file_path, start_pos, end_pos, temp_file_path):
    """
    Обрабатывает часть JSON-файла и сохраняет результаты во временный файл.
    
    Args:
        file_path: Путь к исходному JSON-файлу
        start_pos: Начальная позиция в файле
        end_pos: Конечная позиция в файле
        temp_file_path: Путь к временному файлу для результатов
        
    Returns:
        int: Количество обработанных записей
    """
    processed = 0
    
    try:
        # Открываем исходный файл и перемещаемся к начальной позиции
        with open(file_path, 'r', encoding='utf-8') as f:
            f.seek(start_pos)
            
            # Читаем данные до конечной позиции
            data = f.read(end_pos - start_pos)
            
            # Добавляем фигурные скобки для создания валидного JSON
            if not data.startswith('{'):
                data = '{' + data
            if not data.endswith('}'):
                data = data + '}'
            
            # Парсим JSON
            try:
                chunk_data = json.loads(data)
            except json.JSONDecodeError:
                # Если не удалось распарсить, пробуем найти последнюю полную запись
                last_valid_pos = data.rfind('"}')
                if last_valid_pos > 0:
                    data = data[:last_valid_pos+2] + '}'
                    chunk_data = json.loads(data)
                else:
                    return 0
            
            # Открываем временный файл для записи результатов
            with open(temp_file_path, 'w', encoding='utf-8') as temp_f:
                # Записываем данные в формате CSV: id,embedding
                for message_id, item_data in chunk_data.items():
                    if isinstance(item_data, dict) and "embedding" in item_data:
                        embedding = item_data.get("embedding")
                        if embedding and isinstance(embedding, list):
                            # Записываем ID и эмбеддинг в CSV-формате
                            temp_f.write(f"{message_id}\t{json.dumps(embedding)}\n")
                            processed += 1
    
    except Exception as e:
        print(f"Ошибка при обработке чанка: {e}")
        import traceback
        traceback.print_exc()
    
    return processed

async def import_embeddings_parallel(input_file, db_params):
    """
    Импортирует эмбеддинги из JSON-файла в базу данных с использованием параллельной обработки.
    
    Args:
        input_file: Путь к входному JSON-файлу
        db_params: Параметры подключения к базе данных
    """
    start_time = time.time()
    total_processed = 0
    
    try:
        # Получаем размер файла
        file_size = os.path.getsize(input_file)
        print(f"Размер файла: {file_size / (1024 * 1024):.2f} МБ")
        
        # Определяем количество чанков
        num_chunks = max(1, min(CPU_COUNT, file_size // CHUNK_SIZE))
        chunk_size = file_size // num_chunks
        
        print(f"Разделение файла на {num_chunks} частей для параллельной обработки")
        
        # Создаем временные файлы для каждого чанка
        temp_files = [tempfile.NamedTemporaryFile(delete=False, suffix='.csv') for _ in range(num_chunks)]
        temp_file_paths = [f.name for f in temp_files]
        for f in temp_files:
            f.close()
        
        # Определяем границы чанков
        chunk_boundaries = []
        for i in range(num_chunks):
            start_pos = i * chunk_size
            end_pos = file_size if i == num_chunks - 1 else (i + 1) * chunk_size
            chunk_boundaries.append((start_pos, end_pos))
        
        # Обрабатываем чанки параллельно
        print(f"Начинаем параллельную обработку файла с использованием {num_chunks} процессов...")
        
        with ProcessPoolExecutor(max_workers=num_chunks) as executor:
            # Запускаем задачи на обработку чанков
            futures = [
                executor.submit(
                    process_json_chunk, 
                    input_file, 
                    start_pos, 
                    end_pos, 
                    temp_file_paths[i]
                )
                for i, (start_pos, end_pos) in enumerate(chunk_boundaries)
            ]
            
            # Получаем результаты
            for future in futures:
                processed = future.result()
                total_processed += processed
        
        print(f"Параллельная обработка завершена. Обработано {total_processed} записей.")
        
        # Импортируем данные из временных файлов в базу данных
        print("Начинаем импорт данных в базу данных...")
        
        # Используем COPY для быстрой загрузки данных
        conn = psycopg2.connect(**db_params)
        conn.autocommit = False
        
        try:
            with conn.cursor() as cursor:
                # Создаем временную таблицу
                cursor.execute("""
                    CREATE TEMP TABLE temp_embeddings (
                        message_id INTEGER,
                        embedding JSONB
                    )
                """)
                
                # Загружаем данные из временных файлов
                for temp_file in temp_file_paths:
                    if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                        with open(temp_file, 'r', encoding='utf-8') as f:
                            cursor.copy_from(
                                f, 
                                'temp_embeddings', 
                                columns=('message_id', 'embedding')
                            )
                
                # Обновляем основную таблицу
                cursor.execute("""
                    UPDATE channel_posts
                    SET embedding = temp.embedding::jsonb
                    FROM temp_embeddings temp
                    WHERE channel_posts.id = temp.message_id
                """)
                
                # Получаем количество обновленных строк
                cursor.execute("SELECT COUNT(*) FROM temp_embeddings")
                total_rows = cursor.fetchone()[0]
                
                # Фиксируем транзакцию
                conn.commit()
                
                print(f"Импорт завершен. Обновлено {total_rows} записей.")
        
        except Exception as e:
            conn.rollback()
            print(f"Ошибка при импорте данных: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            conn.close()
        
        # Удаляем временные файлы
        for temp_file in temp_file_paths:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
        # Выводим статистику
        total_time = time.time() - start_time
        print(f"\nИмпорт завершен за {total_time:.2f} секунд")
        print(f"Средняя скорость: {total_processed / total_time:.2f} записей/сек")
    
    except Exception as e:
        print(f"Ошибка при импорте данных: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Основная функция для запуска импорта"""
    # Проверяем наличие необходимых библиотек
    try:
        import ijson
        import psycopg2
        import psycopg2.extras
    except ImportError as e:
        missing_lib = str(e).split("'")[1]
        print(f"Ошибка: библиотека {missing_lib} не установлена.")
        print(f"Установите ее с помощью команды: pip install {missing_lib}")
        return
    
    # Путь к входному файлу
    input_file = "exports/texts_with_embeddings.json"
    
    # Проверяем существование файла
    if not os.path.exists(input_file):
        print(f"Ошибка: файл {input_file} не найден")
        return
    
    # Получаем параметры подключения к БД
    db_params = extract_db_params(POSTGRES_URL)
    
    # Запускаем импорт
    await import_embeddings_parallel(input_file, db_params)

if __name__ == "__main__":
    asyncio.run(main()) 