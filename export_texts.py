import asyncio
import json
import os
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from database.models import ChannelPosts
from config import POSTGRES_ASYNC_URL

async def export_texts_to_json(output_file="texts_for_embedding.json", batch_size=100):
    """
    Экспортирует тексты без эмбеддингов из базы данных в JSON-файл.
    
    Args:
        output_file (str): Путь к выходному JSON-файлу.
        batch_size (int): Размер пакета для обработки за один раз.
    """
    print(f"Начинаем экспорт текстов без эмбеддингов в файл {output_file}")
    
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
        # Подготавливаем структуру данных для JSON
        result_data = {}
        total_exported = 0
        offset = 0
        
        async with async_session() as session:
            while True:
                # Получаем пакет сообщений без эмбеддингов
                query = select(ChannelPosts).where(
                    ChannelPosts.embedding == None
                ).limit(batch_size).offset(offset)
                
                result = await session.execute(query)
                posts = result.scalars().all()
                
                if not posts:
                    break  # Если больше нет сообщений, выходим из цикла
                
                # Обрабатываем полученные сообщения
                for post in posts:
                    if post.message and post.message.strip():  # Проверяем, что текст не пустой
                        # Добавляем запись в результат
                        result_data[post.id] = {
                            "text": post.message,
                            "embedding": []  # Пустой список для эмбеддингов
                        }
                
                # Увеличиваем смещение для следующего пакета
                offset += batch_size
                total_exported += len(posts)
                print(f"Обработано {total_exported} сообщений...")
        
        # Сохраняем результат в JSON-файл
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        print(f"Экспорт завершен. Всего экспортировано {len(result_data)} сообщений в файл {output_file}")
        
    except Exception as e:
        print(f"Ошибка при экспорте данных: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Закрываем соединение с базой данных
        await engine.dispose()

async def main():
    """Основная функция для запуска экспорта"""
    # Создаем директорию для экспорта, если она не существует
    os.makedirs("exports", exist_ok=True)
    
    # Путь к выходному файлу
    output_file = "exports/texts_for_embedding.json"
    
    # Запускаем экспорт
    await export_texts_to_json(output_file=output_file)

if __name__ == "__main__":
    asyncio.run(main()) 