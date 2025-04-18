import logging
from typing import List, Dict, Any, Optional
from sqlalchemy import select, Text, Column, Integer, BigInteger, DateTime, String, func, distinct
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import asyncio
import aiofiles
import gc
import traceback
import json
import uuid
from tqdm.asyncio import tqdm
from pydantic import BaseModel, Field

# Импортируем ключ API из конфига
from config import GEMINI_API_KEY, POSTGRES_ASYNC_URL
# Импортируем модели БД
from database.models import Base, ChannelPosts, Channels # Добавлен Channels

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()
class ChannelPosts(Base):
    __tablename__ = 'channel_messages'
    id = Column(Integer, primary_key=True)
    message_id = Column(BigInteger, index=True)
    peer_id = Column(BigInteger, index=True)
    date = Column(DateTime(timezone=True))
    message = Column(Text)

# --- Pydantic схема: ОДИН плоский список строк (теперь по 5 на сообщение) ---
class GeneratedQueries(BaseModel):
    """A flat list of generated search queries, 5 for each input message."""
    queries: List[str] = Field(
        description="FLAT list containing exactly 5 generated search queries for the first message, followed by 5 for the second, and so on."
    )
# --- Конец Pydantic схемы ---

class DataProcessor:
    # Возвращаем num_llms=1 для батчей, batch_size=10
    def __init__(self, num_llms: int = 100, batch_size: int = 10):
        logger.info(f"Инициализация DataProcessor: {num_llms} LLM, Размер батча: {batch_size} (обработка по каналам)")
        self.num_llms = num_llms
        self.channel_message_limit = 3000 # <<< Новый лимит на канал
        self.batch_size = batch_size # Количество сообщений в одном API вызове
        self.queries_per_message = 5 # <<< Изменено на 5
        
        logger.info("Подключение к базе данных...")
        self.engine = create_async_engine(POSTGRES_ASYNC_URL, echo=False)
        self.SessionLocal = async_sessionmaker(bind=self.engine, class_=AsyncSession)
        logger.info("Подключение к БД успешно установлено")
        
        logger.info("Инициализация Google Gemini LLM...")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.7, # Оставляем температуру для разнообразия
            google_api_key=GEMINI_API_KEY
        )
        logger.info("LLM успешно инициализирована")
        
        # Используем схему GeneratedQueries
        logger.info(f"Настройка LLM для структурированного вывода ({GeneratedQueries.__name__})...")
        try:
             self.structured_llm = self.llm.with_structured_output(GeneratedQueries)
        except Exception as e:
             logger.error(f"Ошибка при настройке structured_output: {e}.", exc_info=True)
             raise

        # Обновленный промпт: просим 5 запросов на сообщение, плоский список
        self.query_generation_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Ты — ассистент, создающий обучающие данные. Тебе будет предоставлен список текстов (сообщений).
Для КАЖДОГО текста в списке придумай ровно {self.queries_per_message} РАЗНЫХ, реалистичных, коротких и конкретных поисковых запросов или вопросов пользователя.
Верни результат в виде ОДНОГО ПЛОСКОГО СПИСКА строк. Список должен содержать сначала все {self.queries_per_message} запросов для первого сообщения, затем все {self.queries_per_message} запросов для второго сообщения, и так далее.
Общее количество элементов в списке должно быть ТОЧНО равно количеству входных сообщений умноженному на {self.queries_per_message}."""),
            ("human", "Вот список сообщений:\n{batch_messages_list}")
        ])

        self.query_generation_chain = self.query_generation_prompt | self.structured_llm

    # --- Новый метод для получения ID каналов ---
    async def get_channel_ids(self) -> List[int]:
        """Получает список ID всех каналов из таблицы channels."""
        logger.info("Получение списка ID каналов из БД...")
        async with self.SessionLocal() as session:
             # Выбираем только tg_channel_id
             stmt = select(Channels.tg_channel_id)
             result = await session.execute(stmt)
             channel_ids = [row[0] for row in result.fetchall()]
             logger.info(f"Получено {len(channel_ids)} ID каналов.")
        return channel_ids

    # --- Переименован и изменен для получения сообщений одного канала ---
    async def get_messages_for_channel(self, channel_id: int) -> List[str]:
        """Получает до self.channel_message_limit сообщений для указанного канала."""
        logger.debug(f"Получение сообщений для канала ID: {channel_id} (лимит: {self.channel_message_limit})")
        async with self.SessionLocal() as session:
            stmt = select(ChannelPosts.message).where(
                ChannelPosts.peer_id == channel_id, # <<< Фильтр по ID канала (peer_id)
                ChannelPosts.message.isnot(None)
                # Убрали фильтр func.length(ChannelPosts.message) > 100
            ).limit(self.channel_message_limit) # <<< Лимит на канал
            
            result = await session.execute(stmt)
            messages = [row[0] for row in result.fetchall() if row[0] and row[0].strip()]
            logger.debug(f"Получено {len(messages)} сообщений для канала {channel_id}.")
        return messages

    # Восстановлена и адаптирована версия для батчей с плоским списком
    async def process_message_batch_for_qa(self, message_batch: List[str], semaphore: asyncio.Semaphore) -> List[Dict[str, str]]:
        """Обрабатывает батч сообщений, генерируя по 5 запросов на каждое (плоский список)."""
        async with semaphore:
            batch_results = []
            # Генерируем doc_ids заранее, по одному на каждое сообщение в батче
            doc_ids = [str(uuid.uuid4()) for _ in message_batch]
            try:
                if not message_batch:
                    return []

                input_data = {"batch_messages_list": message_batch}
                # Ожидаемое количество = 10 сообщений * 5 запросов/сообщение = 50
                expected_total_queries = len(message_batch) * self.queries_per_message
                logger.debug(f"Генерация {expected_total_queries} запросов для батча из {len(message_batch)} сообщений (structured flat)...")

                # Ожидаем объект GeneratedQueries с полем queries: List[str]
                response: GeneratedQueries = await self.query_generation_chain.ainvoke(input_data)

                generated_queries = response.queries

                # Валидация: количество запросов в плоском списке
                if len(generated_queries) != expected_total_queries:
                    logger.warning(f"Несоответствие общего количества запросов. Ожидалось {expected_total_queries}, получено: {len(generated_queries)}. Ответ: {response}")
                    # Если количество не совпадает, надежно сопоставить невозможно, отбрасываем батч
                    return []

                # Итерируемся по плоскому списку запросов
                for idx, query_text in enumerate(generated_queries):
                    # Определяем индекс исходного сообщения
                    message_index = idx // self.queries_per_message
                    # Получаем текст и doc_id соответствующего сообщения
                    original_message = message_batch[message_index]
                    doc_id = doc_ids[message_index] # Используем пред-сгенерированный ID

                    query_text_cleaned = query_text.strip()
                    if not query_text_cleaned:
                         logger.debug(f"Пропущен пустой запрос (индекс {idx}) для сообщения: {original_message[:50]}...")
                         continue

                    # Генерируем УНИКАЛЬНЫЙ query_id для каждого запроса
                    query_id = str(uuid.uuid4())

                    # Добавляем словарь в результаты
                    batch_results.append({
                        "query_id": query_id,
                        "query_text": query_text_cleaned,
                        "doc_id": doc_id,
                        "doc_text": original_message
                    })

                logger.debug(f"Батч обработан (structured flat), создано {len(batch_results)} записей QA.")
                return batch_results

            except Exception as e:
                logger.error(f"Ошибка при обработке батча сообщений (structured flat): {e}", exc_info=True)
                logger.error(f"Батч, вызвавший ошибку (первые 50 символов каждого): {[msg[:50] + '...' for msg in message_batch]}")
                return [] # Отбрасываем батч при любой ошибке

    # --- Переработанная версия для итерации по каналам ---
    async def generate_qa_pairs(self) -> List[Dict[str, str]]:
        """Получает ID каналов, для каждого канала получает сообщения,
           делит на батчи и генерирует QA пары (по 5 на сообщение)."""
        all_valid_results = []
        total_processed_messages = 0
        total_batches_overall = 0
        tasks = []
        semaphore = asyncio.Semaphore(self.num_llms)

        try:
            logger.info("Начало генерации пар запрос-ответ по каналам...")
            channel_ids1 = await self.get_channel_ids()
            channel_ids = [1972167123, 1457520543, 1177340418, 1495729307, 1978809739, 1434942369]
            if not channel_ids:
                logger.warning("Не найдено ни одного канала для обработки.")
                return []

            logger.info(f"Найдено {len(channel_ids)} каналов. Обработка сообщений для каждого...")

            # Цикл по каналам
            for channel_id in channel_ids:
                logger.info(f"--- Обработка канала ID: {channel_id} ---")
                messages = await self.get_messages_for_channel(channel_id)

            if not messages:
                    logger.info(f"Нет сообщений для обработки в канале {channel_id}.")
                    continue

                current_channel_messages_count = len(messages)
                total_processed_messages += current_channel_messages_count
                logger.info(f"Получено {current_channel_messages_count} сообщений для канала {channel_id}.")

                # Делим сообщения ТЕКУЩЕГО канала на батчи
                message_batches = [messages[i:i + self.batch_size]
                                   for i in range(0, current_channel_messages_count, self.batch_size)]
                channel_batches_count = len(message_batches)
                total_batches_overall += channel_batches_count
                logger.info(f"Канал {channel_id}: Разделено на {channel_batches_count} батчей.")

                # Создаем задачи для батчей ТЕКУЩЕГО канала
                for batch in message_batches:
                     if batch:
                         task = asyncio.create_task(self.process_message_batch_for_qa(batch, semaphore))
                         tasks.append(task) # Добавляем в общий список задач

            # --- Обработка всех созданных задач ---
            if not tasks:
                 logger.warning("Не создано ни одной задачи для обработки батчей по всем каналам.")
                 return []

            logger.info(f"Запуск асинхронной обработки {len(tasks)} батчей со всех каналов...")
            # Используем tqdm для отслеживания прогресса по общему числу батчей
            for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Обработка батчей"):
                 batch_result = await f # Результат - плоский список словарей для одного батча
                 if batch_result:
                     all_valid_results.extend(batch_result)
                 # Добавляем задержку ПОСЛЕ обработки КАЖДОГО батча
                 await asyncio.sleep(1.5) # Оставляем задержку 1.5с

            logger.info(f"Успешно сгенерировано {len(all_valid_results)} QA пар.")
            logger.info(f"Всего обработано сообщений: {total_processed_messages} из {len(channel_ids)} каналов ({total_batches_overall} батчей).")
            return all_valid_results
            
        except Exception as e:
            logger.error(f"Критическая ошибка в процессе генерации QA пар по каналам: {e}", exc_info=True)
            return []
        finally:
            gc.collect()


async def main():
    logger.info("Запуск программы для генерации QA датасета (по каналам)")
    # Используем num_llms=1 для батчевой обработки
    processor = DataProcessor(num_llms=100, batch_size=10) # batch_size можно настроить

    qa_results = await processor.generate_qa_pairs() # Плоский список всех пар со всех каналов

    if not qa_results:
        logger.warning("Не было сгенерировано ни одной QA пары. Запись в файл не производится.")
        try:
             await processor.engine.dispose()
             logger.info("Соединение с БД закрыто (нет данных для записи).")
        except Exception as db_e:
             logger.error(f"Ошибка при закрытии соединения с БД: {db_e}")
        logger.info("Программа завершена (нет данных для записи).")
        return

    # Формирование JSON
    output_data = {
        "queries": {},
        "relevant_docs": {},
        "corpus": {}
    }
    logger.info(f"Формирование JSON структуры из {len(qa_results)} QA пар...")
    processed_docs = set()
    for item in qa_results:
        qid = item['query_id']
        qtext = item['query_text']
        did = item['doc_id']
        dtext = item['doc_text']

        # --- ВОЗВРАЩАЕМ ПРЕФИКСЫ ---
        output_data["queries"][qid] = f"query:{qtext}" # <<< Добавляем префикс 'query:'

        if did not in processed_docs:
            output_data["corpus"][did] = f"passage:{dtext}" # <<< Добавляем префикс 'passage:'
            processed_docs.add(did)
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---

        if qid not in output_data["relevant_docs"]:
             output_data["relevant_docs"][qid] = []
        if did not in output_data["relevant_docs"][qid]:
            output_data["relevant_docs"][qid].append(did)

    # output_filename1 = 'train_dataset_structured_5q_by_channel.json' # Если нужно, раскомментируйте и измените имя для тренировочного
    output_filename = 'validation_dataset_structured_5q_by_channel.json'
    logger.info(f"Запись {len(output_data['queries'])} запросов и {len(output_data['corpus'])} документов в файл {output_filename}...")
    try:
        async with aiofiles.open(output_filename, 'w', encoding='utf-8') as f:
            json_string = json.dumps(output_data, ensure_ascii=False, indent=4)
            await f.write(json_string)
        logger.info(f"Данные успешно записаны в {output_filename}")
    except Exception as e:
        logger.error(f"Ошибка при записи в файл {output_filename}: {e}")

    logger.info("Закрытие соединения с БД")
    try:
    await processor.engine.dispose()
        logger.info("Соединение с БД успешно закрыто.")
    except Exception as db_e:
        logger.error(f"Ошибка при закрытии соединения с БД: {db_e}")

    logger.info("Программа завершена")

if __name__ == "__main__":
    asyncio.run(main())