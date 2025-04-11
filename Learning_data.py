import logging
from typing import List
from sqlalchemy import select, Text, Column, Integer, BigInteger, DateTime, String, func
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import asyncio
import re  # Добавим в начало файла
import aiofiles
import gc
import traceback

# Импортируем ключ API из конфига
from config import GEMINI_API_KEY, POSTGRES_ASYNC_URL

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

class DataProcessor:
    def __init__(self, messages_limit: int = 500, sentences_per_chunk: int = 100, num_llms: int = 2):
        logger.info(f"Инициализация DataProcessor: {messages_limit} сообщений, {sentences_per_chunk} предложений в чанке, {num_llms} LLM")
        self.messages_limit = messages_limit
        self.sentences_per_chunk = sentences_per_chunk
        self.num_llms = num_llms
        
        logger.info("Подключение к базе данных...")
        self.engine = create_async_engine(
            POSTGRES_ASYNC_URL, # Используем URL из конфига
            echo=False
        )
        self.SessionLocal = async_sessionmaker(bind=self.engine, class_=AsyncSession)
        logger.info("Подключение к БД успешно установлено")
        
        logger.info("Инициализация Google Gemini LLM...")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            google_api_key=GEMINI_API_KEY # Используем ключ из конфига
        )
        logger.info("LLM успешно инициализирована")
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Проанализируй текст, раздели его на предложения и найди в каждом предложении сущности. 
            Верни результат СТРОГО в формате:
                ('Первое предложение текста.', {{'entities': [('текст_сущности', 'тип_сущности')]}},
                ('Второе предложение текста.', {{'entities': [('текст_сущности', 'тип_сущности')]}})
            
            Важно:
            1. Раздели входной текст на отдельные предложения
            2. Каждое предложение должно быть отдельным элементом списка
            3. Для каждого предложения укажи найденные в нем сущности
            4. Для каждой сущности укажи ТОЛЬКО её текст и тип
            5. Сохраняй знаки препинания и форматирование в предложениях
            6. НЕ ПЕРЕНОСИ ПО СТРОКАМ НИЧЕГО. ОДНО ПРЕДЛОЖЕНИЕ - ОДНА СТРОКА
            
            Типы сущностей:
            - ORG: компании, организации (Google, GitHub, Meta)
            - PER: имена людей
            - LOC: места, локации
            - DATE: даты, периоды времени
            - MONEY: цены, суммы
            - PRODUCT: Всеоблемющая тема по типу названия продуктов модели ллмок, марки машин, модели GPU (H100, A100, RTX 4090) версия айфон и т д
            - ROLE: должности, позиции (Staff Research Scientist, CEO, ML Engineer)
            - MISC: прочие важные сущности
            
            Пример правильного формата:
                ('Модель 27B влезает в одну H100 GPU в bf16.', {{'entities': [('H100', 'PRODUCT'), ('GPU', 'PRODUCT'), ('bf16', 'PRODUCT')]}}),
                ('John Smith работает Staff Research Scientist в GitHub.', {{'entities': [('John Smith', 'PER'), ('Staff Research Scientist', 'ROLE'), ('GitHub', 'ORG')]}}),
                ('Цена A100 составит 20000 долларов.', {{'entities': [('A100', 'PRODUCT'), ('20000 долларов', 'MONEY')]}}),
                ('Meta выпустила LLaMA 2 для коммерческого использования.', {{'entities': [('Meta', 'ORG'), ('LLaMA 2', 'PRODUCT')]}}),
            
            Верни ТОЛЬКО предложения и их сущностями."""),
            ("human", "{text}")
        ])
        self.chain = self.prompt | self.llm

        # Добавляем валидационный промпт
        self.validation_prompt = ChatPromptTemplate.from_messages([
            ("system", """Ты - эксперт по проверке правильности форматирования данных для NER.
            Проверь правильность форматирования предоставленных данных.
            ОЧЕНЬ КРИТИЧЕСКИ ОТНОСИСЬ К ЭТОМУ ФОРМАТУ.
            
            ВАЖНО:
            1. Должны быть ТОЛЬКО элементы в формате ('текст', {{"entities": [("текст_сущности", "тип_сущности")]}})
            2. Все остальное (разметка, JSON, скобки, кавычки) должно быть удалено
            3. Элементы должны быть разделены ТОЛЬКО запятыми
            4. НЕ ДОЛЖНО БЫТЬ никаких переносов строк между элементами
            5. НЕ ДОЛЖНО БЫТЬ никаких лишних пробелов между элементами
            6. НЕ ДОЛЖНО БЫТЬ никаких скобок или кавычек между элементами
            
            Правила проверки:
            1. Каждый элемент должен быть кортежем с текстом и словарем entities
            2. Каждая сущность должна быть кортежем из текста и типа
            3. Типы должны быть из списка: ORG, PER, LOC, DATE, MONEY, PRODUCT, ROLE, MISC
            4. Текст должен быть в одинарных кавычках
            5. Формат JSON должен быть строго соблюден
            
            Если находишь ошибки или лишние элементы, исправь и верни ТОЛЬКО правильные элементы через запятую.
            Если ошибок нет, верни исходный текст без изменений.
            
            Пример правильного формата:
            ('John works at Google', {{"entities": [("John", "PER"), ("Google", "ORG")]}}), ('Jane works at Meta', {{"entities": [("Jane", "PER"), ("Meta", "ORG")]}})
"""),
            ("human", "{input_text}")
        ])
        
        # Создаем цепочку для валидации
        self.validation_chain = self.validation_prompt | self.llm

    async def get_messages(self, channel_id: int) -> List[str]:
        logger.info(f"Получение сообщений из канала {channel_id}")
        async with self.SessionLocal() as session:
            stmt = select(ChannelPosts.message).where(
                ChannelPosts.peer_id == channel_id,
                ChannelPosts.message.isnot(None),
                func.length(ChannelPosts.message) > 100
            ).order_by(
                ChannelPosts.date.desc()
            ).limit(self.messages_limit)
            
            result = await session.execute(stmt)
            messages = [row[0] for row in result.fetchall() if row[0]]
            logger.info(f"Получено {len(messages)} сообщений из канала {channel_id}")
        return messages

    def split_into_sentences(self, text: str) -> List[str]:
        """Разбивает текст на предложения."""
        pattern = r'(?<=[.!?…])\s+'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    async def validate_chunk(self, content: str) -> str:
        """Валидирует и исправляет форматирование чанка данных."""
        try:
            logger.info("Начало валидации чанка данных...")
            # Логируем первые 200 символов входных данных
            logger.info(f"Входные данные (первые 200 символов): {content[:200]}")
            
            response = await self.validation_chain.ainvoke({"input_text": content})
            
            # Логируем первые 200 символов выходных данных
            logger.info(f"Выходные данные (первые 200 символов): {response.content[:200]}")
            
            if response.content != content:
                logger.info("Валидация внесла изменения в данные")
            else:
                logger.info("Данные прошли валидацию без изменений")
                
            logger.info("Валидация чанка завершена успешно")
            return response.content
        except Exception as e:
            logger.error(f"Ошибка при валидации чанка: {e}")
            logger.warning("Возвращаем исходный контент без валидации")
            return content

    async def process_chunk_async(self, sentences: List[str], llm_number: int):
        """Асинхронная обработка всех предложений для одной LLM"""
        try:
            logger.info(f"LLM {llm_number}: всего получено {len(sentences)} предложений")
            
            # Разбиваем предложения на чанки по 100
            for i in range(0, len(sentences), self.sentences_per_chunk):
                chunk = sentences[i:i + self.sentences_per_chunk]
                chunk_number = i // self.sentences_per_chunk + 1
                total_chunks = (len(sentences) + self.sentences_per_chunk - 1) // self.sentences_per_chunk
                
                logger.info(f"LLM {llm_number}: обработка чанка {chunk_number}/{total_chunks} ({len(chunk)} предложений)")
                text_for_llm = "\n".join(chunk)
                
                # Первичная обработка
                logger.info(f"LLM {llm_number}, чанк {chunk_number}: отправка запроса в модель")
                response = await self.chain.ainvoke({"text": text_for_llm})
                
                # Валидация результата
                logger.info(f"LLM {llm_number}, чанк {chunk_number}: валидация результатов")
                validated_content = await self.validate_chunk(response.content)
                
                # Запись валидированного результата
                logger.info(f"LLM {llm_number}, чанк {chunk_number}: запись результатов")
                async with aiofiles.open('TEST_DATA.txt', 'a', encoding='utf-8') as f:
                    await f.write(validated_content + "\n\n")
                
                logger.info(f"LLM {llm_number}: чанк {chunk_number}/{total_chunks} успешно обработан")
                await asyncio.sleep(2)  # Пауза между чанками
                
            logger.info(f"LLM {llm_number}: все чанки успешно обработаны")
        except Exception as e:
            logger.error(f"LLM {llm_number}: ошибка при обработке: {e}")

    async def process_channel(self, channel_id: int) -> None:
        try:
            logger.info(f"Начало обработки канала {channel_id}")
            messages = await self.get_messages(channel_id)
            if not messages:
                logger.warning(f"Нет сообщений для канала {channel_id}")
                return

            full_text = " ".join(messages)
            logger.info(f"Объединено {len(messages)} сообщений в текст")

            all_sentences = self.split_into_sentences(full_text)
            logger.info(f"Текст разбит на {len(all_sentences)} предложений")
            
            # Разделяем все предложения между LLM
            sentences_per_llm = len(all_sentences) // self.num_llms
            chunks_for_llms = [all_sentences[i:i + sentences_per_llm] for i in range(0, len(all_sentences), sentences_per_llm)]
            logger.info(f"Предложения разделены между {len(chunks_for_llms)} LLM по ~{sentences_per_llm} предложений")
            
            # Создаем задачи для асинхронного выполнения
            tasks = []
            for i, chunk in enumerate(chunks_for_llms):
                logger.info(f"Создание задачи для LLM {i+1}: {len(chunk)} предложений")
                task = asyncio.create_task(self.process_chunk_async(chunk, i+1))
                tasks.append(task)
            
            # Запускаем все задачи параллельно
            logger.info(f"Запуск {len(tasks)} параллельных LLM")
            await asyncio.gather(*tasks)
            logger.info(f"Все LLM для канала {channel_id} завершили работу")
            
        except Exception as e:
            logger.error(f"Ошибка обработки канала {channel_id}: {e}")

    def _extract_named_entities(self, text):
        """Синхронное извлечение именованных сущностей через spaCy NER"""
        if not text or not self.nlp_model:
            return []
        
        try:
            # Логируем размер входного текста
            text_size = len(text) / 1024
            print(f"[MEM] Размер текста: {text_size:.2f} KB")
            
            # Ограничиваем длину текста для обработки, если он слишком большой
            if len(text) > 10000:
                text = text[:10000]  # Обрабатываем только первые 10000 символов
                print(f"[MEM] Текст обрезан до 10000 символов")
            
            # Обрабатываем текст с помощью spaCy
            doc = self.nlp_model(text)
            
            # Список известных типов NER в русской модели spaCy
            valid_entity_types = [
                'PER', 'PERSON',  # Люди
                'LOC', 'GPE',     # Места, геополитические сущности
                'ORG',            # Организации
                'DATE', 'TIME',   # Даты и время
                'MONEY',          # Валюта
                'PRODUCT',        # Продукты
                'EVENT',          # События
                'FAC'             # Здания, сооружения
            ]
            
            # Извлекаем именованные сущности только определенных типов
            # и фильтруем короткие
            entities = []
            for ent in doc.ents:
                # Проверяем тип сущности и длину
                if ent.label_ in valid_entity_types and len(ent.text) > 2:
                    # Проверяем и исправляем индексы
                    start = ent.start_char
                    end = ent.end_char
                    
                    # Проверяем, что индексы в пределах текста
                    if start >= 0 and end <= len(text):
                        # Проверяем, что текст сущности совпадает
                        if text[start:end] == ent.text:
                            entities.append(ent.text.lower())
                        else:
                            print(f"[WARNING] Несовпадение текста сущности: {text[start:end]} != {ent.text}")
                    else:
                        print(f"[WARNING] Индексы вне диапазона: {start}, {end} для текста длиной {len(text)}")
            
            # Удаляем дубликаты
            result = list(set(entities))
            
            # Очищаем большие переменные
            del doc, entities
            gc.collect()  # Принудительный сбор мусора после обработки NER
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Ошибка извлечения именованных сущностей: {e}")
            print(f"[ERROR] {traceback.format_exc()}")
            return []


async def main():
    logger.info("Запуск программы")
    processor = DataProcessor(messages_limit=500, sentences_per_chunk=100, num_llms=10)
    channels = [1466120158, 1511414765, 1298404306, 1322983992, 1141171940, 1600337678, 1269768079, 1331268451, 1406256149, 1238460311, 1322983992, 1447317686, 1460745685, 1195426632, 1054210809, 1380975080, 1720216167, 1498073945]
    
    logger.info(f"Начало обработки {len(channels)} каналов")
    for channel_id in channels:
        logger.info(f"Канал {channel_id}: начало обработки")
        await processor.process_channel(channel_id)
        logger.info(f"Канал {channel_id}: обработка завершена")
        await asyncio.sleep(5)

    logger.info("Закрытие соединения с БД")
    await processor.engine.dispose()
    logger.info("Программа завершена")

if __name__ == "__main__":
    asyncio.run(main())