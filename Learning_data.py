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
            # Убираем фильтр по длине, берем все сообщения
            stmt = select(ChannelPosts.message).where(
                ChannelPosts.peer_id == channel_id,
                ChannelPosts.message.isnot(None)
                # func.length(ChannelPosts.message) > 100 # Убираем фильтр
            ).order_by(
                ChannelPosts.date.desc() # Оставляем сортировку
            ).limit(self.messages_limit)
            
            result = await session.execute(stmt)
            messages = [row[0] for row in result.fetchall() if row[0] and row[0].strip()]
            logger.info(f"Получено {len(messages)} непустых сообщений из канала {channel_id} (лимит: {self.messages_limit})")
        return messages

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

    async def process_single_message_ner(self, message_text: str, semaphore: asyncio.Semaphore):
        """Обрабатывает ОДНО сообщение для генерации NER-разметки."""
        async with semaphore:
            try:
                # Очищаем текст от лишних пробелов/переносов
                cleaned_text = ' '.join(message_text.split())
                if len(cleaned_text) < 10: # Пропускаем слишком короткие
                    logger.debug(f"Пропуск короткого сообщения: {cleaned_text[:50]}")
                    return None
                    
                # Обновляем промпт для обработки целого сообщения
                ner_prompt = ChatPromptTemplate.from_messages([
                    ("system", """Проанализируй ТЕКСТ СООБЩЕНИЯ ЦЕЛИКОМ и найди в нем сущности. 
                    Верни результат СТРОГО в формате ОДНОГО кортежа:
                        ('Полный текст сообщения с сохраненными переносами строк и форматированием.', {{'entities': [('текст_сущности', 'ТИП_СУЩНОСТИ')]}})
                    
                    Важно:
                    1. Обработай ВЕСЬ входной текст как ОДИН элемент.
                    2. Найди ВСЕ сущности во всем тексте сообщения.
                    3. Для каждой сущности укажи ТОЛЬКО её текст и тип.
                    4. Сохраняй оригинальные переносы строк и форматирование в тексте сообщения.
                    5. Текст сообщения должен быть заключен в одинарные кавычки.
                    6. НЕ ДЕЛАЙ переносов строк внутри самого кортежа.
                    
                    Типы сущностей: ORG, PER, LOC, DATE, MONEY, PRODUCT, ROLE, MISC
                    
                    Пример правильного формата:
                    ('Привет! John Smith из Meta выпустил LLaMA 2. Цена A100 упала до 20000 долларов в NYC.', {{'entities': [('John Smith', 'PER'), ('Meta', 'ORG'), ('LLaMA 2', 'PRODUCT'), ('A100', 'PRODUCT'), ('20000 долларов', 'MONEY'), ('NYC', 'LOC')]}})
                    
                    Верни ТОЛЬКО ОДИН кортеж с полным текстом сообщения и списком его сущностей."""),
                    ("human", "{text}")
                ])
                ner_chain = ner_prompt | self.llm

                logger.debug(f"Отправка сообщения в LLM для NER: {cleaned_text[:100]}...")
                response = await ner_chain.ainvoke({"text": cleaned_text})
                
                # Простая валидация формата (начинается с ( и заканчивается )))
                if response.content and response.content.strip().startswith("(") and response.content.strip().endswith(")))"):
                    validated_content = response.content.strip()
                    logger.debug(f"Получен валидный NER результат: {validated_content[:100]}...")
                    return validated_content
                else:
                    logger.warning(f"Невалидный формат ответа LLM для NER: {response.content}")
                    return None # Возвращаем None при невалидном формате

            except Exception as e:
                logger.error(f"Ошибка при обработке сообщения для NER: '{cleaned_text[:50]}...' - {e}", exc_info=True)
                return None
            finally:
                 await asyncio.sleep(1) # Небольшая пауза после каждого запроса к LLM

    async def process_channel(self, channel_id: int) -> None:
        try:
            logger.info(f"Начало обработки канала {channel_id}")
            messages = await self.get_messages(channel_id)
            if not messages:
                logger.warning(f"Нет сообщений для канала {channel_id}")
                return
            
            logger.info(f"Обработка {len(messages)} сообщений из канала {channel_id}...")
            
            # Используем семафор для ограничения параллельных запросов к LLM
            semaphore = asyncio.Semaphore(self.num_llms) 
            tasks = []
            for msg_text in messages:
                 task = asyncio.create_task(self.process_single_message_ner(msg_text, semaphore))
                 tasks.append(task)
                 
            # Собираем результаты
            results = await asyncio.gather(*tasks)
            
            # Фильтруем None и записываем валидные результаты в файл
            valid_results = [res for res in results if res is not None]
            logger.info(f"Канал {channel_id}: Получено {len(valid_results)} валидных NER результатов из {len(messages)} сообщений.")

            if valid_results:
                try:
                    async with aiofiles.open('TEST_DATA.txt', 'a', encoding='utf-8') as f:
                        # Записываем каждый результат с запятой и переносом строки
                        await f.write("\n".join([res + "," for res in valid_results]) + "\n") 
                    logger.info(f"Канал {channel_id}: Результаты ({len(valid_results)}) записаны в TEST_DATA.txt")
                except Exception as e:
                    logger.error(f"Канал {channel_id}: Ошибка записи результатов в файл: {e}")
            
        except Exception as e:
            logger.error(f"Ошибка обработки канала {channel_id}: {e}", exc_info=True)
        finally:
            gc.collect() # Сбор мусора после обработки канала

async def main():
    logger.info("Запуск программы")
    # Увеличиваем лимит сообщений, т.к. обрабатываем каждое
    # Уменьшаем количество параллельных LLM, т.к. теперь это семафор
    processor = DataProcessor(messages_limit=2000, sentences_per_chunk=100, num_llms=10) 
    # Берем ВСЕ каналы (или определяем их по-другому)
    # channels = [1466120158, 1511414765, 1298404306, 1322983992, 1141171940, 1600337678, 1269768079, 1331268451, 1406256149, 1238460311, 1322983992, 1447317686, 1460745685, 1195426632, 1054210809, 1380975080, 1720216167, 1498073945]
    async with processor.SessionLocal() as session:
        result = await session.execute(select(ChannelPosts.peer_id).distinct())
        channels = [row[0] for row in result.fetchall() if row[0]]
    logger.info(f"Найдено {len(channels)} уникальных каналов в БД для обработки.")
    
    logger.info(f"Начало обработки {len(channels)} каналов")
    # Создаем файл или очищаем его перед началом
    try:
        async with aiofiles.open('TEST_DATA.txt', 'w', encoding='utf-8') as f:
            await f.write("TRAINING_DATA = [\n") # Начало списка Python
    except Exception as e:
        logger.error(f"Не удалось создать/очистить файл TEST_DATA.txt: {e}")
        return

    for channel_id in channels:
        logger.info(f"Канал {channel_id}: начало обработки")
        await processor.process_channel(channel_id)
        logger.info(f"Канал {channel_id}: обработка завершена")
        # Убрана пауза между каналами, т.к. паузы есть между батчами внутри канала
        # await asyncio.sleep(5) 

    # Завершаем список в файле
    try:
        async with aiofiles.open('TEST_DATA.txt', 'a', encoding='utf-8') as f:
            # Удаляем последнюю запятую и перенос строки, если они есть
            # await f.seek(0, os.SEEK_END) # Не работает с aiofiles так просто
            # size = await f.tell()
            # if size > 2:
            #      await f.seek(size - 2)
            #      last_chars = await f.read(2)
            #      if last_chars == ",\n":
            #          await f.seek(size - 2)
            #          await f.truncate() 
            await f.write("\n] # Конец списка Python") 
    except Exception as e:
        logger.error(f"Не удалось завершить файл TEST_DATA.txt: {e}")

    logger.info("Закрытие соединения с БД")
    await processor.engine.dispose()
    logger.info("Программа завершена")

if __name__ == "__main__":
    asyncio.run(main())