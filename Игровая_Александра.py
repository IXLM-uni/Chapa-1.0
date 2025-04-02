import os
import sys
import asyncio
import logging
import numpy as np
import faiss
import argparse
from dotenv import load_dotenv
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Text, DateTime, BigInteger, ARRAY, Float
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
import math
import plotly.express as px
from sklearn.decomposition import PCA

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

# Настройки БД и API ключи
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASS = "12345"
POSTGRES_ASYNC_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Имя модели эмбеддинга
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

# Определение модели базы данных
Base = declarative_base()

class ChannelMessage(Base):
    __tablename__ = "channel_messages"
    
    id = Column(Integer, primary_key=True)
    post_author = Column(String)
    peer_id = Column(BigInteger)
    message_id = Column(BigInteger)
    message = Column(Text)
    key_words = Column(Text)
    forwards = Column(Integer)
    embedding = Column(ARRAY(Float))
    date = Column(DateTime)

# Класс для работы с FAISS индексом
class FaissIndexManager:
    def __init__(self, omp_threads: int = 4):
        self.omp_threads = omp_threads
        self.index = None
        self.message_ids = []
        faiss.omp_set_num_threads(self.omp_threads)
        logger.info(f"FAISS инициализирован с {self.omp_threads} потоками")
        
    async def create_index(self, message_embeddings: List[Tuple[int, List[float]]]) -> bool:
        """
        Создает FAISS индекс из списка эмбеддингов.
        
        Args:
            message_embeddings: Список кортежей (id, embedding)
            
        Returns:
            bool: Успешно ли создан индекс
        """
        try:
            # Фильтруем пустые и некорректные эмбеддинги
            filtered_embeddings = []
            expected_dim = None
            
            # Определяем ожидаемую размерность из первого непустого эмбеддинга
            for msg_id, emb in message_embeddings:
                if emb and len(emb) > 0:
                    expected_dim = len(emb)
                    break
                    
            if expected_dim is None:
                logger.error("Не найдено ни одного валидного эмбеддинга")
                return False
                
            logger.info(f"Ожидаемая размерность эмбеддингов: {expected_dim}")
            
            # Фильтруем эмбеддинги с неправильной размерностью
            for msg_id, emb in message_embeddings:
                if emb and len(emb) == expected_dim:
                    filtered_embeddings.append((msg_id, emb))
                else:
                    logger.warning(f"Пропущен эмбеддинг для сообщения {msg_id}: неверная размерность {len(emb) if emb else 0} вместо {expected_dim}")
            
            logger.info(f"Отфильтровано {len(message_embeddings) - len(filtered_embeddings)} некорректных эмбеддингов")
            
            if not filtered_embeddings:
                logger.error("После фильтрации не осталось валидных эмбеддингов")
                return False
                
            # Извлекаем ID сообщений и эмбеддинги
            self.message_ids = [msg_id for msg_id, _ in filtered_embeddings]
            embeddings = [np.array(emb, dtype=np.float32) for _, emb in filtered_embeddings]
            
            # Создаем матрицу из эмбеддингов
            embeddings_matrix = np.vstack(embeddings).astype(np.float32)
            dimension = embeddings_matrix.shape[1]
            
            # Создаем и обучаем индекс
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings_matrix)
            
            logger.info(f"FAISS индекс создан успешно. Размерность: {dimension}, количество векторов: {self.index.ntotal}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при создании FAISS индекса: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def search(self, query_embedding: List[float], top_k: int = 5) -> Tuple[List[int], List[float]]:
        """
        Выполняет поиск ближайших эмбеддингов для заданного запроса.
        
        Args:
            query_embedding: Эмбеддинг запроса
            top_k: Количество результатов
            
        Returns:
            Tuple[List[int], List[float]]: Список ID сообщений и расстояний
        """
        if self.index is None:
            logger.error("FAISS индекс не инициализирован")
            return [], []
            
        try:
            # Преобразуем запрос в нужный формат
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Выполняем поиск
            distances, indices = self.index.search(query_vector, top_k)
            
            # Получаем ID сообщений из индексов
            result_message_ids = [self.message_ids[idx] for idx in indices[0] if idx < len(self.message_ids)]
            result_distances = distances[0].tolist()
            
            return result_message_ids, result_distances
            
        except Exception as e:
            logger.error(f"Ошибка при поиске в FAISS индексе: {e}")
            import traceback
            traceback.print_exc()
            return [], []

# Класс для работы с базой данных и текстом
class DatabaseProcessor:
    def __init__(self, db_url: str):
        self.engine = create_async_engine(db_url)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
        # Загружаем модель E5 для создания эмбеддингов
        try:
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info(f"Модель эмбеддинга загружена: {EMBEDDING_MODEL_NAME}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели эмбеддинга: {e}")
            self.embedding_model = None
        
    async def get_all_message_embeddings(self) -> List[Tuple[int, List[float]]]:
        """
        Получает все ID и эмбеддинги из базы данных.
        
        Returns:
            List[Tuple[int, List[float]]]: Список кортежей (id, embedding)
        """
        try:
            async with self.async_session() as session:
                # Получаем только непустые эмбеддинги
                query = select(ChannelMessage.id, ChannelMessage.embedding).where(
                    ChannelMessage.embedding != None
                )
                result = await session.execute(query)
                message_embeddings = result.all()
                
            if not message_embeddings:
                logger.warning("В базе данных не найдено эмбеддингов сообщений")
                return []
            
            # Фильтруем пустые эмбеддинги
            valid_embeddings = []
            for msg_id, embedding in message_embeddings:
                if embedding and len(embedding) > 0:
                    valid_embeddings.append((msg_id, embedding))
                else:
                    logger.warning(f"Пропущен пустой эмбеддинг для сообщения ID: {msg_id}")
            
            logger.info(f"Загружено {len(valid_embeddings)} валидных эмбеддингов из {len(message_embeddings)} записей")
            return valid_embeddings
            
        except Exception as e:
            logger.error(f"Ошибка при получении эмбеддингов из БД: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def get_message_by_id(self, message_id: int) -> Optional[Dict[str, Any]]:
        """
        Получает сообщение по его ID.
        
        Args:
            message_id: ID сообщения
            
        Returns:
            Optional[Dict[str, Any]]: Данные сообщения или None
        """
        try:
            async with self.async_session() as session:
                # Выводим полный запрос для диагностики
                logger.debug(f"Получение сообщения с ID: {message_id}")
                query = select(ChannelMessage).where(ChannelMessage.id == message_id)
                result = await session.execute(query)
                message = result.scalar_one_or_none()
                
            if not message:
                logger.warning(f"Сообщение с ID {message_id} не найдено в базе данных")
                return None
            
            # Добавляем проверку на пустые поля
            data = {
                "id": message.id,
                "author": message.post_author or "Неизвестный",
                "peer_id": message.peer_id,
                "message_id": message.message_id,
                "message": message.message or "Текст сообщения отсутствует",
                "key_words": message.key_words,
                "date": message.date
            }
            
            # Логируем для диагностики
            logger.debug(f"Получено сообщение: {data['message'][:50]}..." if data['message'] and len(data['message']) > 50 else data['message'])
            return data
            
        except Exception as e:
            logger.error(f"Ошибка при получении сообщения из БД с ID {message_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def get_embedding(self, text: str) -> List[float]:
        """
        Получает эмбеддинг для текста с использованием модели E5.
        
        Args:
            text: Текст для эмбеддинга
            
        Returns:
            List[float]: Эмбеддинг текста
        """
        if self.embedding_model is None:
            logger.error("Модель эмбеддинга не инициализирована")
            return []
            
        try:
            # В E5 рекомендуется добавлять префикс для запросов
            query_text = f"query: {text}"
            
            # SentenceTransformer возвращает numpy массив, преобразуем в список
            embedding = self.embedding_model.encode(query_text).tolist()
            return embedding
            
        except Exception as e:
            logger.error(f"Ошибка при получении эмбеддинга: {e}")
            import traceback
            traceback.print_exc()
            return []

# Класс для поиска похожих сообщений с ранжированием
class RankedMessageSearcher:
    def __init__(self, db_processor, faiss_manager, 
                 similarity_weight=0.01, 
                 length_weight=0.01, 
                 recency_weight=0.98,
                 full_text=False):
        self.db_processor = db_processor
        self.faiss_manager = faiss_manager
        
        # Веса для факторов ранжирования
        self.similarity_weight = similarity_weight
        self.length_weight = length_weight
        self.recency_weight = recency_weight
        
        # Режим вывода полного текста
        self.full_text = full_text
        
        # Статистические метрики для нормализации
        self.avg_message_length = 300  # Начальное значение, будет уточнено
        self.max_message_length = 3000  # Примерное максимальное значение
        self.min_date = datetime.now() - timedelta(days=365*2)  # предполагаем, что сообщения не старше 2 лет
        
        logger.info(f"Инициализация ранжирования с весами: "
                    f"сходство={similarity_weight}, "
                    f"длина={length_weight}, "
                    f"свежесть={recency_weight}, "
                    f"полный текст={full_text}")
    
    async def update_metrics(self, sample_size=1000):
        """Обновляет статистические метрики на основе данных из БД"""
        logger.info("Обновление метрик для нормализации факторов ранжирования...")
        
        try:
            async with self.db_processor.async_session() as session:
                # Получаем выборку для оценки средней длины и минимальной даты
                query = select(ChannelMessage.message, ChannelMessage.date) \
                       .where(ChannelMessage.message != None) \
                       .limit(sample_size)
                result = await session.execute(query)
                samples = result.all()
                
                if not samples:
                    logger.warning("Не найдены образцы для обновления метрик")
                    return
                
                # Вычисляем среднюю длину сообщений
                message_lengths = [len(msg) if msg else 0 for msg, _ in samples]
                if message_lengths:
                    self.avg_message_length = sum(message_lengths) / len(message_lengths)
                    self.max_message_length = max(message_lengths)
                
                # Определяем самую раннюю дату
                dates = [date for _, date in samples if date]
                if dates:
                    self.min_date = min(dates)
                
                logger.info(f"Обновлены метрики: средняя длина={self.avg_message_length:.2f}, "
                           f"макс. длина={self.max_message_length}, "
                           f"мин. дата={self.min_date}")
                
        except Exception as e:
            logger.error(f"Ошибка при обновлении метрик: {e}")
    
    def normalize_similarity(self, similarity_score: float) -> float:
        """
        Нормализует оценку семантического сходства.
        FAISS возвращает расстояние (меньше = лучше), преобразуем в сходство.
        """
        # Для расстояния L2: преобразуем в оценку [0,1]
        # Типичные значения расстояний 0-10, поэтому делим на 10
        MAX_DISTANCE = 10.0
        distance = min(similarity_score, MAX_DISTANCE)
        return 1.0 - (distance / MAX_DISTANCE)
    
    def calculate_length_score(self, text_length: int) -> float:
        """
        Вычисляет нормализованную оценку длины сообщения.
        Используем логарифмическую шкалу для смягчения преимущества очень длинных текстов.
        """
        if text_length <= 0:
            return 0.0
        
        # Логарифмическая нормализация длины
        normalized_length = math.log(1 + text_length) / math.log(1 + self.max_message_length)
        return min(normalized_length, 1.0)
    
    def calculate_recency_score(self, date: datetime) -> float:
        """
        Вычисляет нормализованную оценку свежести сообщения.
        Более новые сообщения получают более высокие оценки.
        """
        if not date:
            return 0.5  # Средняя оценка для сообщений без даты
        
        now = datetime.now()
        
        # Ограничиваем временной диапазон
        min_date = self.min_date
        max_age = (now - min_date).total_seconds()
        
        if max_age <= 0:
            return 1.0  # Избегаем деления на ноль
        
        # Вычисляем возраст сообщения и нормализуем
        message_age = (now - date).total_seconds()
        message_age = max(0, min(message_age, max_age))  # Ограничиваем диапазон
        
        # Преобразуем возраст в оценку свежести (новее = выше)
        recency_score = 1.0 - (message_age / max_age)
        return recency_score
    
    def calculate_final_score(self, similarity: float, length: int, date: datetime) -> float:
        """
        Вычисляет итоговую оценку на основе взвешенной комбинации факторов.
        
        Args:
            similarity: Оценка семантического сходства (меньше = лучше)
            length: Длина текста сообщения
            date: Дата публикации сообщения
            
        Returns:
            float: Итоговая оценка для ранжирования
        """
        # Нормализуем все факторы
        similarity_score = self.normalize_similarity(similarity)
        length_score = self.calculate_length_score(length)
        recency_score = self.calculate_recency_score(date)
        
        # Вычисляем взвешенную сумму
        final_score = (
            self.similarity_weight * similarity_score +
            self.length_weight * length_score +
            self.recency_weight * recency_score
        )
        
        logger.debug(f"Оценки: сходство={similarity_score:.3f}, длина={length_score:.3f}, "
                     f"свежесть={recency_score:.3f}, итого={final_score:.3f}")
        
        return final_score
    
    async def search_messages(self, query_text: str, top_k: int = 33) -> List[Dict]:
        """
        Выполняет поиск похожих сообщений по тексту запроса с ранжированием.
        
        Args:
            query_text: Текст запроса
            top_k: Количество результатов
            
        Returns:
            List[Dict]: Список результатов поиска
        """
        # Получение эмбеддинга запроса
        query_embedding = await self.db_processor.get_embedding(query_text)
        if not query_embedding:
            logger.error("Не удалось получить эмбеддинг для запроса")
            return []
            
        # Поиск похожих сообщений (получаем больше, чем нужно, для последующего ранжирования)
        candidate_k = min(top_k * 3, 100)  # Берем в 3 раза больше кандидатов, но не более 20
        message_ids, distances = await self.faiss_manager.search(query_embedding, top_k=candidate_k)
        
        if not message_ids:
            logger.warning("Не найдено похожих сообщений")
            return []
        
        logger.info(f"Найдено {len(message_ids)} кандидатов, выполняем ранжирование...")
        
        # Формирование кандидатов для ранжирования
        candidates = []
        for i, (msg_id, distance) in enumerate(zip(message_ids, distances)):
            message_data = await self.db_processor.get_message_by_id(msg_id)
            if message_data:
                # Получаем данные для расчета ранга
                message_text = message_data.get("message", "")
                message_length = len(message_text) if message_text else 0
                message_date = message_data.get("date")
                
                # Вычисляем итоговый ранг
                final_score = self.calculate_final_score(
                    similarity=distance, 
                    length=message_length, 
                    date=message_date
                )
                
                candidates.append({
                    "№": i+1,
                    "ID": msg_id,
                    "Автор": message_data.get("author", "Н/Д"),
                    "Сообщение": message_text or "Текст отсутствует",
                    "Длина": message_length,
                    "Дата": message_date,
                    "Сходство": distance,
                    "Итоговый_ранг": final_score
                })
            else:
                logger.warning(f"Не удалось получить данные для сообщения ID: {msg_id}")
        
        # Сортируем по итоговому рангу (выше = лучше)
        ranked_results = sorted(candidates, key=lambda x: x["Итоговый_ранг"], reverse=True)
        
        # Возвращаем только необходимое количество результатов
        top_results = ranked_results[:top_k]
        
        # Обновляем номера после сортировки
        for i, result in enumerate(top_results):
            result["№"] = i + 1
        
        return top_results
    
    def print_results(self, results: List[Dict]):
        """
        Выводит результаты поиска в консоль.
        
        Args:
            results: Список результатов поиска
        """
        if not results:
            print("\nРезультаты не найдены или произошла ошибка\n")
            return
            
        print("\n=== Результаты поиска ===")
        print(f"Веса ранжирования: сходство={self.similarity_weight}, "
              f"длина текста={self.length_weight}, "
              f"свежесть={self.recency_weight}")
        
        for i, result in enumerate(results):
            print(f"\n[{i+1}] ID: {result['ID']}, Итоговый ранг: {result['Итоговый_ранг']:.2f}")
            print(f"Автор: {result['Автор'] if result['Автор'] else 'Не указан'}")
            
            date_str = result['Дата'].strftime('%d.%m.%Y %H:%M') if result.get('Дата') else 'Н/Д'
            print(f"Дата: {date_str}, Длина: {result['Длина']} символов")
            
            print(f"Семантическое сходство: {1.0 - result['Сходство']/10:.2f}")
            
            # Выводим сообщение (полностью или фрагмент)
            if self.full_text:
                print(f"Сообщение:\n{result['Сообщение']}")
            else:
                message_preview = result['Сообщение'][:200] + "..." if len(result['Сообщение']) > 200 else result['Сообщение']
                print(f"Сообщение: {message_preview}")
            print("-" * 50)

    async def visualize_embeddings(self, query_text: str, n_random: int = 500):
        """
        Создает визуализацию эмбеддингов: результаты поиска (красным) и случайные сообщения (синим).
        
        Args:
            query_text: Текст запроса
            n_random: Количество случайных сообщений для отображения
            
        Returns:
            None: Открывает интерактивный график Plotly
        """
        try:
            logger.info(f"Создание визуализации эмбеддингов для запроса: '{query_text}'")
            
            # Получение эмбеддинга запроса
            query_embedding = await self.db_processor.get_embedding(query_text)
            if not query_embedding:
                logger.error("Не удалось получить эмбеддинг для запроса")
                return
                
            # Поиск похожих сообщений
            message_ids, distances = await self.faiss_manager.search(query_embedding, top_k=30)
            
            if not message_ids:
                logger.warning("Не найдено похожих сообщений")
                return
                
            logger.info(f"Получено {len(message_ids)} релевантных сообщений для визуализации")
            
            # Получаем эмбеддинги релевантных сообщений напрямую из индекса
            relevant_embeddings = []
            relevant_texts = []
            relevant_ids = []
            
            # Получаем текстовые данные релевантных сообщений
            for msg_id in message_ids:
                message_data = await self.db_processor.get_message_by_id(msg_id)
                if message_data:
                    relevant_ids.append(msg_id)
                    relevant_texts.append(message_data.get("message", ""))
                    
                    # Получаем эмбеддинг из базы данных
                    async with self.db_processor.async_session() as session:
                        query = select(ChannelMessage.embedding).where(ChannelMessage.id == msg_id)
                        result = await session.execute(query)
                        embedding = result.scalar_one_or_none()
                        if embedding:
                            relevant_embeddings.append(embedding)
                        else:
                            logger.warning(f"Эмбеддинг не найден для сообщения {msg_id}")
            
            if not relevant_embeddings:
                logger.error("Не удалось получить эмбеддинги для релевантных сообщений")
                return
            
            logger.info(f"Получено {len(relevant_embeddings)} эмбеддингов релевантных сообщений")
            
            # Получаем случайные сообщения для сравнения
            random_messages = []
            async with self.db_processor.async_session() as session:
                # Запрос для получения случайных сообщений, не входящих в найденные
                query = select(ChannelMessage.id, ChannelMessage.message, ChannelMessage.embedding).where(
                    ChannelMessage.id.notin_(message_ids),
                    ChannelMessage.embedding != None
                ).order_by(func.random()).limit(n_random)
                
                result = await session.execute(query)
                random_data = result.all()
                
                for msg_id, text, embedding in random_data:
                    if embedding:
                        random_messages.append({
                            "id": msg_id,
                            "text": text or "",
                            "embedding": embedding
                        })
            
            logger.info(f"Получено {len(random_messages)} случайных сообщений для сравнения")
            
            # Создаем массив всех эмбеддингов
            embeddings_data = []
            
            # Добавляем релевантные сообщения
            for i, emb in enumerate(relevant_embeddings):
                embeddings_data.append({
                    "embedding": emb,
                    "id": relevant_ids[i] if i < len(relevant_ids) else -2,
                    "text": relevant_texts[i] if i < len(relevant_texts) else "",
                    "type": "релевантное"
                })
            
            # Добавляем случайные сообщения
            for msg in random_messages:
                embeddings_data.append({
                    "embedding": msg["embedding"],
                    "id": msg["id"],
                    "text": msg["text"],
                    "type": "случайное"
                })
            
            if not embeddings_data:
                logger.error("Нет данных для визуализации")
                return
            
            # Извлекаем эмбеддинги в numpy массив
            all_embeddings = np.array([msg["embedding"] for msg in embeddings_data])
            
            # Добавляем эмбеддинг запроса
            all_embeddings = np.vstack([all_embeddings, np.array(query_embedding)])
            
            # Применяем PCA для уменьшения размерности до 2D
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(all_embeddings)
            
            # Создаем DataFrame для Plotly
            df = pd.DataFrame({
                "x": reduced_embeddings[:-1, 0],
                "y": reduced_embeddings[:-1, 1],
                "id": [msg["id"] for msg in embeddings_data],
                "text": [msg["text"][:100] + "..." if len(msg["text"]) > 100 else msg["text"] for msg in embeddings_data],
                "type": [msg["type"] for msg in embeddings_data],
                "full_text": [msg["text"] for msg in embeddings_data]
            })
            
            # Добавляем запрос как отдельную точку
            query_point = pd.DataFrame({
                "x": [reduced_embeddings[-1, 0]],
                "y": [reduced_embeddings[-1, 1]],
                "id": [-1],
                "text": ["ЗАПРОС: " + query_text],
                "type": ["запрос"],
                "full_text": [query_text]
            })
            
            df = pd.concat([df, query_point])
            
            # Цветовая схема
            color_map = {"релевантное": "red", "случайное": "blue", "запрос": "green"}
            
            # Размеры маркеров
            size_map = {"релевантное": 10, "случайное": 6, "запрос": 15}
            
            # Создаем столбец для размера маркеров
            df["marker_size"] = df["type"].map(size_map)
            
            # Добавляем ранги для релевантных сообщений
            df["ранг"] = 0
            for i, row in df[df["type"] == "релевантное"].iterrows():
                msg_id = row["id"]
                try:
                    rank = message_ids.index(msg_id) + 1
                    df.at[i, "ранг"] = rank
                except ValueError:
                    pass
            
            # Создаем интерактивный график
            fig = px.scatter(
                df, x="x", y="y", 
                color="type", 
                color_discrete_map=color_map,
                size="marker_size",
                size_max=15,
                hover_data=["id", "full_text", "ранг"],
                title=f"Карта эмбеддингов для запроса: '{query_text}' (топ-30 и {len(random_messages)} случайных)",
                labels={"x": "PCA компонент 1", "y": "PCA компонент 2", "type": "Тип сообщения"}
            )
            
            # Настройка отображения текста при наведении
            fig.update_traces(
                hovertemplate="<b>ID:</b> %{customdata[0]}<br><b>Ранг:</b> %{customdata[2]}<br><b>Текст:</b> %{customdata[1]}"
            )
            
            # Добавляем номера для топ-10 релевантных сообщений
            top_relevant = df[(df["type"] == "релевантное") & (df["ранг"] <= 10)]
            for i, row in top_relevant.iterrows():
                fig.add_annotation(
                    x=row["x"],
                    y=row["y"],
                    text=str(int(row["ранг"])),
                    showarrow=False,
                    font=dict(color="white", size=10),
                    bgcolor="rgba(0,0,0,0.7)",
                    bordercolor="red",
                    borderwidth=1
                )
            
            # Открываем график в браузере
            fig.show()
            
            logger.info("Визуализация эмбеддингов создана успешно")
            
        except Exception as e:
            logger.error(f"Ошибка при создании визуализации эмбеддингов: {e}")
            import traceback
            traceback.print_exc()

# Обновляем основную функцию для использования нового класса
async def main():
    # Обработка аргументов командной строки
    parser = argparse.ArgumentParser(description='FAISS поиск похожих сообщений с ранжированием')
    parser.add_argument('query', nargs='?', default=None, help='Начальный запрос для поиска')
    parser.add_argument('--sim', type=float, default=0.001, help='Вес семантического сходства (0-1)')
    parser.add_argument('--len', type=float, default=0.001, help='Вес длины текста (0-1)') 
    parser.add_argument('--rec', type=float, default=0.998, help='Вес свежести сообщения (0-1)')
    parser.add_argument('--debug', action='store_true', help='Включить режим отладки')
    parser.add_argument('--full', action='store_true', help='Выводить полный текст сообщений')
    parser.add_argument('--no-viz', action='store_true', help='Отключить автоматическую визуализацию')
    args = parser.parse_args()
    
    # Настройка уровня логирования для отладки
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Включен режим отладки")
    
    # Валидация весов
    total_weight = args.sim + args.len + args.rec
    if abs(total_weight - 1.0) > 0.01:  # Допускаем погрешность
        logger.warning(f"Общая сумма весов ({total_weight}) отличается от 1.0. Нормализуем веса.")
        # Нормализуем веса, чтобы их сумма была равна 1
        args.sim /= total_weight
        args.len /= total_weight
        args.rec /= total_weight
    
    # Инициализация компонентов
    db_processor = DatabaseProcessor(POSTGRES_ASYNC_URL)
    faiss_manager = FaissIndexManager()
    
    # Получение эмбеддингов и создание индекса
    logger.info("Получение эмбеддингов из базы данных...")
    message_embeddings = await db_processor.get_all_message_embeddings()
    
    if message_embeddings:
        logger.info(f"Найдено {len(message_embeddings)} эмбеддингов, создание FAISS индекса...")
        success = await faiss_manager.create_index(message_embeddings)
        if not success:
            logger.error("Не удалось создать FAISS индекс")
            return
    else:
        logger.error("Нет эмбеддингов для создания индекса")
        return
    
    # Создаем объект для поиска сообщений с ранжированием
    message_searcher = RankedMessageSearcher(
        db_processor, 
        faiss_manager,
        similarity_weight=args.sim,
        length_weight=args.len,
        recency_weight=args.rec,
        full_text=args.full
    )
    
    # Обновляем метрики для нормализации
    await message_searcher.update_metrics()
    
    # Если есть начальный запрос, обрабатываем его
    if args.query:
        print(f"\nВыполняем поиск по запросу: '{args.query}'")
        results = await message_searcher.search_messages(args.query)
        
        # Визуализация по умолчанию включена, если не указан флаг --no-viz
        if not args.no_viz:
            await message_searcher.visualize_embeddings(args.query)
        
        message_searcher.print_results(results)
    
    # Интерактивный режим для поиска
    print("\n=== FAISS Тестер для поиска с ранжированием ===")
    print(f"Веса: сходство={args.sim:.2f}, длина={args.len:.2f}, свежесть={args.rec:.2f}")
    print(f"Режим вывода: {'полный текст' if args.full else 'предпросмотр'}")
    print(f"Визуализация: {'отключена' if args.no_viz else 'включена'}")
    print("Введите текст запроса или 'выход' для завершения\n")
    
    while True:
        query = input("Ваш запрос: ")
        if query.lower() in ['выход', 'exit', 'quit']:
            break
            
        results = await message_searcher.search_messages(query)
        
        # Визуализация по умолчанию включена, если не указан флаг --no-viz
        if not args.no_viz:
            await message_searcher.visualize_embeddings(query)
            
        message_searcher.print_results(results)

if __name__ == "__main__":
    try:
        pd.set_option('display.max_colwidth', None)  # Не обрезать текстовые колонки
        pd.set_option('display.width', 120)  # Увеличить ширину отображения
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nПрограмма завершена пользователем")
    except Exception as e:
        logger.error(f"Непредвиденная ошибка: {e}")
        import traceback
        traceback.print_exc()