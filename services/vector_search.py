import faiss
import logging
import numpy as np
import traceback
import asyncio
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta
import math

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Делаем переменные приватными с префиксом _
_global_vector_search = None
_global_embedding_model = None

# Экспортируем функции для доступа к глобальным переменным
def get_vector_search():
    return _global_vector_search

def get_embedding_model():
    return _global_embedding_model

def init_globals(emb_model=None, vs=None):
    """
    Инициализирует глобальные переменные.
    """
    global _global_embedding_model, _global_vector_search
    
    if emb_model is not None:
        _global_embedding_model = emb_model
        logger.info("Глобальная переменная embedding_model инициализирована")
        
    if vs is not None:
        _global_vector_search = vs
        logger.info("Глобальная переменная vector_search инициализирована")

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
            # Добавляем подробное логирование
            logger.info(f"НАЧАЛО СОЗДАНИЯ ИНДЕКСА: Получено {len(message_embeddings)} эмбеддингов")
            
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
            
            # Глобальную переменную устанавливаем через функцию init_globals
            init_globals(vs=self)
            
            logger.info(f"ИНДЕКС УСПЕШНО СОЗДАН: Размерность: {dimension}, количество векторов: {self.index.ntotal}")
            return True
            
        except Exception as e:
            logger.error(f"ОШИБКА при создании FAISS индекса: {e}")
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
            traceback.print_exc()
            return [], []

    async def update_index(self, new_message_embeddings: List[Tuple[int, List[float]]]) -> bool:
        """
        Обновляет индекс новыми эмбеддингами.
        
        Args:
            new_message_embeddings: Список кортежей (id, embedding)
            
        Returns:
            bool: Успешность обновления
        """
        try:
            if not new_message_embeddings:
                logger.info("Нет новых эмбеддингов для обновления индекса")
                return True
                
            # Если индекс еще не создан, создаем его с новыми данными
            if self.index is None:
                return await self.create_index(new_message_embeddings)
            
            # Фильтруем эмбеддинги с правильной размерностью
            dimension = self.index.d
            filtered_embeddings = []
            
            for msg_id, emb in new_message_embeddings:
                if emb and len(emb) == dimension:
                    filtered_embeddings.append((msg_id, emb))
                else:
                    logger.warning(f"Пропущен эмбеддинг для сообщения {msg_id}: неверная размерность {len(emb) if emb else 0} вместо {dimension}")
            
            if not filtered_embeddings:
                logger.warning("После фильтрации не осталось валидных эмбеддингов для обновления")
                return True
            
            # Извлекаем ID и эмбеддинги
            new_ids = [msg_id for msg_id, _ in filtered_embeddings]
            new_embeddings = [np.array(emb, dtype=np.float32) for _, emb in filtered_embeddings]
            
            # Создаем матрицу из новых эмбеддингов
            new_embeddings_matrix = np.vstack(new_embeddings).astype(np.float32)
            
            # Добавляем новые векторы в индекс
            self.index.add(new_embeddings_matrix)
            
            # Обновляем список ID
            self.message_ids.extend(new_ids)
            
            logger.info(f"FAISS индекс обновлен. Добавлено {len(filtered_embeddings)} векторов. Всего векторов: {self.index.ntotal}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при обновлении FAISS индекса: {e}")
            traceback.print_exc()
            return False

    async def get_index_stats(self) -> dict:
        """
        Возвращает статистику индекса.
        
        Returns:
            dict: Статистика индекса
        """
        if self.index is None:
            return {"status": "not_initialized", "vectors": 0}
            
        return {
            "status": "active",
            "vectors": self.index.ntotal,
            "dimensions": self.index.d,
            "message_ids_count": len(self.message_ids)
        } 

# После класса FaissIndexManager добавляем класс для ранжирования

class VectorResearcher:
    """
    Класс для выполнения векторного поиска с ранжированием результатов.
    Содержит подкласс RankedMessageSearcher для ранжирования результатов.
    """
    
    def __init__(self, db_requests_instance):
        self.db_requests = db_requests_instance
        self.faiss_manager = get_vector_search()
        self.embedding_model = get_embedding_model()
        
        # Проверка наличия необходимых компонентов
        if not self.faiss_manager:
            logger.error("FAISS индекс не инициализирован")
        
        if not self.embedding_model:
            logger.error("Модель эмбеддингов не инициализирована")
        
        logger.info(f"VectorResearcher инициализирован: FAISS={bool(self.faiss_manager)}, Model={bool(self.embedding_model)}")
    
    class RankedMessageSearcher:
        def __init__(self, parent, 
                     similarity_weight=0.4, 
                     length_weight=0.1, 
                     recency_weight=0.5,
                     full_text=False):
            self.parent = parent
            self.db_processor = parent.db_requests
            self.faiss_manager = parent.faiss_manager
            
            # Веса для факторов ранжирования
            self.similarity_weight = similarity_weight
            self.length_weight = length_weight
            self.recency_weight = recency_weight
            
            # Режим вывода полного текста
            self.full_text = full_text
            
            # Статистические метрики для нормализации (статичные, как запрошено)
            self.avg_message_length = 300  
            self.max_message_length = 3000  
            self.min_date = datetime.now() - timedelta(days=365*2)  # предполагаем, что сообщения не старше 2 лет
            
            logger.info(f"RankedMessageSearcher инициализирован с весами: "
                        f"сходство={similarity_weight}, "
                        f"длина={length_weight}, "
                        f"свежесть={recency_weight}, "
                        f"полный текст={full_text}")
        
        async def normalize_similarity(self, similarity_score: float) -> float:
            """
            Нормализует оценку семантического сходства.
            FAISS возвращает расстояние (меньше = лучше), преобразуем в сходство.
            """
            # Для расстояния L2: преобразуем в оценку [0,1]
            # Типичные значения расстояний 0-10, поэтому делим на 10
            MAX_DISTANCE = 10.0
            distance = min(similarity_score, MAX_DISTANCE)
            return 1.0 - (distance / MAX_DISTANCE)
        
        async def calculate_length_score(self, text_length: int) -> float:
            """
            Вычисляет нормализованную оценку длины сообщения.
            Используем логарифмическую шкалу для смягчения преимущества очень длинных текстов.
            """
            if text_length <= 0:
                return 0.0
            
            # Логарифмическая нормализация длины
            normalized_length = math.log(1 + text_length) / math.log(1 + self.max_message_length)
            return min(normalized_length, 1.0)
        
        async def calculate_recency_score(self, date: datetime) -> float:
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
        
        async def calculate_final_score(self, similarity: float, length: int, date: datetime) -> float:
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
            similarity_score = await self.normalize_similarity(similarity)
            length_score = await self.calculate_length_score(length)
            recency_score = await self.calculate_recency_score(date)
            
            # Вычисляем взвешенную сумму
            final_score = (
                self.similarity_weight * similarity_score +
                self.length_weight * length_score +
                self.recency_weight * recency_score
            )
            
            logger.debug(f"Оценки: сходство={similarity_score:.3f}, длина={length_score:.3f}, "
                        f"свежесть={recency_score:.3f}, итого={final_score:.3f}")
            
            return final_score
        
        async def search_messages(self, query_text: str, top_k: int = 5) -> List[Dict]:
            """
            Выполняет поиск похожих сообщений по тексту запроса с ранжированием.
            
            Args:
                query_text: Текст запроса
                top_k: Количество результатов
                
            Returns:
                List[Dict]: Список результатов поиска
            """
            # Получение эмбеддинга запроса из родительского класса
            embedding_model = self.parent.embedding_model
            if embedding_model:
                query_text_with_prefix = f"query: {query_text}"
                query_embedding = await asyncio.to_thread(embedding_model.encode, query_text_with_prefix)
                query_embedding = query_embedding.tolist()
            else:
                logger.error("Модель эмбеддинга не инициализирована")
                return []
                
            # Поиск похожих сообщений (получаем больше, чем нужно, для последующего ранжирования)
            candidate_k = min(top_k * 3, 20)  # Берем в 3 раза больше кандидатов, но не более 20
            message_ids, distances = await self.faiss_manager.search(query_embedding, top_k=candidate_k)
            
            if not message_ids:
                logger.warning("Не найдено похожих сообщений")
                return []
            
            logger.info(f"Найдено {len(message_ids)} кандидатов, выполняем ранжирование...")
            
            # Формирование кандидатов для ранжирования
            candidates = []
            for i, (msg_id, distance) in enumerate(zip(message_ids, distances)):
                # Используем db_processor напрямую из родительского класса
                message_data = await self.db_processor.get_document_by_id(msg_id)
                if message_data:
                    # Получаем данные для расчета ранга
                    message_text = message_data.get("text", "")
                    message_length = len(message_text) if message_text else 0
                    message_date = message_data.get("date")
                    
                    # Вычисляем итоговый ранг
                    final_score = await self.calculate_final_score(
                        similarity=distance, 
                        length=message_length, 
                        date=message_date
                    )
                    
                    # Получаем ссылку на сообщение
                    message_link = message_data.get("message_link")
                    
                    candidates.append({
                        "№": i+1,
                        "ID": msg_id,
                        "Автор": message_data.get("author", "Н/Д"),
                        "Сообщение": message_text or "Текст отсутствует",
                        "Длина": message_length,
                        "Дата": message_date,
                        "Сходство": distance,
                        "Итоговый_ранг": final_score,
                        "Канал_ID": message_data.get("channel_id"),
                        "message_link": message_link  # Используем ссылку на сообщение
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
        
        async def format_search_results(self, results: List[Dict]) -> str:
            """
            Форматирует результаты поиска в строку для вывода.
            
            Args:
                results: Список результатов поиска
                
            Returns:
                str: Форматированная строка с результатами
            """
            if not results:
                return "Результаты не найдены или произошла ошибка"
                
            formatted_results = []
            formatted_results.append("=== Результаты поиска ===\n")
            
            for result in results:
                result_str = []
                result_str.append(f"[{result['№']}] ID: {result['ID']}, Ранг: {result['Итоговый_ранг']:.2f}")
                result_str.append(f"Автор: {result['Автор'] if result['Автор'] else 'Не указан'}")
                
                date_str = result['Дата'].strftime('%d.%m.%Y %H:%M') if result.get('Дата') else 'Н/Д'
                result_str.append(f"Дата: {date_str}, Длина: {result['Длина']} символов")
                
                # Сообщение (полностью или фрагмент)
                if self.full_text:
                    result_str.append(f"Сообщение:\n{result['Сообщение']}")
                else:
                    message_preview = result['Сообщение'][:200] + "..." if len(result['Сообщение']) > 200 else result['Сообщение']
                    result_str.append(f"Сообщение: {message_preview}")
                
                # Добавляем ссылку на сообщение
                if result.get('message_link'):
                    result_str.append(f"Сообщение: {result['message_link']}")
                else:
                    result_str.append(f"Сообщение ID: {result.get('ID', 'Н/Д')}")
                    
                formatted_results.append("\n".join(result_str))
                formatted_results.append("-" * 50)
                
            return "\n".join(formatted_results)
    
    async def get_ranked_searcher(self, similarity_weight=0.6, length_weight=0.2, recency_weight=0.2, full_text=False):
        """
        Создает и возвращает экземпляр RankedMessageSearcher.
        
        Args:
            similarity_weight: Вес семантического сходства
            length_weight: Вес длины текста
            recency_weight: Вес свежести
            full_text: Режим вывода полного текста
            
        Returns:
            RankedMessageSearcher: Экземпляр класса для ранжированного поиска
        """
        if not self.faiss_manager:
            logger.error("FAISS индекс не инициализирован")
            return None
            
        if not self.embedding_model:
            logger.error("Модель эмбеддингов не инициализирована")
            return None
            
        # Создаем экземпляр подкласса
        return self.RankedMessageSearcher(
            parent=self,
            similarity_weight=similarity_weight,
            length_weight=length_weight,
            recency_weight=recency_weight,
            full_text=full_text
        )

# Функция для удобного создания VectorResearcher
def init_vector_researcher(db_processor):
    """
    Инициализирует и возвращает экземпляр VectorResearcher.
    
    Args:
        db_processor: Экземпляр DatabaseRequests
        
    Returns:
        VectorResearcher: Экземпляр класса для векторного поиска
    """
    return VectorResearcher(db_processor)

# Функция для создания ранжированного поисковика
async def init_ranked_message_searcher(db_processor, similarity_weight=0.6, length_weight=0.2, recency_weight=0.2, full_text=False):
    """
    Создает и инициализирует объект RankedMessageSearcher.
    
    Args:
        db_processor: Объект для работы с БД
        similarity_weight: Вес семантического сходства
        length_weight: Вес длины текста
        recency_weight: Вес свежести
        full_text: Режим вывода полного текста
        
    Returns:
        RankedMessageSearcher: Инициализированный объект для поиска
    """
    # Создаем VectorResearcher
    researcher = init_vector_researcher(db_processor)
    
    if not researcher:
        logger.error("Не удалось создать VectorResearcher")
        return None
        
    # Получаем экземпляр RankedMessageSearcher
    return await researcher.get_ranked_searcher(
        similarity_weight=similarity_weight,
        length_weight=length_weight,
        recency_weight=recency_weight,
        full_text=full_text
    ) 