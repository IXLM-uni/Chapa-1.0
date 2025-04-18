import faiss
import logging
import numpy as np
import traceback
import asyncio
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta
import math
import pickle
import os

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
    def __init__(self, dimension: int = 1024, nlist: int = 100, m: int = 8, bits: int = 8, nprobe: int = 10, omp_threads: int = 4):
        self.dimension = dimension
        self.nlist = nlist
        self.m = m
        self.bits = bits
        self.nprobe = nprobe
        self.omp_threads = omp_threads
        self.index = None
        # Заменяем message_ids на vector_metadata для хранения (message_id, channel_id)
        self.vector_metadata: List[Tuple[int, int]] = []
        faiss.omp_set_num_threads(self.omp_threads)
        logger.info(f"FAISS инициализирован с {self.omp_threads} потоками. Параметры IVFPQ: nlist={nlist}, m={m}, bits={bits}, nprobe={nprobe}")
        
    async def train_index(self, training_embeddings: np.ndarray) -> bool:
        """
        Обучает индекс IndexIVFPQ на предоставленных векторах.

        Args:
            training_embeddings: Numpy массив векторов для обучения.

        Returns:
            bool: Успешно ли обучен индекс.
        """
        try:
            if training_embeddings.shape[1] != self.dimension:
                logger.error(f"Ошибка обучения: размерность векторов ({training_embeddings.shape[1]}) не совпадает с заданной ({self.dimension})")
                return False

            if len(training_embeddings) < self.nlist:
                 logger.warning(f"Количество векторов для обучения ({len(training_embeddings)}) меньше nlist ({self.nlist}). Качество индекса может быть низким.")
                 # Можно продолжить обучение, но лучше иметь больше данных

            logger.info(f"Начало обучения IndexIVFPQ на {len(training_embeddings)} векторах...")
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFPQ(quantizer, self.dimension, self.nlist, self.m, self.bits)

            # Используем to_thread для CPU-bound операции обучения
            await asyncio.to_thread(self.index.train, training_embeddings)

            self.index.nprobe = self.nprobe # Устанавливаем nprobe после обучения
            logger.info(f"Индекс IndexIVFPQ успешно обучен. nprobe установлен на {self.nprobe}. Индекс готов к добавлению векторов.")
            return True
        except Exception as e:
            logger.error(f"Ошибка при обучении FAISS индекса: {e}", exc_info=True)
            self.index = None # Сбрасываем индекс в случае ошибки
            return False

    async def add_embeddings(self, message_embeddings_data: List[Tuple[int, int, List[float]]]) -> bool:
        """
        Добавляет эмбеддинги с метаданными в обученный FAISS индекс.
        Заменяет старый create_index.

        Args:
            message_embeddings_data: Список кортежей (message_id, channel_id, embedding)

        Returns:
            bool: Успешно ли добавлены эмбеддинги
        """
        if self.index is None or not self.index.is_trained:
            logger.error("Ошибка добавления: индекс не существует или не обучен. Вызовите train_index() сначала.")
            return False

        try:
            logger.info(f"НАЧАЛО ДОБАВЛЕНИЯ В ИНДЕКС: Получено {len(message_embeddings_data)} эмбеддингов с метаданными")

            filtered_embeddings_with_meta = []
            # Фильтруем эмбеддинги по размерности
            for msg_id, channel_id, emb in message_embeddings_data:
                if emb and len(emb) == self.dimension:
                    filtered_embeddings_with_meta.append((msg_id, channel_id, emb))
                else:
                    logger.warning(f"Пропущен эмбеддинг для сообщения {msg_id} в канале {channel_id} при добавлении: неверная размерность {len(emb) if emb else 0} вместо {self.dimension}")

            logger.info(f"Отфильтровано {len(message_embeddings_data) - len(filtered_embeddings_with_meta)} некорректных эмбеддингов")

            if not filtered_embeddings_with_meta:
                logger.warning("После фильтрации не осталось валидных эмбеддингов для добавления")
                return True # Не ошибка, просто нечего добавлять

            # Извлекаем метаданные и эмбеддинги
            new_metadata = [(msg_id, channel_id) for msg_id, channel_id, _ in filtered_embeddings_with_meta]
            new_embeddings = [np.array(emb, dtype=np.float32) for _, _, emb in filtered_embeddings_with_meta]

            embeddings_matrix = np.vstack(new_embeddings).astype(np.float32)

            # Используем to_thread для CPU-bound операции добавления
            await asyncio.to_thread(self.index.add, embeddings_matrix)

            self.vector_metadata.extend(new_metadata)

            # Устанавливаем глобальные переменные после первого добавления (если нужно)
            # init_globals(vs=self)

            logger.info(f"В ИНДЕКС УСПЕШНО ДОБАВЛЕНО: {len(filtered_embeddings_with_meta)} векторов. Всего векторов: {self.index.ntotal}, метаданных: {len(self.vector_metadata)}")
            return True

        except Exception as e:
            logger.error(f"ОШИБКА при добавлении в FAISS индекс: {e}", exc_info=True)
            return False

    async def save_index(self, index_filepath: str, metadata_filepath: str) -> bool:
        """Сохраняет индекс и метаданные на диск."""
        if self.index is None:
            logger.error("Невозможно сохранить: индекс не существует.")
            return False
        try:
            logger.info(f"Сохранение FAISS индекса в {index_filepath}...")
            # Используем to_thread для I/O операции
            await asyncio.to_thread(faiss.write_index, self.index, index_filepath)
            logger.info(f"Сохранение метаданных в {metadata_filepath}...")
            with open(metadata_filepath, 'wb') as f:
                await asyncio.to_thread(pickle.dump, self.vector_metadata, f)
            logger.info("Индекс и метаданные успешно сохранены.")
            return True
        except Exception as e:
            logger.error(f"Ошибка при сохранении индекса/метаданных: {e}", exc_info=True)
            return False

    async def load_index(self, index_filepath: str, metadata_filepath: str) -> bool:
        """Загружает индекс и метаданные с диска."""
        try:
            if not os.path.exists(index_filepath) or not os.path.exists(metadata_filepath):
                logger.error(f"Ошибка загрузки: файлы не найдены ({index_filepath}, {metadata_filepath})")
                return False

            logger.info(f"Загрузка FAISS индекса из {index_filepath}...")
            # Используем to_thread для I/O операции
            self.index = await asyncio.to_thread(faiss.read_index, index_filepath)
            logger.info(f"Загрузка метаданных из {metadata_filepath}...")
            with open(metadata_filepath, 'rb') as f:
                self.vector_metadata = await asyncio.to_thread(pickle.load, f)

            self.index.nprobe = self.nprobe # Устанавливаем nprobe после загрузки
            logger.info(f"Индекс ({self.index.ntotal} векторов) и метаданные ({len(self.vector_metadata)}) успешно загружены. nprobe={self.nprobe}")

            # Обновляем глобальные переменные после загрузки
            init_globals(vs=self)
            return True
        except Exception as e:
            logger.error(f"Ошибка при загрузке индекса/метаданных: {e}", exc_info=True)
            self.index = None
            self.vector_metadata = []
            return False

    async def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5, 
        target_channel_ids: Optional[List[int]] = None
    ) -> Tuple[List[int], List[float]]:
        """
        Выполняет поиск ближайших эмбеддингов для заданного запроса, 
        с возможностью фильтрации по списку каналов.
        
        Args:
            query_embedding: Эмбеддинг запроса
            top_k: Количество результатов для возврата
            target_channel_ids: Список ID каналов для фильтрации (если None или пустой, ищет по всем)
            
        Returns:
            Tuple[List[int], List[float]]: Список ID сообщений и расстояний
        """
        # Определяем абсолютный путь к лог-файлу
        log_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "found_messages.txt")) # В корневой папке Chapa
        logger.info(f"Attempting to log FAISS results to: {log_file_path}") # Лог перед попыткой записи

        # --- Блок записи в лог-файл --- 
        try:
            with open(log_file_path, 'a', encoding='utf-8') as f:
                f.write(f"\n--- FAISS Search @ {datetime.now()} ---\
")
                # f.write(f"Query Embedding (first 10 dims): {query_embedding[:10]}\n") # Раскомментировать для отладки эмбеддинга
                f.write(f"Target Channel IDs: {target_channel_ids}\n")
                # Запись search_k и top_k должна быть после их определения
        except Exception as e_log:
            logger.error(f"Ошибка записи в лог-файл {log_file_path} (initial part): {e_log}", exc_info=True)
        # --------------------------------

        if self.index is None or self.index.ntotal == 0:
            logger.error("FAISS индекс не инициализирован или пуст")
            return [], []
            
        try:
            if not self.index.is_trained:
                logger.error("Поиск невозможен: индекс не обучен.")
                return [], []

            # Убедимся, что nprobe установлен (хотя он должен устанавливаться при train/load)
            if not hasattr(self.index, 'nprobe') or self.index.nprobe != self.nprobe:
                logger.warning(f"Устанавливаем nprobe на {self.nprobe} перед поиском")
                self.index.nprobe = self.nprobe

            query_vector = np.array([query_embedding], dtype=np.float32)
            
            results_message_ids = []
            results_distances = []

            # Проверяем, нужно ли фильтровать (список не None и не пустой)
            should_filter = target_channel_ids is not None and len(target_channel_ids) > 0

            # Определяем количество кандидатов для поиска в FAISS
            search_k = top_k
            if should_filter:
                search_k = min(max(top_k * 10, 50), self.index.ntotal) # Ищем больше кандидатов для фильтрации

            # Выполняем поиск в FAISS
            distances, indices = self.index.search(query_vector, search_k)
            
            # --- Продолжение логирования "сырых" результатов FAISS ---            
            try:
                with open(log_file_path, 'a', encoding='utf-8') as f:
                    f.write(f"Requested Top K: {top_k}, FAISS Search K: {search_k}\n") # Теперь search_k определен
                    f.write(f"FAISS Raw Results (Top {len(indices[0])}):\n")
                    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
                        if idx != -1 and idx < len(self.vector_metadata):
                            message_id, channel_id = self.vector_metadata[idx]
                            f.write(f"  {i+1}. Raw Index: {idx}, Msg ID: {message_id}, Chan ID: {channel_id}, Distance: {dist:.4f}\n")
                        else:
                             f.write(f"  {i+1}. Raw Index: {idx} (Invalid), Distance: {dist:.4f}\n")
            except Exception as e_log_results:
                logger.error(f"Ошибка записи в лог-файл {log_file_path} (results part): {e_log_results}", exc_info=True)
            # ---------------------------------------------

            if not should_filter:
                # Обработка без фильтрации
                for idx, dist in zip(indices[0], distances[0]):
                    if idx != -1 and idx < len(self.vector_metadata): # FAISS может вернуть -1
                        message_id, _ = self.vector_metadata[idx]
                        results_message_ids.append(message_id)
                        results_distances.append(dist)
            else:
                # Поиск с фильтрацией по списку каналов
                search_k = min(max(top_k * 10, 50), self.index.ntotal) # Ищем еще больше кандидатов
                
                # Преобразуем список ID в set для быстрой проверки
                target_channels_set = set(target_channel_ids) 
                
                for idx, dist in zip(indices[0], distances[0]):
                    if idx != -1 and idx < len(self.vector_metadata): # FAISS может вернуть -1
                        message_id, channel_id = self.vector_metadata[idx]
                        # Проверяем вхождение в set
                        if channel_id in target_channels_set:
                            results_message_ids.append(message_id)
                            results_distances.append(dist)
                            if len(results_message_ids) >= top_k:
                                break 
                                
            return results_message_ids, results_distances
            
        except Exception as e:
            logger.error(f"Ошибка при поиске в FAISS индексе (target_channel_ids={target_channel_ids}): {e}")
            traceback.print_exc()
            return [], []

    async def update_index(self, new_message_embeddings: List[Tuple[int, int, List[float]]]) -> bool:
        """
        Обновляет индекс (добавляет новые эмбеддинги) и метаданные.
        Примечание: для IVFPQ это просто вызов add_embeddings.

        Args:
            new_message_embeddings: Список кортежей (message_id, channel_id, embedding)

        Returns:
            bool: Успешность обновления
        """
        try:
            if not new_message_embeddings:
                logger.info("Нет новых эмбеддингов для обновления индекса")
                return True

            if self.index is None or not self.index.is_trained:
                logger.error("Невозможно обновить индекс: он не существует или не обучен.")
                return False

            # Просто вызываем новый метод добавления
            success = await self.add_embeddings(new_message_embeddings)
            # По хорошему, здесь или в bot.py нужно периодически вызывать save_index
            return success

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
            return {"status": "not_initialized", "vectors": 0, "metadata_entries": 0}
            
        return {
            "status": "active",
            "vectors": self.index.ntotal,
            "dimensions": self.index.d,
            "metadata_entries": len(self.vector_metadata) # Обновленный ключ и значение
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
                     similarity_weight=0.01, 
                     length_weight=0.01, 
                     recency_weight=0.98,
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
        
        async def search_messages(
            self, 
            query_text: str, 
            top_k: int = 5, 
            target_channel_ids: Optional[List[int]] = None
        ) -> List[Dict] | str:
            """
            Выполняет поиск похожих сообщений по тексту запроса с ранжированием.
            
            Args:
                query_text: Текст запроса
                top_k: Количество результатов
                target_channel_ids: Список ID каналов для фильтрации (если None, ищет по всем)
                
            Returns:
                List[Dict]: Список результатов поиска
            """
            # Получаем модель напрямую из глобальной области видимости
            embedding_model = get_embedding_model() # Используем get_embedding_model() напрямую

            if not embedding_model:
                # Обновляем сообщение в логе для ясности
                logger.error("Модель эмбеддинга не инициализирована (получено из get_embedding_model в search_messages)") 
                return "Модель для создания эмбеддингов не найдена. Невозможно выполнить поиск."
            
            # Убираем лишний else блок и дублирующуюся логику получения эмбеддинга
            query_text_with_prefix = f"query: {query_text}"
            query_embedding = await asyncio.to_thread(embedding_model.encode, query_text_with_prefix)

            candidate_k = min(top_k * 3, 20) # Оставляем как есть, т.к. faiss_manager.search сам увеличит k при фильтрации
            
            # Передаем target_channel_ids в search
            message_ids, distances = await self.faiss_manager.search(
                query_embedding, 
                top_k=candidate_k, 
                target_channel_ids=target_channel_ids
            )
            
            if not message_ids:
                # Логируем причину отсутствия результатов
                if target_channel_ids:
                     logger.warning(f"Не найдено похожих сообщений в указанных каналах: {target_channel_ids}")
                else:
                     logger.warning("Не найдено похожих сообщений во всем индексе")
                return []
            
            logger.info(f"Найдено {len(message_ids)} кандидатов для ранжирования (после фильтрации каналов, если была)...")
            
            candidates = []
            for i, (msg_id, distance) in enumerate(zip(message_ids, distances)):
                message_data = await self.db_processor.get_document_by_id(msg_id)
                if message_data:
                    message_text = message_data.get("text", "")
                    message_length = len(message_text) if message_text else 0
                    message_date = message_data.get("date")
                    
                    final_score = await self.calculate_final_score(
                        similarity=distance, 
                        length=message_length, 
                        date=message_date
                    )
                    
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
                        "message_link": message_link
                    })
                else:
                    logger.warning(f"Не удалось получить данные для сообщения ID: {msg_id}")
            
            ranked_results = sorted(candidates, key=lambda x: x["Итоговый_ранг"], reverse=True)
            top_results = ranked_results[:top_k]
            
            for i, result in enumerate(top_results):
                result["№"] = i + 1
            
            return top_results # Возвращаем список словарей в случае успеха
        
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