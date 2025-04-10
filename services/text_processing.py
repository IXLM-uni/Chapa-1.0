import asyncio
import gc
import os
import sys
import traceback
import psutil
import requests
import json
import time
from pathlib import Path
from sqlalchemy import select, func, update
from database.models import ChannelPosts
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import OpenVINOEmbeddings
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor
from database.requests import DatabaseRequests
import spacy
from typing import List, Dict
import logging
import torch

logger = logging.getLogger(__name__)

class TextProcessor:
    """Класс для фоновой обработки текстовых сообщений: извлечение лемм и создание эмбеддингов."""

    def __init__(self, db: DatabaseRequests, nlp_model: spacy.language.Language,
                 embedding_model_name: str,
                 ov_model_cache_dir="openvino_models", max_workers=None):
        self.db = db
        self.nlp_model = nlp_model
        self.embedding_model_name = embedding_model_name
        self.ov_model_cache_dir = Path(ov_model_cache_dir)
        self.ov_model_path = self.ov_model_cache_dir / self.embedding_model_name.replace('/', '_')
        self.current_embedding_model = None # Модель будет загружена лениво
        self.device = None # Устройство будет определено лениво
        
        os.makedirs(self.ov_model_cache_dir, exist_ok=True)
        
        # Логируем размеры NLP модели
        nlp_size = self._get_size(nlp_model) / (1024 * 1024)
        print(f"[MEM] TextProcessor: Размер NLP модели: {nlp_size:.2f} MB")
        
        print(f"Инициализация TextProcessor (ленивая загрузка модели) для {self.embedding_model_name}")
        
    def _get_size(self, obj, seen=None):
        """Рекурсивно получает размер объекта в байтах"""
        size = sys.getsizeof(obj)
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)
        if isinstance(obj, dict):
            size += sum(self._get_size(v, seen) for v in obj.values())
            size += sum(self._get_size(k, seen) for k in obj.keys())
        elif hasattr(obj, '__dict__'):
            size += self._get_size(obj.__dict__, seen)
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
            try:
                size += sum(self._get_size(i, seen) for i in obj)
            except:
                pass
        return size
    
    def _sync_extract_lemmas(self, text: str) -> List[str]:
        """Синхронная функция для извлечения лемм (без глаголов) с помощью spaCy."""
        if not text or not self.nlp_model:
            return []
        try:
            doc = self.nlp_model(text)
            lemmas = [
                token.lemma_.lower()
                for token in doc
                if not token.is_stop
                and not token.is_punct
                and not token.is_space
                and len(token.lemma_) > 2
                and token.pos_ in {'NOUN', 'PROPN', 'ADJ'} # Исключаем 'VERB'
            ]
            return list(set(lemmas))
        except Exception as e:
            logger.error(f"Sync Error in _sync_extract_lemmas for text: '{text[:50]}...': {e}", exc_info=True)
            return []
        
    async def _extract_lemmas(self, text: str) -> List[str]:
        """Асинхронно извлекает леммы, запуская CPU-bound spaCy в потоке."""
        if not text or not self.nlp_model:
            return []
        try:
            lemmas = await asyncio.to_thread(self._sync_extract_lemmas, text)
            return lemmas
        except Exception as e:
            logger.error(f"Async Error in _extract_lemmas wrapper for text: '{text[:50]}...': {e}", exc_info=True)
            return []

    def _load_embedding_model(self):
        """Загружает модель эмбеддингов в зависимости от доступного устройства."""
        if self.current_embedding_model:
            return # Модель уже загружена

        # --- Выбор устройства и загрузка модели --- 
        if torch.cuda.is_available():
            self.device = 'cuda'
            print("[TextProcessor] CUDA доступен, загружаем SentenceTransformer модель на GPU...")
            try:
                self.current_embedding_model = SentenceTransformer(
                    model_name_or_path=self.embedding_model_name,
                    device=self.device
                )
                print("[TextProcessor] SentenceTransformer модель загружена.")
            except Exception as e:
                logger.error(f"Ошибка при загрузке SentenceTransformer: {e}", exc_info=True)
                raise RuntimeError("Не удалось загрузить модель SentenceTransformer") from e
        else:
            self.device = 'cpu'
            print("[TextProcessor] CUDA недоступен, используем OpenVINO на CPU...")
            try:
                # OpenVINOEmbeddings сама обработает загрузку/экспорт/кэширование
                model_kwargs = {"device": "CPU"} # Укажите "GPU", если есть Intel GPU и хотите его использовать
                encode_kwargs = {"normalize_embeddings": True}
                
                print(f"[TextProcessor] Инициализация OpenVINOEmbeddings для модели: {self.embedding_model_name}")
                print(f"[TextProcessor] Модель будет сохранена/загружена из: {self.ov_model_path}")
                
                self.current_embedding_model = OpenVINOEmbeddings(
                    model_name_or_path=self.embedding_model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs,
                )
                # Вызов embed_query для инициализации/загрузки/конвертации
                print("[TextProcessor] Инициализация OpenVINO модели (может занять время при первом запуске)...")
                _ = self.current_embedding_model.embed_query("Initialization query")
                print("[TextProcessor] OpenVINO модель инициализирована.")
                
            except Exception as e:
                logger.error(f"Ошибка при инициализации OpenVINOEmbeddings: {e}", exc_info=True)
                raise RuntimeError("Не удалось инициализировать модель OpenVINOEmbeddings") from e

    def _sync_create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Синхронная функция для вычисления эмбеддингов батчами."""
        if not texts:
            return []
        try:
            # Ленивая загрузка модели при первом вызове
            self._load_embedding_model()

            # Используем загруженную модель self.current_embedding_model
            # Она будет либо SentenceTransformer, либо OpenVINOEmbeddings
            print(f"[TextProcessor] Вычисление эмбеддингов на {self.device}...")
            
            # OpenVINOEmbeddings ожидает список текстов
            # SentenceTransformer тоже работает со списком
            all_embeddings = self.current_embedding_model.embed_documents(texts)
            
            print(f"[TextProcessor] Эмбеддинги вычислены для {len(texts)} текстов.")

            # Очистка памяти GPU, если используется
            if self.device == 'cuda':
                torch.cuda.empty_cache()

            return all_embeddings
        except Exception as e:
            logger.error(f"Ошибка при создании эмбеддингов: {e}", exc_info=True)
            return [[] for _ in texts]

    async def _batch_create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Асинхронно создает эмбеддинги, запуская синхронную функцию в потоке."""
        if not texts:
            return []
        return await asyncio.to_thread(self._sync_create_embeddings, texts)

    async def run_background_processing_loop(self, interval: int = 60, batch_size: int = 20):
        """Основной цикл фоновой обработки сообщений (леммы + эмбеддинги)."""
        print(f"[TextProcessor] Запуск фоновой обработки сообщений (каждые {interval} сек, батч {batch_size})...")
        while True:
            processed_in_cycle = 0
            try:
                # 1. Получаем порцию необработанных сообщений
                posts_batch = await self.db.get_unprocessed_posts(batch_size)

                if not posts_batch:
                    # print(f"[TextProcessor] Нет необработанных сообщений. Пауза...")
                    await asyncio.sleep(interval)
                    continue

                print(f"[TextProcessor] Получено {len(posts_batch)} сообщений для обработки.")
                texts_to_process = [post.message for post in posts_batch if post.message]
                post_ids_map = {i: post.id for i, post in enumerate(posts_batch) if post.message}

                if not texts_to_process:
                    # Помечаем посты без текста как обработанные
                    ids_to_mark = [post.id for post in posts_batch]
                    if ids_to_mark:
                       await self.db.mark_posts_as_processed(ids_to_mark)
                    print(f"[TextProcessor] В батче нет сообщений с текстом. Помечено {len(ids_to_mark)} постов как обработанные.")
                    await asyncio.sleep(1) # Короткая пауза перед следующей итерацией
                    continue

                # 2. Создаем эмбеддинги для всех текстов в батче
                # TODO: Реализовать OpenVINO/GPU, пока используем CPU
                print(f"[TextProcessor] Создание эмбеддингов для {len(texts_to_process)} текстов...")
                embeddings = await self._batch_create_embeddings(texts_to_process)
                print(f"[TextProcessor] Эмбеддинги созданы.")

                processed_ids = []
                entities_processed_count = 0
                embeddings_updated_count = 0
                
                # 3. Обрабатываем каждый пост из батча
                for i, post in enumerate(posts_batch):
                    if post.id not in post_ids_map.values(): # Пропускаем посты без текста
                        continue
                    
                    original_index = -1
                    for idx, pid in post_ids_map.items():
                         if pid == post.id:
                              original_index = idx
                              break
                         
                    if original_index == -1:
                         logger.warning(f"Не найден индекс для post.id {post.id}")
                         continue
                         
                    try:
                        # --- Извлекаем и сохраняем сущности (леммы) ---
                        lemmas = await self._extract_lemmas(post.message)
                        if lemmas:
                            await self.db.add_entities_for_post(post.id, lemmas)
                            entities_processed_count += 1

                        # --- Обновляем эмбеддинг (если он был успешно создан) ---
                        if original_index < len(embeddings) and embeddings[original_index]:
                            await self.db.update_post_embedding(post.id, embeddings[original_index])
                            embeddings_updated_count += 1
                        else:
                             logger.warning(f"Эмбеддинг для поста {post.id} не создан или пуст.")
                        
                        # Добавляем ID в список для пометки как обработанный
                        processed_ids.append(post.id)
                        processed_in_cycle +=1
                    
                    except Exception as e:
                        logger.error(f"Ошибка при обработке поста ID {post.id}: {e}", exc_info=True)
                        # Не добавляем в processed_ids, чтобы повторить попытку позже

                # 4. Помечаем успешно обработанные посты
                if processed_ids:
                    await self.db.mark_posts_as_processed(processed_ids)
                    print(f"[TextProcessor] Успешно обработано: {len(processed_ids)} постов (сущностей: {entities_processed_count}, эмбеддингов: {embeddings_updated_count}).")
                
                # Очистка
                del posts_batch, texts_to_process, embeddings, processed_ids, post_ids_map
                gc.collect()

                # Короткая пауза перед следующей выборкой, если обработали что-то
                if processed_in_cycle > 0:
                     await asyncio.sleep(1)
                else: # Если ничего не обработали (например, из-за ошибок), ждем дольше
                     print(f"[TextProcessor] В этом цикле не обработано ни одного поста. Пауза {interval} сек...")
                     await asyncio.sleep(interval)

            except asyncio.CancelledError:
                print("[TextProcessor] Фоновая обработка остановлена.")
                break
            except Exception as e:
                logger.error(f"[TextProcessor] Критическая ошибка в цикле фоновой обработки: {e}", exc_info=True)
                print(f"[TextProcessor] Пауза на 300 секунд после критической ошибки...")
                await asyncio.sleep(300) # Длительная пауза при серьезной ошибке