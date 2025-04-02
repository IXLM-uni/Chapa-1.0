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
from sqlalchemy import select, func
from database.models import ChannelPosts
from sentence_transformers import SentenceTransformer
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor

class TextProcessor:
    """Упрощенная обработка текстовых сообщений с использованием OpenVINO"""

    def __init__(self, nlp_model=None, embedding_model_name=None, db=None, 
                 ov_model_cache_dir="openvino_models", max_workers=None):
        self.nlp_model = nlp_model
        self.db = db
        
        # Инициализация модели эмбеддингов
        os.makedirs(ov_model_cache_dir, exist_ok=True)
        model_path = Path(ov_model_cache_dir) / embedding_model_name.replace('/', '_')
        
        # Динамически определим количество процессов позже, игнорируя max_workers
        print(f"Инициализация TextProcessor (параметр max_workers={max_workers} игнорируется)")
        
        self.model_name = embedding_model_name  # Сохраняем только имя
        # Не создаем модель здесь, а только при использовании
        
        # Логируем размер NLP модели, если она передана
        if nlp_model:
            nlp_size = self._get_size(nlp_model) / (1024 * 1024)
            print(f"[MEM] Размер NLP модели: {nlp_size:.2f} MB")
    
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
    
    async def process_messages(self, batch_size=25, limit=100, offset=0):
        """Запуск обработки сообщений пакетами с использованием ORM"""
        start_time = time.time()
        processed = 0
        
        mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
        print(f"[DEBUG] Начало process_messages: batch_size={batch_size}, limit={limit}, offset={offset}, RAM: {mem_before:.1f} MB")
        
        async with self.db.session() as session:
            for i in range(0, limit, batch_size):
                # Используем ORM вместо прямого SQL
                query = select(ChannelPosts).where(ChannelPosts.embedding == None).limit(batch_size).offset(offset + i)
                print(f"[DEBUG] Выполнение SQL запроса: limit={batch_size}, offset={offset + i}")
                result = await session.execute(query)
                messages = result.scalars().all()
                
                if not messages:
                    print(f"[DEBUG] Сообщений не найдено, выход из цикла")
                    break
                
                print(f"Обработка пакета из {len(messages)} сообщений")
                processed += len(messages)
                await self._process_message_batch(messages)
                print(f"Обработано {len(messages)} сообщений")
                gc.collect()  # Принудительный сбор мусора после обработки пакета
                mem_after = psutil.Process().memory_info().rss / (1024 * 1024)
                print(f"[DEBUG] RAM после обработки пакета: {mem_after:.1f} MB, Δ: {mem_after-mem_before:.1f} MB")
        
        elapsed = time.time() - start_time
        if processed > 0:
            print(f"Всего обработано {processed} сообщений за {elapsed:.2f} сек ({processed/elapsed:.2f} сообщ/сек)")
    
    async def _process_message_batch(self, messages):
        """Обработка одного пакета сообщений с использованием ORM"""
        # Подготовка текстов (игнорируем пустые)
        print(f"[DEBUG] _process_message_batch: получено {len(messages)} сообщений")
        valid_messages = [(msg.id, msg.message) for msg in messages if msg.message and msg.message.strip()]
        
        if not valid_messages:
            print(f"[DEBUG] Нет валидных сообщений для обработки")
            return
        
        print(f"[DEBUG] Подготовлено {len(valid_messages)} валидных сообщений для эмбеддинга")
        msg_ids, texts = zip(*valid_messages) if valid_messages else ([], [])
        
        # Логируем размер переменных
        texts_size = self._get_size(texts) / (1024 * 1024)
        print(f"[MEM] texts: {texts_size:.2f} MB")
        
        # Создание эмбеддингов
        print(f"[DEBUG] Вызов _batch_create_embeddings для {len(texts)} текстов")
        embeddings = await self._batch_create_embeddings(texts)
        
        # Логируем размер эмбеддингов
        emb_size = self._get_size(embeddings) / (1024 * 1024)
        print(f"[MEM] embeddings: {emb_size:.2f} MB")
        
        print(f"[DEBUG] Получено {len(embeddings)} эмбеддингов")
        
        # Очищаем переменную texts после использования
        del valid_messages
        gc.collect()
        
        # Обновление базы данных через ORM
        async with self.db.session() as session:
            for j, msg_id in enumerate(msg_ids):
                if j >= len(embeddings):
                    print(f"[WARNING] Индекс {j} вне диапазона эмбеддингов ({len(embeddings)})")
                    continue
                
                # Асинхронно извлекаем именованные сущности
                current_text = texts[j]
                named_entities = await self._extract_named_entities_async(current_text)
                print(f"[DEBUG] Именованные сущности для текста {j}: {named_entities}")
                
                # Находим сообщение по ID и обновляем
                post = await session.get(ChannelPosts, msg_id)
                if post:
                    post.embedding = embeddings[j]
                    post.key_words = named_entities
                    print(f"[DEBUG] Сохраняем эмбеддинг размера {len(embeddings[j])} и {len(named_entities)} именованных сущностей для ID {msg_id}")
                else:
                    print(f"[WARNING] Сообщение с ID={msg_id} не найдено в БД")
                
                # Очищаем переменные после каждой итерации
                del current_text, named_entities
                if j % 10 == 0:  # Периодически запускаем сборщик мусора
                    gc.collect()
            
            print(f"[DEBUG] Сохранение {len(msg_ids)} обновленных сообщений в БД")
            await session.commit()
            print(f"[DEBUG] Транзакция успешно завершена")
        
        # Очищаем большие переменные после использования
        del texts, embeddings, msg_ids
        gc.collect()
    
    async def _batch_create_embeddings(self, texts):
        """Создание эмбеддингов"""
        print(f"[DEBUG] _batch_create_embeddings: подготовка {len(texts)} текстов")
        prepared_texts = [f"passage: {text}" for text in texts]
        
        # Вызываем напрямую асинхронную функцию, вместо оборачивания в to_thread
        print(f"[DEBUG] Вызов _encode_texts для {len(prepared_texts)} текстов")
        result = await self._encode_texts(prepared_texts)
        
        # Очищаем большую переменную
        del prepared_texts
        gc.collect()
        
        print(f"[DEBUG] _encode_texts вернул результат типа {type(result)} длиной {len(result)}")
        return result
    
    async def _encode_texts(self, texts):
        """Асинхронная обертка для синхронной обработки эмбеддингов"""
        # Используем to_thread для синхронной части
        return await asyncio.to_thread(self._encode_texts_sync, texts)
    
    def _encode_texts_sync(self, texts):
        """Синхронная функция для вычисления эмбеддингов"""
        mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
        texts_size = self._get_size(texts) / (1024 * 1024)
        print(f"[MEM] texts в _encode_texts_sync: {texts_size:.2f} MB")
        
        try:
            # Проверяем и создаем модель, если еще не создана
            if not hasattr(self, 'embedding_model'):
                self.embedding_model = self._get_or_create_model()
            
            print(f"[DEBUG] Запуск encode для {len(texts)} текстов, RAM: {mem_before:.1f} MB")
            
            # Обрабатываем тексты небольшими пакетами, если их много
            batch_size = 8
            if len(texts) > 20:
                result = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    print(f"[BATCH] {i//batch_size+1}/{(len(texts)+batch_size-1)//batch_size}: {i}-{min(i+batch_size, len(texts))} из {len(texts)}")
                    batch_result = self.embedding_model.encode(batch, 
                                                 batch_size=batch_size,
                                                 normalize_embeddings=True, 
                                                 convert_to_numpy=True,
                                                 device="cpu").tolist()
                    result.extend(batch_result)
                    
                    # Логируем размер результата после каждого пакета
                    if i % 16 == 0:
                        result_size = self._get_size(result) / (1024 * 1024)
                        print(f"[MEM] result после {i+batch_size} текстов: {result_size:.2f} MB")
                    
                    del batch, batch_result
                    gc.collect()
            else:
                print(f"[BATCH] 1/1: 0-{len(texts)} из {len(texts)}")
                result = self.embedding_model.encode(texts, 
                                             batch_size=batch_size,
                                             normalize_embeddings=True, 
                                             convert_to_numpy=True,
                                             device="cpu").tolist()
            
            # Логируем размер результата
            result_size = self._get_size(result) / (1024 * 1024)
            print(f"[MEM] Итоговый result: {result_size:.2f} MB")
            
            gc.collect()  # Принудительный сбор мусора после создания эмбеддингов
            mem_after = psutil.Process().memory_info().rss / (1024 * 1024)
            print(f"[DEBUG] RAM после encode: {mem_after:.1f} MB, Δ: {mem_after-mem_before:.1f} MB")
            return result
        except Exception as e:
            print(f"[ERROR] Ошибка обработки: {e}")
            print(f"[ERROR] {traceback.format_exc()}")
            return [[] for _ in texts]
    
    def _get_or_create_model(self):
        if not hasattr(self, '_model'):
            # Очищаем память перед созданием модели
            gc.collect()
            mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
            self._model = SentenceTransformer(self.model_name, backend="openvino", 
                                           model_kwargs={"device": "cpu"})
            mem_after = psutil.Process().memory_info().rss / (1024 * 1024)
            model_size = self._get_size(self._model) / (1024 * 1024)
            print(f"[MEM] Размер модели эмбеддингов: {model_size:.2f} MB, RAM Δ: {mem_after-mem_before:.1f} MB")
        return self._model
    
    async def _extract_named_entities_async(self, text):
        """Асинхронное извлечение именованных сущностей через spaCy NER"""
        if not text or not self.nlp_model:
            return []
        
        # Используем asyncio.to_thread для выполнения синхронного кода в отдельном потоке
        return await asyncio.to_thread(self._extract_named_entities, text)

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
                    # Используем текст сущности в нижнем регистре
                    entities.append(ent.text.lower())
            
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

    async def start_background_processing(self, processing_interval=300, batch_size=100, chunks=50):
        """Запуск фоновой обработки сообщений с указанным интервалом"""
        print(f"Запуск фоновой обработки сообщений каждые {processing_interval} секунд")
        
        # Логируем размеры моделей при запуске
        if hasattr(self, 'nlp_model') and self.nlp_model:
            nlp_size = self._get_size(self.nlp_model) / (1024 * 1024)
            print(f"[MEM] Размер NLP модели при запуске: {nlp_size:.2f} MB")
        
        if hasattr(self, 'embedding_model'):
            emb_size = self._get_size(self.embedding_model) / (1024 * 1024)
            print(f"[MEM] Размер модели эмбеддингов при запуске: {emb_size:.2f} MB")
        
        async def background_processor():
            offset = 0
            while True:
                try:
                    mem = psutil.Process().memory_info().rss / (1024 * 1024)
                    print(f"Фоновая обработка: запуск с отступом {offset}, RAM: {mem:.1f} MB")
                    await self.process_messages(batch_size=batch_size, limit=batch_size*chunks, offset=offset)
                    
                    # Получаем количество необработанных сообщений
                    async with self.db.session() as session:
                        query = select(ChannelPosts).where(ChannelPosts.embedding == None)
                        result = await session.execute(query.limit(1))
                        has_more = result.scalars().first() is not None
                    
                    if not has_more:
                        print("Фоновая обработка: все сообщения обработаны, сброс отступа")
                        offset = 0
                    else:
                        offset += batch_size * chunks
                        
                    print(f"Фоновая обработка: ожидание {processing_interval} секунд до следующего запуска")
                    await asyncio.sleep(processing_interval)
                    
                except Exception as e:
                    print(f"Ошибка в фоновой обработке: {type(e).__name__}: {str(e)}")
                    print(f"[ERROR] {traceback.format_exc()}")
                    print("Повторная попытка через 60 секунд")
                    await asyncio.sleep(60)
        
        # Запускаем фоновую задачу
        task = asyncio.create_task(background_processor())
        return "Фоновая обработка запущена"