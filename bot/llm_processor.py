import os
import google.generativeai as genai
import logging
from typing import Dict, List, Optional, Set, Tuple
import asyncio
import math
from datetime import datetime
import time # <--- Импортируем time

from aiogram import Bot
from aiogram.types import Message
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from typing import Type, Dict, Annotated, Literal
from aiogram.fsm.context import FSMContext
from langchain_core.tools import InjectedToolArg, tool
from bot.handlers.start_handlers import show_chats_list
from bot.handlers.channel_add_handler import add_channel_start
from bot.handlers.channel_delete_handler import delete_channel_start
from sqlalchemy import text, select
from config import POSTGRES_ASYNC_URL, GEMINI_API_KEY
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from database.models import ChannelPosts, Entities, PostEntities
from services.vector_search import get_vector_search, get_embedding_model, FaissIndexManager, VectorResearcher
from database.requests import DatabaseRequests
import numpy as np
import spacy # Добавляем импорт spacy

# Глобальная переменная для хранения db_instance
db_instance = None
# Глобальные переменные для моделей
global_nlp_model = None
global_embedding_model = None

# Функция для установки экземпляра DatabaseRequests
def set_db_instance(instance):
    global db_instance
    logging.info(f"Установлен глобальный экземпляр DatabaseRequests: {instance}")
    db_instance = instance

# Функции для установки моделей
def set_nlp_model(model):
    global global_nlp_model
    logging.info("Глобальная переменная nlp_model установлена")
    global_nlp_model = model

def set_embedding_model(model):
    global global_embedding_model
    logging.info("Глобальная переменная embedding_model установлена")
    global_embedding_model = model

@tool
async def channel_action_tool(
    action: Literal["show_chats_list", "add_channel_start", "delete_channel_start"],
    message_text: Annotated[str, InjectedToolArg]
) -> str:
    """
    Определяет, какое действие с каналами нужно выполнить.
    
    Args:
        action: Имя функции для вызова: 
            - "show_chats_list" - показать список каналов
            - "add_channel_start" - добавить новый канал
            - "delete_channel_start" - удалить канал
        message_text: Текст сообщения пользователя

        верни только имя функции из списка action
    """
    return action

async def _extract_lemmas_from_text(text: str) -> List[str]:
    """Извлекает леммы (сущ., прил.) из текста с помощью глобальной NLP модели."""
    if not text or not global_nlp_model:
        return []
    try:
        doc = await asyncio.to_thread(global_nlp_model, text)
        lemmas = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and not token.is_space
            and len(token.lemma_) > 2
            and token.pos_ in {'NOUN', 'PROPN', 'ADJ'}
        ]
        return list(set(lemmas))
    except Exception as e:
        logger.error(f"Ошибка при извлечении лемм из текста: '{text[:50]}...': {e}", exc_info=True)
        return []

async def _extract_entities_with_llm(query_text: str) -> List[str]:
    """Использует LLM для извлечения ключевых сущностей из запроса."""
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=GEMINI_API_KEY)
        prompt = f"Извлеки ключевые слова (существительные, имена собственные) из следующего запроса, верни их списком Python: '{query_text}'"
        response = await llm.ainvoke(prompt)
        # Пытаемся извлечь список из ответа LLM
        content = response.content.strip()
        if content.startswith('[') and content.endswith(']'):
            try:
                # Используем eval с осторожностью, только для простого списка строк
                entities = eval(content)
                if isinstance(entities, list) and all(isinstance(item, str) for item in entities):
                    logger.info(f"LLM извлекла сущности: {entities}")
                    return entities
            except Exception as e:
                logger.warning(f"Не удалось распарсить ответ LLM как список: {content}, ошибка: {e}")
        logger.warning(f"LLM вернула не список сущностей: {content}")
        return []
    except Exception as e:
        logger.error(f"Ошибка при извлечении сущностей с помощью LLM: {e}", exc_info=True)
        return []

async def _rank_and_format_results(
    candidates: List[Tuple[int, float]], # Список (message_id, faiss_distance)
    query_text: str,
    top_k: int = 5
) -> str:
    """Ранжирует кандидатов, форматирует результат и генерирует саммари с LLM."""
    if not candidates:
        return "Не найдено подходящих сообщений."

    # Используем существующую логику ранжирования (предполагаем, что она доступна)
    # Нужно создать экземпляр VectorResearcher или импортировать его логику
    # В данном случае, создадим временный экземпляр RankedMessageSearcher
    # (нужен доступ к db_instance)
    if not db_instance:
        return "Ошибка: Экземпляр DatabaseRequests не инициализирован."

    # Используем общие настройки весов
    similarity_weight = 0.4
    length_weight = 0.1
    recency_weight = 0.5
    full_text = True # Для передачи в LLM нужен полный текст

    ranked_candidates_data = []
    researcher = VectorResearcher(db_instance)
    message_searcher = researcher.RankedMessageSearcher(
        parent=researcher, # Передаем созданный экземпляр
        similarity_weight=similarity_weight,
        length_weight=length_weight,
        recency_weight=recency_weight,
        full_text=full_text
    )

    for msg_id, distance in candidates:
        message_data = await db_instance.get_document_by_id(msg_id)
        if message_data:
            message_text = message_data.get("text", "")
            message_length = len(message_text) if message_text else 0
            message_date = message_data.get("date")
            message_link = message_data.get("message_link")

            final_score = await message_searcher.calculate_final_score(
                similarity=distance,
                length=message_length,
                date=message_date
            )

            ranked_candidates_data.append({
                "ID": msg_id,
                "Автор": message_data.get("author", "Н/Д"),
                "Сообщение": message_text or "Текст отсутствует",
                "Дата": message_date,
                "message_link": message_link,
                "Итоговый_ранг": final_score
            })
        else:
            logger.warning(f"Не удалось получить данные для ранжирования сообщения ID: {msg_id}")

    # Сортируем по рангу и берем топ-K
    ranked_results = sorted(ranked_candidates_data, key=lambda x: x["Итоговый_ранг"], reverse=True)
    top_results = ranked_results[:top_k]

    if not top_results:
        return "Не удалось найти релевантные сообщения после ранжирования."

    # Формируем контекст для LLM
    messages_data = []
    for result in top_results:
        message_content = result.get("Сообщение", "")
        author = result.get("Автор", "Неизвестный")
        message_link = result.get("message_link")

        message_entry = f"Сообщение от {author}: {message_content}"
        if message_link:
            message_entry += f"\nСсылка на сообщение: {message_link}"
        messages_data.append(message_entry)

    context = "\n\n".join(messages_data)

    # Генерируем ответ с LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7,
        google_api_key=GEMINI_API_KEY
    )

    prompt = f"""
Запрос пользователя: {query_text}

Задача:
0. Не используй в ответе разметку, она не работает
1. Игнорировать документы, не относящиеся к запросу, даже не упоминай их существование
2. 1 абзац - Составить краткое содержание новостей(30 слов), основанное ТОЛЬКО на релевантных документах
3. 2 и далее абзацы - Написать по порядку (1. 2. 3. и т.д.) через табуляцю документы ПОЛНОСТЬЮ, НЕ СОКРЩЗАЯ (если длинное сократи до 100 слов если длинные) 
4. После каждого сообщения предоставь ссылку к нему в формате https://t.me/htech_plus/1538

Найденные документы:
{context}
"""

    try:
        ai_response = await llm.ainvoke(prompt)
        return ai_response.content
    except Exception as e:
        logger.error(f"Ошибка при генерации ответа LLM: {e}", exc_info=True)
        return f"Произошла ошибка при генерации ответа: {str(e)}"

@tool
async def semantic_search_tool(
    query: str,
    message_text: Annotated[str, InjectedToolArg],
    target_channel_ids: Optional[List[int]] = None
) -> str:
    """
    Выполняет семантический поиск по сообщениям из заданных каналов (или всех) на основе запроса пользователя.
    Используется для общих запросов, когда неясны конкретные сущности.

    Примеры запросов:
    - "Найти сообщения про искусственный интеллект"
    - "Есть ли что-нибудь о новых технологиях в канале Tech?"
    - "Какие новости были о событиях вчера?"
    - "Расскажи про последние обновления в мире IT"

    Args:
        query: Поисковый запрос пользователя.
        message_text: Текст сообщения пользователя (автоматически передаётся).
        target_channel_ids: Список ID каналов для поиска. Если None или пустой, поиск по всем каналам пользователя.
    """
    start_time = time.time() # <--- Засекаем время начала
    try:
        # Логируем переданные каналы
        if target_channel_ids:
            logging.info(f"Выполняем семантический поиск по запросу: '{query}' в каналах: {target_channel_ids}")
        else:
            logging.info(f"Выполняем семантический поиск по запросу: '{query}' во всех каналах.")
            target_channel_ids = [] # Убедимся, что это пустой список, если None

        vector_search = get_vector_search()
        embedding_model = get_embedding_model() # Получаем глобальную модель

        if not vector_search or not vector_search.index:
            return "Векторный индекс не создан. Невозможно выполнить семантический поиск."

        if not embedding_model:
            return "Модель для создания эмбеддингов не найдена. Невозможно выполнить поиск."

        if not db_instance:
            return "Объект для работы с базой данных не инициализирован. Невозможно получить документы."

        search_query = query.strip() or message_text

        # 1. Получаем эмбеддинг запроса
        query_embedding = await asyncio.to_thread(embedding_model.encode, f"query: {search_query}")
        query_embedding = query_embedding.tolist()

        # 2. Ищем кандидатов в FAISS
        candidate_k = 15 # Ищем немного больше кандидатов для ранжирования
        message_ids, distances = await vector_search.search(
            query_embedding,
            top_k=candidate_k,
            target_channel_ids=target_channel_ids
        )

        if not message_ids:
            if target_channel_ids:
                return "Не удалось найти релевантные сообщения в указанных каналах по этому запросу."
            else:
                return "Не удалось найти релевантные сообщения по вашему запросу."

        candidates = list(zip(message_ids, distances))

        # 3. Ранжируем и форматируем результат
        result = await _rank_and_format_results(candidates, search_query, top_k=5)
        
    except Exception as e:
        logging.error(f"Ошибка при выполнении семантического поиска: {str(e)}", exc_info=True)
        result = f"Произошла ошибка при выполнении семантического поиска: {str(e)}"
    finally:
        end_time = time.time() # <--- Засекаем время окончания
        duration = end_time - start_time
        logging.info(f"semantic_search_tool выполнился за {duration:.4f} секунд")

    return result # <--- Возвращаем результат

#@tool
async def entity_semantic_search_tool(
    query: str,
    message_text: Annotated[str, InjectedToolArg],
    target_channel_ids: Optional[List[int]] = None
) -> str:
    """
    Выполняет поиск по сообщениям, сначала извлекая ключевые сущности из запроса,
    затем фильтруя сообщения по этим сущностям и ранжируя результаты семантически.
    Используется, когда запрос содержит конкретные объекты или темы.

    Примеры запросов:
    - "Что нового про компанию Tesla?"
    - "Найди информацию о Gemini 1.5 Flash в канале AI News"
    - "Расскажи про обновления iPhone"

    Args:
        query: Поисковый запрос пользователя.
        message_text: Текст сообщения пользователя (автоматически передаётся).
        target_channel_ids: Список ID каналов для поиска. Если None или пустой, поиск по всем каналам пользователя.
    """
    try:
        search_query = query.strip() or message_text
        logging.info(f"Выполняем поиск по сущностям для запроса: '{search_query}'")

        vector_search = get_vector_search()
        embedding_model = get_embedding_model()

        if not vector_search or not vector_search.index:
            return "Векторный индекс не создан. Невозможно выполнить поиск."
        if not embedding_model:
            return "Модель для создания эмбеддингов не найдена."
        if not db_instance:
            return "Объект для работы с базой данных не инициализирован."
        if not global_nlp_model:
            return "NLP модель не инициализирована для лемматизации."

        # 1. Извлекаем сущности с помощью LLM (или просто используем леммы запроса)
        # entities = await _extract_entities_with_llm(search_query)
        # Пока используем леммы из запроса для простоты
        lemmas = await _extract_lemmas_from_text(search_query)
        if not lemmas:
            logger.warning(f"Не удалось извлечь леммы/сущности из запроса: {search_query}. Выполняем обычный семантический поиск.")
            # Откатываемся к обычному семантическому поиску
            return await semantic_search_tool.ainvoke({"query": query, "message_text": message_text, "target_channel_ids": target_channel_ids})

        logger.info(f"Извлеченные леммы: {lemmas}")

        # 2. Находим ID сущностей в БД
        entity_ids = await db_instance.get_entity_ids_by_lemmas(lemmas)
        if not entity_ids:
            logger.warning(f"Не найдены сущности в БД для лемм: {lemmas}. Выполняем обычный семантический поиск.")
            return await semantic_search_tool.ainvoke({"query": query, "message_text": message_text, "target_channel_ids": target_channel_ids})

        logger.info(f"Найдены ID сущностей: {entity_ids}")

        # 3. Находим ID постов, связанных с этими сущностями
        relevant_post_ids_list = await db_instance.get_post_ids_by_entity_ids(entity_ids)
        if not relevant_post_ids_list:
            return f"Не найдено сообщений, связанных с сущностями: {', '.join(lemmas)}."
        relevant_post_ids = set(relevant_post_ids_list)
        logger.info(f"Найдено {len(relevant_post_ids)} ID постов, связанных с сущностями.")

        # 4. Получаем эмбеддинг запроса
        query_embedding = await asyncio.to_thread(embedding_model.encode, f"query: {search_query}")
        query_embedding = query_embedding.tolist()

        # 5. Ищем кандидатов в FAISS (больше, чем top_k, т.к. будем фильтровать)
        candidate_k = 50 # Ищем больше кандидатов
        message_ids, distances = await vector_search.search(
            query_embedding,
            top_k=candidate_k,
            target_channel_ids=target_channel_ids # Применяем фильтр каналов, если есть
        )

        if not message_ids:
            return "Не удалось найти похожие сообщения в индексе."

        # 6. Фильтруем кандидатов: оставляем только те, что связаны с нужными сущностями
        filtered_candidates = []
        for msg_id, dist in zip(message_ids, distances):
            if msg_id in relevant_post_ids:
                filtered_candidates.append((msg_id, dist))

        logger.info(f"После фильтрации по сущностям осталось {len(filtered_candidates)} кандидатов.")

        if not filtered_candidates:
            return f"Не найдено сообщений, семантически близких к запросу и связанных с сущностями: {', '.join(lemmas)}."

        # 7. Ранжируем и форматируем результат (используем общую функцию)
        return await _rank_and_format_results(filtered_candidates, search_query, top_k=5)

    except Exception as e:
        logging.error(f"Ошибка при выполнении поиска по сущностям: {str(e)}", exc_info=True)
        return f"Произошла ошибка при выполнении поиска по сущностям: {str(e)}"

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleGeminiProcessor:
    """Простой обработчик сообщений с Gemini API без дублирования функциональности aiogram."""

    def __init__(self, bot: Bot = None, nlp_model = None, embedding_model = None):
        # Используем ключ API из config
        api_key = GEMINI_API_KEY
        if not api_key:
            raise ValueError("Отсутствует ключ API Gemini (GEMINI_API_KEY)")
        genai.configure(api_key=api_key)
        
        # Создаем исходную модель и сохраняем ее
        self.original_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=1, 
            google_api_key=api_key
        )
        
        # Создаем модель с инструментами для использования в обычных сценариях
        # Регистрируем ОБЕ тулзы поиска
        self.llm = self.original_llm.bind_tools([
            channel_action_tool,
            semantic_search_tool,
            #entity_semantic_search_tool # Добавляем новую тулзу
        ])
        
        self.bot = bot
        self.llm_prompt_template = """Ты бот пока ты безрукий поэтому просто ответь"""
        self.llm_prompt = PromptTemplate(template=self.llm_prompt_template, input_variables=["text"])
        self.function_complete = False

        logger.info("SimpleGeminiProcessor инициализирован с Gemini Flash 1.5")
    
        # Устанавливаем глобальные модели, если они переданы
        if nlp_model:
            set_nlp_model(nlp_model)
        if embedding_model:
            set_embedding_model(embedding_model)
    
    async def process_message(self, user_id: int, message: Message, chat_history, state: FSMContext = None) -> str:
        """
        Обработать сообщение пользователя через Gemini API.
        """
        logger.info(f">>> НАЧАЛО process_message для user_id={user_id}")
        try:
            # Логируем входные данные
            logger.info(f"Текст сообщения: '{message.text[:100]}...'")
            logger.info(f"Наличие chat_history: {chat_history is not None}")
            
            # Передаем текст сообщения в LLM с инструментами
            logger.info("Отправляем запрос в LLM с инструментами...")
            # --- ОБНОВЛЕННЫЙ ПРОМПТ ---
            prompt = f"""Ты - многофункциональный ассистент. Твоя задача - помочь пользователю. У тебя есть доступ к инструментам для выполнения специфических задач.

Проанализируй сообщение пользователя и реши, что делать:

1.  **Управление каналами:** Если пользователь явно просит показать список каналов, добавить новый или удалить существующий, вызови инструмент 'channel_action_tool' с соответствующим 'action'.
2.  **Поиск информации:** Если пользователь ищет конкретную информацию, новости или данные, которые могут содержаться в сообщениях каналов:
    а) Переформулируй его запрос в краткий и емкий поисковый запрос (например, "новости про AI", "обзор iPhone 15", "рецепт пиццы").
    б) Вызови инструмент 'semantic_search_tool', передав этот *переформулированный* запрос в аргумент 'query'.
3.  **Обычная беседа:** Если пользователь просто хочет поговорить, задает общие вопросы, приветствует тебя, или его сообщение не требует использования инструментов (управления каналами или поиска информации в них) - **НЕ ВЫЗЫВАЙ ИНСТРУМЕНТЫ**. Просто ответь на его сообщение как разговорный ИИ, используя свои знания и историю диалога (если она есть).

**Примеры:**
    - Пользователь: "Привет" -> Ответ: "Привет! Чем могу помочь?" (Без инструмента)
    - Пользователь: "Покажи мои каналы" -> Инструмент: channel_action_tool(action='show_chats_list')
    - Пользователь: "Найди что-нибудь про квантовые компьютеры" -> Инструмент: semantic_search_tool(query='квантовые компьютеры новости')
    - Пользователь: "Как твои дела?" -> Ответ: "Я - компьютерная программа, у меня нет дел. А как ваши?" (Без инструмента)
    - Пользователь: "Я не хочу выбирать канал, просто хочу поговорить" -> Ответ: "Хорошо, о чем бы вы хотели поговорить?" (Без инструмента)

**Сообщение пользователя:** '{message.text}'"""
            # --- КОНЕЦ ОБНОВЛЕННОГО ПРОМПТА ---

            logger.info(f"Промпт: {prompt[:200]}...") # Логируем начало промпта

            ai_msg = await self.llm.ainvoke(
                prompt,
                {"message_text": message.text} # message_text все еще нужен для InjectedToolArg, даже если не используется явно в промпте
            )

            logger.info(f"Получен ответ от LLM, есть tool_calls: {bool(ai_msg.tool_calls)}")
            logger.info(f"Размер content: {len(ai_msg.content) if ai_msg.content else 'None'} символов")
            if ai_msg.content:
                logger.info(f"Начало content: '{ai_msg.content[:100]}...'")

            # Если есть вызовы инструментов, то есть был определен интент
            if ai_msg.tool_calls:
                logger.info(f"Выполнен вызов инструмента: {ai_msg.tool_calls}")
                
                tool_name = ai_msg.tool_calls[0]['name']
                logger.info(f"Имя инструмента: {tool_name}")
                
                if tool_name == "channel_action_tool":
                    function_name = ai_msg.content
                    logger.info(f"Определено действие: {function_name}")
                    
                    # Извлекаем значение action из args
                    action = ai_msg.tool_calls[0]['args']['action']
                    logger.info(f"Извлечено действие: {action}")
                    
                    # Возвращаем имя функции и флаг function_complete=True
                    return action, True
                
                elif tool_name == "semantic_search_tool" or tool_name == "entity_semantic_search_tool": # Обрабатываем обе тулзы поиска
                    logger.info(f"Обрабатываем вызов инструмента: {tool_name}")
                    # Извлекаем значение query из args
                    query = ai_msg.tool_calls[0]['args'].get('query', message.text) # Берем query или исходное сообщение
                    logger.info(f"Извлечен запрос: {query[:100]}...")

                    # Вызываем соответствующую тулзу с (теперь уже) извлеченным запросом
                    tool_to_call = semantic_search_tool if tool_name == "semantic_search_tool" else entity_semantic_search_tool

                    result = await tool_to_call.ainvoke({
                        "query": query, # <--- Используем извлеченный query 
                        "message_text": message.text, # message_text все еще нужен для InjectedToolArg
                        "target_channel_ids": [] # Пока поиск только по всем каналам
                    })
                    logger.info(f"Получен результат запроса ({tool_name}) длиной {len(result)} символов")
                    
                    # Проверка на пустой результат
                    if not result or len(result.strip()) == 0:
                        logger.warning("Получен пустой результат семантического поиска")
                        return "К сожалению, не удалось найти информацию по вашему запросу.", False
                    
                    # Возвращаем результат и флаг function_complete=False
                    return result, False
            
            # Если нет вызовов инструментов, обрабатываем как обычное сообщение
            logger.info("Обрабатываем как обычное сообщение (инструмент не вызван)...")
            if chat_history:
                logger.info("Добавляем сообщение в историю чата...")
                await chat_history.aadd_messages([HumanMessage(content=message.text)])
                
                logger.info("Получаем все сообщения из истории...")
                all_messages = await chat_history.aget_messages()
                logger.info(f"Получено {len(all_messages)} сообщений из истории")
                
                # Фильтруем сообщения, чтобы исключить пустые
                filtered_messages = []
                for msg in all_messages:
                    # Проверяем, что контент не пустой
                    if hasattr(msg, 'content') and msg.content and isinstance(msg.content, str) and len(msg.content.strip()) > 0:
                        filtered_messages.append(msg)
                    else:
                        logger.warning(f"Пропущено пустое или некорректное сообщение в истории: {msg}")
                
                # Если после фильтрации остались сообщения, используем их
                if filtered_messages:
                    logger.info(f"Отправляем {len(filtered_messages)} отфильтрованных сообщений в LLM")
                    
                    # Ограничиваем историю последними 10 сообщениями
                    if len(filtered_messages) > 10:
                        filtered_messages = filtered_messages[-10:]
                    
                    # Проверяем, есть ли контент в последнем сообщении от пользователя
                    if not message.text or not message.text.strip():
                        return "Пожалуйста, введите текст вашего запроса.", False
                        
                    try:
                        # Создаем системный промпт для контекста
                        system_message = HumanMessage(content="Ты информационный ассистент, который помогает пользователю с вопросами и запросами. Отвечай детально и полезно, ищи информацию в своей базе знаний.")
                        
                        # Добавляем системный промпт в начало сообщений
                        messages_with_context = [system_message] + filtered_messages
                        
                        # Используем модель без инструментов для прямого ответа
                        response = await self.original_llm.ainvoke(messages_with_context)
                        
                        # Проверяем наличие контента в ответе
                        if not response.content or len(response.content.strip()) == 0:
                            logger.warning("LLM вернула пустой ответ")
                            error_message = "Извините, не удалось сформировать ответ на ваш вопрос. Пожалуйста, попробуйте перефразировать запрос."
                            await chat_history.aadd_messages([AIMessage(content=error_message)])
                            return error_message, False
                        
                        logger.info(f"Получен ответ размером {len(response.content)} символов")
                        
                        logger.info("Добавляем ответ в историю чата...")
                        await chat_history.aadd_messages([AIMessage(content=response.content)])
                        
                        logger.info(">>> КОНЕЦ process_message - возвращаем ответ из истории")
                        return response.content, False
                        
                    except Exception as e:
                        logger.error(f"Ошибка при вызове LLM: {str(e)}", exc_info=True)
                        fallback_message = "Извините, произошла ошибка при обработке вашего запроса. Попробуйте позже или измените формулировку."
                        await chat_history.aadd_messages([AIMessage(content=fallback_message)])
                        return fallback_message, False
                else:
                    logger.warning("После фильтрации не осталось валидных сообщений. Используем только текущее сообщение.")
                    # Если после фильтрации не осталось сообщений, используем только текущее
                    if not message.text or not message.text.strip():
                        return "Пожалуйста, введите текст вашего запроса.", False
                        
                    try:
                        # Добавляем четкую инструкцию для модели
                        enhanced_prompt = f"Запрос пользователя: {message.text}\n\nПожалуйста, предоставь информативный и полезный ответ на этот запрос."
                        
                        response = await self.original_llm.ainvoke(enhanced_prompt)
                        
                        # Проверяем наличие контента в ответе
                        if not response.content or len(response.content.strip()) == 0:
                            logger.warning("LLM вернула пустой ответ")
                            return "Извините, не удалось сформировать ответ на ваш вопрос. Пожалуйста, попробуйте перефразировать запрос.", False
                        
                        logger.info(f"Получен ответ размером {len(response.content)} символов")
                        
                        logger.info(">>> КОНЕЦ process_message - возвращаем прямой ответ")
                        return response.content, False
                        
                    except Exception as e:
                        logger.error(f"Ошибка при вызове LLM без истории: {str(e)}", exc_info=True)
                        return "Извините, произошла ошибка при обработке вашего запроса. Попробуйте позже или измените формулировку.", False
            else:
                logger.info("Вызываем LLM без истории...")
                try:
                    # Добавляем системный промпт для контекста
                    enhanced_prompt = f"Вопрос: {message.text}\n\nПожалуйста, дай полный и информативный ответ."
                    
                    response = await self.original_llm.ainvoke(enhanced_prompt)
                    
                    # Проверяем наличие контента в ответе
                    if not response.content or len(response.content.strip()) == 0:
                        logger.warning("LLM вернула пустой ответ")
                        return "Извините, не удалось сформировать ответ на ваш вопрос. Пожалуйста, попробуйте перефразировать запрос.", False
                    
                    logger.info(f"Получен ответ размером {len(response.content)} символов")
                    
                    logger.info(">>> КОНЕЦ process_message - возвращаем прямой ответ")
                    return response.content, False
                    
                except Exception as e:
                    logger.error(f"Ошибка при вызове LLM без истории: {str(e)}", exc_info=True)
                    return "Извините, произошла ошибка при обработке вашего запроса. Попробуйте позже или измените формулировку.", False
            
        except Exception as e:
            logger.error(f"Ошибка в Gemini API: {str(e)}", exc_info=True)
            # Возвращаем сообщение об ошибке и False для флага function_complete
            return f"Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте другую формулировку или обратитесь позже.", False