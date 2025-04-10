import os
import google.generativeai as genai
import logging
from typing import Dict
from aiogram import Bot
from aiogram.types import Message
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from typing import Optional, Type
from typing import Dict, Annotated, Literal
from aiogram.fsm.context import FSMContext
from langchain_core.tools import InjectedToolArg, tool
from bot.handlers.start_handlers import show_chats_list, add_channel_start
from bot.handlers.channel_delete_handler import delete_channel_start
from sqlalchemy import text, select
from config import POSTGRES_ASYNC_URL
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from database.models import ChannelPosts
from services.vector_search import get_vector_search, get_embedding_model
from database.requests import DatabaseRequests

# Глобальная переменная для хранения db_instance
db_instance = None

# Функция для установки экземпляра DatabaseRequests
def set_db_instance(instance):
    global db_instance
    logging.info(f"Установлен глобальный экземпляр DatabaseRequests: {instance}")
    db_instance = instance

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

@tool
async def semantic_search_tool(
    query: str,
    message_text: Annotated[str, InjectedToolArg]) -> str:
    """
    Выполняет семантический поиск по сообщениям из каналов на основе запроса пользователя.
    
    Примеры запросов:
    - "Найти сообщения про искусственный интеллект"
    - "Есть ли что-нибудь о новых технологиях?"
    - "Какие новости были о событиях вчера?"
    - "Расскажи про последние обновления в мире IT"
    
    Args:
        query: Поисковый запрос поальзователя
        message_text: Текст сообщения пользователя (автоматически передаётся)
    """
    try:
        logging.info(f"Выполняем семантический поиск по запросу: {query}")
        
        # Получаем актуальные значения через функции
        vector_search = get_vector_search()
        embedding_model = get_embedding_model()
        
        # Добавим диагностическое логирование
        logging.info(f"Состояние vector_search: {vector_search}")
        logging.info(f"Состояние db_instance: {db_instance}")
        
        if vector_search:
            logging.info(f"Состояние индекса: {vector_search.index is not None}")
            if vector_search.index:
                logging.info(f"Размер индекса: {vector_search.index.ntotal} векторов")
        
        if not vector_search or not vector_search.index:
            return "Векторный индекс не создан. Невозможно выполнить семантический поиск."
        
        if not embedding_model:
            return "Модель для создания эмбеддингов не найдена. Невозможно выполнить поиск."
        
        if not db_instance:
            return "Объект для работы с базой данных не инициализирован. Невозможно получить документы."
            
        # Создаем экземпляр класса ранжирования внутри VectorResearcher
        from services.vector_search import init_ranked_message_searcher
        
        # Инициализируем поисковик с ранжированием
        message_searcher = await init_ranked_message_searcher(
            db_processor=db_instance,
            similarity_weight=0.4,  # Вес семантического сходства
            length_weight=0.1,      # Вес длины текста
            recency_weight=0.,     # Вес свежести сообщения
            full_text=True          # Показывать полный текст
        )
        
        if not message_searcher:
            return "Не удалось инициализировать механизм поиска с ранжированием."
        
        # Используем query вместо message_text для фактического поиска
        search_query = query.strip() or message_text  # query как основной, message_text как запасной вариант
        
        # Выполняем поиск с ранжированием
        ranked_results = await message_searcher.search_messages(search_query, top_k=5)
        
        if not ranked_results:
            return "Не удалось найти релевантные сообщения по вашему запросу."
            
        # Форматируем результаты для удобного чтения
        formatted_results = await message_searcher.format_search_results(ranked_results)
        logging.info(f"Получены отранжированные результаты: {len(ranked_results)} сообщений")
        
        # Создаем ответ с использованием LLM
        # Добавляем API ключ напрямую
        api_key = "AIzaSyAR3IRvu_WIrMPfbnyL5wyhcgXBW2UCGcU" 
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0.7,
            google_api_key=api_key
        )
        
        # Подготавливаем сообщения из результатов
        messages_data = []
        for result in ranked_results:
            message_content = result.get("Сообщение", "")
            author = result.get("Автор", "Неизвестный")
            
            # Получаем ссылку на сообщение вместо ссылки на канал
            message_id = result.get("ID")
            message_link = None
            
            if message_id:
                # Используем специальную функцию для получения ссылки на сообщение
                message_link = await db_instance.get_message_link(message_id)
            
            # Если не удалось получить ссылку на сообщение, пробуем получить из результата
            if not message_link and result.get("message_link"):
                message_link = result.get("message_link")
            
            # Добавляем информацию к тексту сообщения
            message_entry = f"Сообщение от {author}: {message_content}"
            if message_link:
                message_entry += f"\nСсылка на сообщение: {message_link}"
            
            messages_data.append(message_entry)
        
        context = "\n\n".join(messages_data)
        
        # Используем переданный текст сообщения для user_query и
        # найденный результат для context - избегаем дублирования имен переменных
        user_query = message_text  # сохраняем исходный запрос пользователя
        
        logging.info(f"НАЙДЕННАЯ ПЕРЕМЕННАЯ user_query: {user_query}")
        logging.info(f"НАЙДЕННАЯ ПЕРЕМЕННАЯ context: {context[:200]}...")
        
        prompt = f"""
        Запрос пользователя: {user_query}

Задача:
0. Не используй в ответе разметку, она не работает
1. Игнорировать документы, не относящиеся к запросу, даже не упоминай их существование
2. 1 абзац - Составить краткое содержание новостей(30 слов), основанное ТОЛЬКО на релевантных документах
3. 2 и далее абзацы - Написать по порядку (1. 2. 3. и т.д.) через табуляцю документы ПОЛНОСТЬЮ, НЕ СОКРЩЗАЯ (если длинное сократи до 100 слов если длинные) 
4. После каждого сообщения предоставь ссылку к нему в формате https://t.me/htech_plus/1538

Найденные документы:
{context}
"""
        
        ai_response = await llm.ainvoke(prompt)
        return ai_response.content
        
    except Exception as e:
        logging.error(f"Ошибка при выполнении семантического поиска: {str(e)}", exc_info=True)
        return f"Произошла ошибка при выполнении семантического поиска: {str(e)}"

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleGeminiProcessor:
    """Простой обработчик сообщений с Gemini API без дублирования функциональности aiogram."""

    def __init__(self, bot: Bot = None):
        # Получаем ключ API из переменной окружения
        api_key = "AIzaSyAR3IRvu_WIrMPfbnyL5wyhcgXBW2UCGcU"
        genai.configure(api_key=api_key)
        
        # Создаем исходную модель и сохраняем ее
        self.original_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=1, 
            google_api_key=api_key
        )
        
        # Создаем модель с инструментами для использования в обычных сценариях
        self.llm = self.original_llm.bind_tools([channel_action_tool, semantic_search_tool])
        
        self.bot = bot
        self.llm_prompt_template = """Ты бот пока ты безрукий поэтому просто ответь"""
        self.llm_prompt = PromptTemplate(template=self.llm_prompt_template, input_variables=["text"])
        self.function_complete = False

        logger.info("SimpleGeminiProcessor инициализирован с Gemini Flash 1.5")
    
    async def process_message(self, user_id: int, message: Message, chat_history, state: FSMContext = None) -> str:
        """
        Обработать сообщение пользователя через Gemini API.
        """
        logger.info(f">>> НАЧАЛО process_message для user_id={user_id}")
        try:
            # Логируем входные данные
            logger.info(f"Текст сообщения: '{message.text[:100]}...'")
            logger.info(f"Наличие chat_history: {chat_history is not None}")
            
            # Проверка на семантический запрос
            search_keywords = ["новости", "найди", "поиск", "информация о", "что известно"]
            is_search_query = any(keyword in message.text.lower() for keyword in search_keywords)
            
            if is_search_query:
                logger.info(f"Обнаружен поисковый запрос: {message.text}")
                # Выполняем семантический поиск
                result = await semantic_search_tool.ainvoke({
                    "query": message.text,
                    "message_text": message.text
                })
                if result and len(result) > 0:
                    logger.info(f"Получен результат поиска длиной {len(result)} символов")
                    return result, False
                else:
                    logger.warning("Получен пустой результат поиска")
                    return "К сожалению, не удалось найти информацию по вашему запросу.", False
            
            # Передаем текст сообщения в LLM с инструментами
            logger.info("Отправляем запрос в LLM с инструментами...")
            prompt = f"Определи, что хочет сделать пользователь: показать список каналов (action=show_chats_list), добавить канал (action=add_channel_start), удалить канал (action=delete_channel_start). Сообщение пользователя: '{message.text}'"
            logger.info(f"Промпт: {prompt[:100]}...")
            
            ai_msg = await self.llm.ainvoke(
                prompt,
                {"message_text": message.text}
            )
            
            # Логируем результат запроса
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
                
                elif tool_name == "semantic_search_tool":
                    logger.info("Обрабатываем семантический поиск...")
                    # Извлекаем значение query из args
                    query = ai_msg.tool_calls[0]['args']['query']
                    logger.info(f"Извлечен запрос: {query[:100]}...")
                    
                    # Выполняем запрос
                    result = await semantic_search_tool.ainvoke({
                        "query": query,
                        "message_text": message.text
                    })
                    logger.info(f"Получен результат запроса длиной {len(result)} символов")
                    
                    # Проверка на пустой результат
                    if not result or len(result.strip()) == 0:
                        logger.warning("Получен пустой результат семантического поиска")
                        return "К сожалению, не удалось найти информацию по вашему запросу.", False
                    
                    # Возвращаем результат и флаг function_complete=False
                    return result, False
            
            # Если нет вызовов инструментов, обрабатываем как обычное сообщение
            logger.info("Обрабатываем как обычное сообщение...")
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