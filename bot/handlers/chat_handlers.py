# bot/handlers/chat_handlers.py

from aiogram import Router, F
from aiogram.types import Message, CallbackQuery
from aiogram.fsm.context import FSMContext
from database.requests import DatabaseRequests
from bot.keyboards import get_chat_actions_keyboard  # Убедитесь, что путь к keyboards верный
from states import UserStates  # Убедитесь, что путь к states верный
from datetime import datetime
import uuid
from sqlalchemy import select
import traceback
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
import logging
from bot.handlers.channel_add_handler import add_channel_start
from bot.handlers.channel_delete_handler import delete_channel_start
from bot.handlers.start_handlers import show_chats_list

logger = logging.getLogger(__name__)

router = Router()

@router.callback_query(F.data.startswith("chat_"))
async def select_chat(callback: CallbackQuery, state: FSMContext):
    """Обработчик выбора чата из inline-клавиатуры."""
    chat_id = callback.data.split("_")[1]
    await callback.message.answer(
        "Выберите действие:",
        reply_markup=get_chat_actions_keyboard()
    )
    await state.set_state(UserStates.WAITING_FOR_QUESTION)
    await state.update_data(current_chat=chat_id)

# *** DEFAULT HANDLER - ОБРАБАТЫВАЕТ ЛЮБЫЕ ТЕКСТОВЫЕ СООБЩЕНИЯ (если не обработаны другими handler-ами) ***
@router.message(F.text)
async def process_question(message: Message, state: FSMContext, llm_processor, db: DatabaseRequests):
    """Обработчик текстовых сообщений и вопросов к боту."""
    try:
        user_id = message.from_user.id
        logger.info(f"Получено сообщение от пользователя {user_id}: {message.text[:100]}...")
        
        async with db.session() as session:
            session_id = await db.get_or_create_session_id(user_id)
            logger.info(f"Получен session_id: {session_id}")
            
            user_chat_history = SQLChatMessageHistory(
                session_id=session_id,
                connection=session.bind,
                table_name="chat_message_history"
            )
            
            logger.info(f"Вызов process_message с параметрами: user_id={user_id}, message.text={message.text[:50]}...")
            prompt = f"Определи, что хочет сделать пользователь: показать список каналов (action=show_chats_list), добавить канал (action=add_channel_start), удалить канал (action=delete_channel_start), или ничего из перечисленного (action=none). Сообщение пользователя: '{message.text}'"
            response, function_complete = await llm_processor.process_message(
                user_id=user_id,
                message=message,
                chat_history=user_chat_history,
                state=state
            )
            
            logger.info(f"Получен ответ от процессора: {response}, function_complete={function_complete}")
            
            # Проверка длины ответа перед отправкой
            if len(response) > 4000:
                logger.warning(f"Ответ слишком длинный ({len(response)} символов), обрезаем до 4000")
                response = response[:4000] + "...\n(сообщение было обрезано из-за ограничений Telegram)"
            
            logger.info(f"Отправляем сообщение длиной {len(response)} символов")
            
            # Действуем в зависимости от флага
            if function_complete:
                # Вызов функции по имени из строки response
                function_map = {
                    "show_chats_list": show_chats_list,
                    "add_channel_start": add_channel_start,
                    "delete_channel_start": delete_channel_start,
                    # SQL-запросы обрабатываются напрямую в process_message
                }
                
                # Проверяем, что функция существует
                if response in function_map:
                    # Вызываем функцию с передачей всех необходимых параметров
                    result = await function_map[response](message, state, db)
                    
                    # Если функция возвращает текст, отправляем его
                    if isinstance(result, str):
                        await message.answer(result)
                else:
                    await message.answer(f"Неизвестная функция: {response}")
            else:
                await message.answer(response)
            
    except Exception as e:
        logger.exception(f"Ошибка в process_question: {str(e)}")
        await message.answer(f"Произошла ошибка: {str(e)}")

# *** УДАЛЯЕМ ЛИШНИЕ HANDLER-Ы для MVP "РЕГИСТРАТУРЫ" - ask_question, show_news_summary, process_time_period и т.д. ***
# *** Оставляем только select_chat и process_question (DEFAULT) ***