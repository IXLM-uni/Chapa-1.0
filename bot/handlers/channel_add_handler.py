# bot/handlers/channel_add_handler.py

from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.fsm.context import FSMContext
from states import ChannelStates  # Импортируем ChannelStates
from database.requests import DatabaseRequests # <--- Импорт DatabaseRequests
from bot.keyboards import get_start_keyboard, get_chats_keyboard # <--- Импорт клавиатур

router = Router()

@router.message(F.forward_from_chat)
@router.message(Command("add_channel"))
async def add_channel_start(message: Message, state: FSMContext, db: DatabaseRequests):
    """Начало процесса добавления канала."""
    if message.forward_from_chat:
        # Если это пересланное сообщение из канала, обрабатываем его сразу
        channel_id = int(message.forward_from_chat.id[4:])
        channel_name = message.forward_from_chat.title
        user_telegram_id = message.from_user.id
        
        # Добавляем канал
        channel_added = await db.add_channel(channel_id=channel_id, telegram_id=user_telegram_id, name=channel_name)
        
        if channel_added:
            await message.answer(f"✅ Канал '{channel_name}' успешно добавлен!")
        else:
            await message.answer("❌ Не удалось добавить канал. Попробуйте позже.")
        return
    
    # Если это команда add_channel
    await message.answer(
        "Пожалуйста, **перешлите сообщение из канала**, который вы хотите добавить.",
        parse_mode="Markdown"
    )
    await state.set_state(ChannelStates.WAITING_FOR_CHANNEL_ID)
