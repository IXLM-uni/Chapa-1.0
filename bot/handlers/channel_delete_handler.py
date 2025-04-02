# bot/handlers/channel_delete_handler.py

from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery, ReplyKeyboardRemove
from aiogram.fsm.context import FSMContext
from states import ChannelDeleteStates # <--- Импорт ChannelDeleteStates
from database.requests import DatabaseRequests
from bot.keyboards import get_chats_keyboard # <--- Импорт клавиатуры выбора чатов
from services.channel_service import ChannelService
router = Router()

@router.message(Command("delete_channel"))
async def delete_channel_start(message: Message,state: FSMContext, db: DatabaseRequests):
    """Начало процесса удаления канала."""
    service = ChannelService(db=db)
    chats = await service.get_user_channels(message.from_user.id)

    if not chats:
        await message.answer("У вас нет добавленных каналов для удаления.")
        return

    await message.answer(
        "Выберите канал, который вы хотите удалить:",
        reply_markup=await get_chats_keyboard(chats)
    )

@router.callback_query(F.data.startswith("chat_"))
async def process_delete_channel_selection(callback: CallbackQuery, db: DatabaseRequests):
    """Обработка выбора канала для удаления."""
    channel_id_to_delete = callback.data.split("_")[1]

    deleted = await db.delete_user_channel(channel_id_to_delete, callback.from_user.id)

    if deleted:
        await callback.message.answer(
            f"Канал с ID {channel_id_to_delete} успешно удален.",
            reply_markup=ReplyKeyboardRemove()
        )
    else:
        await callback.message.answer(
            "Не удалось удалить канал. Попробуйте позже.",
            reply_markup=ReplyKeyboardRemove()
        )