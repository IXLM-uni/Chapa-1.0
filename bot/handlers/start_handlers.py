# bot/handlers/start_handlers.py
from aiogram import Router, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from aiogram.fsm.context import FSMContext
from database.requests import DatabaseRequests
from bot.keyboards import get_start_keyboard, get_chats_keyboard
from states import UserStates, ChannelStates
from services.channel_service import ChannelService

router = Router()

@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext, db: DatabaseRequests):
    """Обработчик команды /start и текстовой команды 'Главное меню'."""
    user_exists = await db.check_user_exists(message.from_user.id)
    if not user_exists:
        await db.add_user(message.from_user.id)

    await message.answer(
        "Добро пожаловать в **Продвинутый Новостной Бот**! 👋\n\n" # Изменено приветствие
        "Я помогу вам получать сводки новостей из Telegram-каналов.\n\n"
        "Выберите действие:", # Более явное приглашение к действию
        reply_markup=get_start_keyboard()
    , parse_mode="Markdown") # Включаем Markdown

    await state.clear()

@router.message(F.text.lower().in_({"главное меню", "меню"}))
@router.message(Command("menu_kb"))
async def show_main_menu_kb(message: Message, state: FSMContext, db: DatabaseRequests):
    """Обработчик для показа главного меню (текстовые команды и кнопка /start)."""
    await cmd_start(message, state, db)

@router.message(F.text.lower().in_({"показать список чатов", "список чатов", "мои чаты", "чаты"}))
@router.message(Command("show_chats"))
async def show_chats_list(message: Message, state: FSMContext, db: DatabaseRequests):
    """Обработчик кнопки 'Показать список чатов' и текстовых команд."""
    service = ChannelService(db=db)
    
    # Получаем список каналов пользователя
    chats = await service.get_user_channels(message.from_user.id)
    
    if not chats:
        await message.answer("У вас пока нет добавленных каналов.")
        return
    
    # Создаем и отправляем клавиатуру с каналами
    keyboard = await get_chats_keyboard(chats)
    await message.answer("Выберите канал из списка:", reply_markup=keyboard)
    
    await state.set_state(UserStates.WAITING_FOR_CHAT_SELECTION)
