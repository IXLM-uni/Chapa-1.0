from aiogram import Router, F
from aiogram.types import Message, CallbackQuery
from aiogram.fsm.context import FSMContext
from database.requests import DatabaseRequests  # Убедитесь, что путь к requests верный
from bot.keyboards import get_time_period_keyboard, get_chat_actions_keyboard  # Убедитесь, что путь к keyboards верный
from states import UserStates  # Убедитесь, что путь к states верный
from datetime import datetime, timedelta

router = Router()

@router.message(UserStates.WAITING_FOR_QUESTION, F.text == "Свод новостей")
async def show_news_summary(message: Message, state: FSMContext):
    """Обработчик кнопки 'Свод новостей'."""
    await message.answer(
        "Выберите период для сводки новостей:",
        reply_markup=get_time_period_keyboard()
    )
    await state.set_state(UserStates.WAITING_FOR_TIME_PERIOD)

@router.callback_query(UserStates.WAITING_FOR_TIME_PERIOD, F.data.startswith("period_"))
async def process_time_period(callback: CallbackQuery, state: FSMContext):
    """Обработчик выбора периода времени для сводки."""
    period = callback.data.split("_")[1]
    if period == "custom":
        await callback.message.answer(
            "Введите начальную дату в формате YYYY-MM-DD:"
        )
        await state.set_state(UserStates.WAITING_FOR_CUSTOM_PERIOD_START)
    else:
        chat_id = (await state.get_data())["current_chat"]
        # Создаем экземпляр DatabaseRequests для работы с БД
        db = DatabaseRequests()

        if period == "week":
            start_date = datetime.now() - timedelta(days=7)
        elif period == "month":
            start_date = datetime.now() - timedelta(days=30)
        elif period == "year":
            start_date = datetime.now() - timedelta(days=365)

        messages = await db.get_messages_by_period(chat_id, start_date, datetime.now())

        await callback.message.answer(
            "Сводка новостей готова. Выберите другое действие:",
            reply_markup=get_chat_actions_keyboard()
        )
        await state.clear()