from aiogram.types import (
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    ReplyKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardRemove
)
from aiogram.utils.keyboard import InlineKeyboardBuilder

# Можно добавить константы для часто используемых текстов
STOP_TEXT = "Стоп"
SHOW_CHATS_TEXT = "Показать список чатов"
ADD_CHANNEL_TEXT = "Добавить канал"

def get_start_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=SHOW_CHATS_TEXT)],
            [KeyboardButton(text=ADD_CHANNEL_TEXT)]
        ],
        resize_keyboard=True
    )

def get_chat_actions_keyboard():
    buttons = [
        [
            ("Задать вопрос", "ask_question"),
            ("Свод новостей", "news_summary")
        ],
        [("Поиск по ключевому слову", "keyword_search")],
        [("Стоп", "stop")]
    ]
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=text, callback_data=data) for text, data in row]
            for row in buttons
        ]
    )

def get_time_period_keyboard():
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Неделя", callback_data="period_week"),
                InlineKeyboardButton(text="Месяц", callback_data="period_month")
            ],
            [
                InlineKeyboardButton(text="Год", callback_data="period_year"),
                InlineKeyboardButton(text="Указать промежуток", callback_data="period_custom")
            ]
        ]
    )

async def get_chats_keyboard(chats):
    builder = InlineKeyboardBuilder()
    for chat in chats:
        builder.add(InlineKeyboardButton(
            text=chat['title'],
            callback_data=f"chat_{chat['id']}"
        ))
    builder.adjust(1)
    return builder.as_markup()

def get_stop_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="Стоп")]],
        resize_keyboard=True,
        one_time_keyboard=True
    )