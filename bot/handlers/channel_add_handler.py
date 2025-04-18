# bot/handlers/channel_add_handler.py

from aiogram import Router, F, Bot
from aiogram.filters import Command
from aiogram.filters.state import StateFilter
from aiogram.types import Message
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from database.requests import DatabaseRequests
from bot.keyboards import get_start_keyboard
from states import ChannelStates # Возвращаем прямой импорт
import logging
from telethon import TelegramClient
from aiogram.utils.markdown import hbold, hitalic, hlink, hcode
import asyncio # Убираем, т.к. лаконичный вариант не требует asyncio
from bot.parser import TelegramParser # <<< Импортируем TelegramParser

logger = logging.getLogger(__name__)
router = Router()

# Вспомогательная функция для фонового парсинга и уведомления
async def parse_and_notify(parser: TelegramParser, channel_id: int, limit: int, bot: Bot, chat_id: int, initial_message_id: int, channel_title: str, channel_link: str):
    """Запускает парсинг, сохраняет сообщения и уведомляет пользователя о завершении."""
    saved_count = 0 # Счетчик сохраненных сообщений
    try:
        logger.info(f"Starting background parsing for channel {channel_id} ('{channel_title}') in chat {chat_id}.")
        # Запускаем парсинг и получаем список сообщений
        messages = await parser._parse_single_channel(channel_id=channel_id, limit=limit) # <<< Получаем сообщения
        logger.info(f"Parsing collected {len(messages)} messages for channel {channel_id} ('{channel_title}').")

        # Сохраняем собранные сообщения в БД, если они есть
        if messages:
            saved_count = await parser.save_to_db(messages) # <<< Сохраняем сообщения
            logger.info(f"Saved {saved_count} messages from channel {channel_id} to DB.")
        else:
            logger.info(f"No new messages to save for channel {channel_id}.")

        logger.info(f"Background parsing and saving finished for channel {channel_id} ('{channel_title}') in chat {chat_id}.")

        # Удаляем начальное сообщение
        try:
            await bot.delete_message(chat_id=chat_id, message_id=initial_message_id)
            logger.debug(f"Deleted initial parsing message {initial_message_id} in chat {chat_id}.")
        except Exception as delete_err:
            logger.warning(f"Failed to delete initial parsing message {initial_message_id} in chat {chat_id}: {delete_err}")

        # Отправляем сообщение о завершении
        # Обновляем текст, чтобы упомянуть количество собранных сообщений
        completion_text = f"✅ Первичный сбор сообщений для канала '{channel_title}' завершен. Собрано: {len(messages)} сообщений."
        await bot.send_message(chat_id=chat_id, text=completion_text)
        logger.info(f"Sent completion notification for channel {channel_id} to chat {chat_id}.")

    except Exception as parse_err:
        logger.error(f"Error during background parsing/saving for channel {channel_id} ('{channel_title}') in chat {chat_id}: {parse_err}", exc_info=True)
        # Попытаемся удалить начальное сообщение даже при ошибке
        try:
            await bot.delete_message(chat_id=chat_id, message_id=initial_message_id)
        except Exception:
            pass
        # Уведомляем пользователя об ошибке
        await bot.send_message(chat_id=chat_id, text=f"❗️ Ошибка во время сбора сообщений для канала '{channel_title}' ({channel_link}). Подробности в логах.")

# Добавляем Command("add_channel") и логику проверки
@router.message(Command("add_channel")) # <<< Добавляем реакцию на команду
@router.message(F.forward_from_chat)   # <<< Оставляем реакцию на пересылку
async def add_channel_start(message: Message, state: FSMContext, client: 'TelegramClient', db: 'DatabaseRequests', parser: 'TelegramParser', bot: Bot):
    """
    Обрабатывает команду /add_channel или пересланное сообщение для добавления канала.
    Если вызвана командой, просит переслать сообщение.
    Если получено пересланное сообщение, добавляет канал и уведомляет о статусе парсинга.
    """
    # --- Проверяем, как вызван обработчик ---
    if message.forward_from_chat is None:
        # Вызвано командой /add_channel
        logger.info(f"User {message.from_user.id} initiated add_channel via command.")
        await message.answer(
            "Пожалуйста, перешлите сюда любое сообщение из канала, который вы хотите добавить. "
            "Убедитесь, что у бота есть доступ к этому каналу (если он приватный)."
        )
        return # Завершаем обработку, ждем пересылки
    # --- Конец проверки ---

    # Если код дошел сюда, значит, это пересланное сообщение
    # --- Лаконичная обработка медиа-групп ---
    if message.media_group_id and message.caption is None:
        logger.debug(f"Ignoring media group message {message.message_id} (media_group_id={message.media_group_id}) as it's not the first one (no caption).")
        return
    # --- Конец обработки медиа-групп ---

    forwarded_chat = message.forward_from_chat # Теперь мы знаем, что это не None
    user_id = message.from_user.id

    # Эта проверка больше не нужна, т.к. мы уже проверили message.forward_from_chat
    # if not forwarded_chat:
    #     logger.warning(f"Message {message.message_id} has F.forward_from_chat but message.forward_from_chat is None.")
    #     await message.answer("Произошла странная ошибка. Попробуйте переслать другое сообщение.")
    #     return

    log_media_group_id = f" (media_group_id={message.media_group_id})" if message.media_group_id else ""
    logger.info(f"Processing forwarded message{log_media_group_id} from channel ID: {forwarded_chat.id}, Title: '{forwarded_chat.title}' for user {user_id}.")

    # --- Извлечение ссылки ---
    channel_link = None
    if forwarded_chat.username:
        # Формируем ссылку вида https://t.me/username
        channel_link = f"https://t.me/{forwarded_chat.username}"
        logger.info(f"Extracted channel link: {channel_link}")
    else:
        # Если у чата нет username, мы не можем сформировать публичную ссылку.
        # Это может быть приватный канал/группа.
        logger.warning(f"Channel ID {forwarded_chat.id} ('{forwarded_chat.title}') does not have a public username.")
        await message.answer(
            f"Не удалось получить публичную ссылку вида @username для канала '{forwarded_chat.title}'. "
            "Возможно, это приватный канал или у него нет юзернейма. "
            "Пожалуйста, перешлите сообщение из публичного канала с юзернеймом."
        )
        return
    # --- Конец извлечения ссылки ---

    # Вызываем db.add_channel и распаковываем результат
    add_status, was_new, added_channel_id = await db.add_channel(
        channel_id=channel_link,  # Передаем извлеченную ссылку
        client=client,
        telegram_id=user_id
    )

    # --- Обработка результата ---
    if add_status is True:
        # Сначала отправляем базовое сообщение об успехе
        await message.answer(f"Канал '{forwarded_chat.title}' ({channel_link}) успешно добавлен в ваш список.")

        # Если канал новый, запускаем парсинг и уведомляем отдельно
        if was_new and added_channel_id is not None:
            # Отправляем сообщение о начале парсинга и сохраняем его
            initial_parse_msg = await message.answer(f"⏳ Начинаю первичный сбор сообщений для '{forwarded_chat.title}'...")
            logger.info(f"Channel {added_channel_id} ('{forwarded_chat.title}') is new. Starting initial parse task for chat {message.chat.id}.")

            # Запускаем парсинг и уведомление в фоновом режиме
            asyncio.create_task(parse_and_notify(
                parser=parser,
                channel_id=added_channel_id,
                limit=10000, # Оставляем ваш лимит
                bot=bot,
                chat_id=message.chat.id,
                initial_message_id=initial_parse_msg.message_id,
                channel_title=forwarded_chat.title,
                channel_link=channel_link
            ))
        # Сообщение об успешном добавлении уже отправлено выше

    elif add_status is None:
        await message.answer(f"Вы уже подписаны на канал '{forwarded_chat.title}' ({channel_link}).")
    else: # add_status is False
        await message.answer(f"Не удалось обработать канал '{forwarded_chat.title}'. Возможно, произошла ошибка или бот не имеет доступа к этому каналу. Попробуйте позже или обратитесь к администратору.")
    # --- Конец обработки результата ---

# Обработчик для случая, если пользователь прислал что-то другое вместо пересылки
# @router.message(ChannelStates.WAITING_FOR_FORWARD)
# async def handle_wrong_input_while_waiting_forward(message: Message, state: FSMContext):
#     """Обрабатывает некорректный ввод, когда ожидается пересланное сообщение."""
#     await message.answer(
#         "Пожалуйста, перешлите сообщение из канала, который вы хотите добавить, или используйте /cancel для отмены."
#     )
