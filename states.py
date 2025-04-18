# states.py
from aiogram.fsm.state import State, StatesGroup

class UserStates(StatesGroup): # <--- UserStates - ОСТАВЛЯЕМ КАК ЕСТЬ (пока)
    WAITING_FOR_CHAT_SELECTION = State()
    WAITING_FOR_QUESTION = State()
    WAITING_FOR_TIME_PERIOD = State()
    WAITING_FOR_CUSTOM_PERIOD_START = State()
    WAITING_FOR_CUSTOM_PERIOD_END = State()
    WAITING_FOR_KEYWORD = State()

class ChannelStates(StatesGroup): # <--- ChannelStates - УПРОЩАЕМ!
    WAITING_FOR_CHANNEL_ID = State() # <--- Оставляем WAITING_FOR_CHANNEL_ID, но можно переименовать
    WAITING_FOR_FORWARD = State() # <--- Раскомментируем это состояние

class ChannelParserStates(StatesGroup): # <--- ChannelParserStates - ОСТАВЛЯЕМ КАК ЕСТЬ (задел на будущее)
    WAITING_FOR_FORWARD_FOR_PARSE = State()

class ChannelDeleteStates(StatesGroup): # <--- ChannelDeleteStates - УПРОЩАЕМ!
    WAITING_FOR_CHANNEL_SELECTION = State() # <--- Переименовываем и УПРОЩАЕМ, теперь только WAITING_FOR_CHANNEL_SELECTION
    # WAITING_FOR_CHANNEL_TO_DELETE = State() # <--- УДАЛЯЕМ WAITING_FOR_CHANNEL_TO_DELETE - ИЗБЫТОЧНО
    # WAITING_FOR_DELETE_CONFIRMATION = State() # <--- УДАЛЯЕМ WAITING_FOR_DELETE_CONFIRMATION - ИЗБЫТОЧНО