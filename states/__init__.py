from aiogram.fsm.state import State
from aiogram.fsm.state import StatesGroup

# Состояния для основного процесса взаимодействия с пользователем
class UserStates(StatesGroup):
    WAITING_FOR_CHAT_SELECTION = State()  # Ожидание выбора чата
    WAITING_FOR_QUESTION = State()        # Ожидание вопроса от пользователя
    ANSWERING_QUESTION = State()          # Процесс ответа на вопрос

# Состояния для процесса добавления канала
class ChannelStates(StatesGroup):
    WAITING_FOR_CHANNEL_ID = State()      # Ожидание ID/форварда из канала
    WAITING_FOR_CHANNEL_NAME = State()    # Ожидание названия канала
    WAITING_FOR_CONFIRMATION = State()    # Ожидание подтверждения

# Состояния для процесса удаления канала
class ChannelDeleteStates(StatesGroup):
    WAITING_FOR_CHANNEL_SELECTION = State()  # Ожидание выбора канала
    CONFIRMING_DELETION = State()            # Подтверждение удаления