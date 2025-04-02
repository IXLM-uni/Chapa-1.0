# Файл для хранения глобальных переменных
import logging

# Глобальные переменные
global_vector_search = None
global_embedding_model = None
async_session = None

# Функция для инициализации
def init_globals(vs=None, em=None, session=None):
    global global_vector_search, global_embedding_model, async_session
    
    if vs is not None:
        global_vector_search = vs
        logging.info("Глобальная переменная vector_search инициализирована")
        
    if em is not None:
        global_embedding_model = em
        logging.info("Глобальная переменная embedding_model инициализирована")
        
    if session is not None:
        async_session = session
        logging.info("Глобальная переменная async_session инициализирована") 