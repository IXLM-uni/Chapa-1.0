import spacy
import logging
from typing import List, Tuple, Dict
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

class SimpleSpacyExtractor:
    """
    Упрощенный класс для извлечения сущностей из текста и форматирования под spaCy
    """
    def __init__(self, db_url: str, google_api_key: str):
        # Загрузка модели spaCy
        try:
            self.nlp = spacy.load("ru_core_news_md")
            logging.info("Модель spaCy ru_core_news_md загружена")
        except OSError:
            logging.error("Ошибка загрузки модели ru_core_news_md")
            raise

        # Подключение к БД
        self.engine = create_engine(db_url)
        
        # Настройка Google API
        self.google_api_key = google_api_key
        
    def fetch_messages(self, channel_id: int, limit: int = 500) -> List[str]:
        """Получение сообщений из БД"""
        with Session(self.engine) as session:
            query = select(ChannelPosts.message).where(
                ChannelPosts.peer_id == channel_id,
                ChannelPosts.message.isnot(None)
            ).limit(limit)
            
            result = session.execute(query)
            messages = [row[0] for row in result if row[0] and row[0].strip()]
            
        return messages

    def process_batch(self, messages: List[str]) -> List[Tuple[str, Dict]]:
        """
        Обработка пакета сообщений и форматирование под spaCy
        """
        training_data = []
        
        # Объединяем сообщения в один текст
        combined_text = " ".join(messages)
        
        # Обработка текста через spaCy
        doc = self.nlp(combined_text)
        
        # Извлечение сущностей и форматирование
        for ent in doc.ents:
            # Определяем тип сущности
            entity_type = self.map_entity_type(ent.label_)
            if entity_type:
                # Форматируем под формат TEST_DATA
                training_example = (
                    ent.sent.text.strip(),  # Берем предложение целиком
                    {
                        'entities': [(ent.start_char - ent.sent.start_char, 
                                    ent.end_char - ent.sent.start_char, 
                                    entity_type)]
                    }
                )
                training_data.append(training_example)
        
        return training_data

    def map_entity_type(self, spacy_label: str) -> str:
        """
        Маппинг меток spaCy в нужный формат
        """
        mapping = {
            'ORG': 'ORG',
            'PERSON': 'PER',
            'LOC': 'LOC',
            'PRODUCT': 'TECH',
            # Добавьте другие маппинги по необходимости
        }
        return mapping.get(spacy_label)

    def generate_training_data(self, channel_id: int) -> List[Tuple[str, Dict]]:
        """
        Основной метод для генерации обучающих данных
        """
        # Получаем сообщения из БД
        messages = self.fetch_messages(channel_id)
        if not messages:
            logging.warning(f"Нет сообщений для канала {channel_id}")
            return []

        # Обрабатываем все сообщения разом
        training_data = self.process_batch(messages)
        
        logging.info(f"Обработано {len(messages)} сообщений, создано {len(training_data)} примеров")
        return training_data

# Пример использования
if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    # Параметры подключения
    DB_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Создаем экстрактор
    extractor = SimpleSpacyExtractor(DB_URL, GOOGLE_API_KEY)
    
    # Генерируем данные для канала
    channel_id = 1466120158
    training_data = extractor.generate_training_data(channel_id)
    
    # Выводим примеры
    for text, annotations in training_data[:5]:
        print(f"\nТекст: {text}")
        print(f"Разметка: {annotations}")