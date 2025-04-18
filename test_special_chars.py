import spacy
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_special_chars() -> List[Tuple[str, List[Tuple[str, str]]]]:
    """
    Тестирует обработку специальных символов в NER
    Возвращает список тестовых примеров с их сущностями
    """
    # Загружаем модель
    nlp = spacy.load("ru_core_news_md")
    
    # Тестовые примеры с разными специальными символами
    test_cases = [
        # Эмодзи и специальные символы
        "⚡️Создай отдельный файл для проверки",
        "📱iPhone 13 Pro Max",
        "🎮PlayStation 5",
        "💻MacBook Pro",
        
        # Специальные символы в названиях
        "C++ Programming Language",
        "Python 3.9",
        "JavaScript (JS)",
        "HTML & CSS",
        
        # Специальные символы в именах
        "Mr. & Mrs. Smith",
        "Dr. Jekyll",
        "O'Connor",
        "Jean-Pierre",
        
        # Специальные символы в датах
        "01/01/2023",
        "2023-12-31",
        "31.12.2023",
        
        # Специальные символы в адресах
        "123 Main St., Apt. 4B",
        "P.O. Box 12345",
        "Suite #100",
        
        # Специальные символы в организациях
        "Apple Inc.",
        "Microsoft Corp.",
        "Google LLC",
        
        # Специальные символы в продуктах
        "iPhone®",
        "Windows™",
        "MacOS®",
        
        # Специальные символы в валютах
        "$100.00",
        "€50.00",
        "£75.00",
        
        # Специальные символы в событиях
        "World Cup 2022™",
        "Olympic Games®",
        "Super Bowl®",
        "gemini-2.5 выпустила новую игру"
    ]
    
    results = []
    for text in test_cases:
        doc = nlp(text)
        entities = []
        
        # Собираем сущности
        for ent in doc.ents:
            entities.append((ent.text, ent.label_))
        
        results.append((text, entities))
        
        # Логируем результаты
        logger.info(f"\nТекст: {text}")
        logger.info("Найденные сущности:")
        for entity in entities:
            logger.info(f"  - {entity[0]} ({entity[1]})")
    
    return results

def analyze_results(results: List[Tuple[str, List[Tuple[str, str]]]]) -> None:
    """
    Анализирует результаты тестирования
    """
    logger.info("\n=== Анализ результатов ===")
    
    # Подсчет статистики
    total_cases = len(results)
    cases_with_entities = sum(1 for _, entities in results if entities)
    total_entities = sum(len(entities) for _, entities in results)
    
    logger.info(f"Всего тестовых случаев: {total_cases}")
    logger.info(f"Случаев с найденными сущностями: {cases_with_entities}")
    logger.info(f"Всего найдено сущностей: {total_entities}")
    
    # Анализ типов сущностей
    entity_types = {}
    for _, entities in results:
        for _, entity_type in entities:
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
    
    logger.info("\nРаспределение типов сущностей:")
    for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {entity_type}: {count}")
    
    # Анализ проблемных случаев
    logger.info("\nПроблемные случаи (без найденных сущностей):")
    for text, entities in results:
        if not entities:
            logger.info(f"  - {text}")

if __name__ == "__main__":
    results = test_special_chars()
    analyze_results(results) 
