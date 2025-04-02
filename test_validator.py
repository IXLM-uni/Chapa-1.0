import asyncio
import logging
from Learning_data import DataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_validator():
    # Создаем экземпляр DataProcessor
    processor = DataProcessor(messages_limit=1, sentences_per_chunk=1, num_llms=1)
    
    # Тестовые данные с разными сценариями
    test_cases = [
        # Правильный формат
        """('John works at Google', {"entities": [(0, 4, "PER"), (12, 18, "ORG")]})""",
        
        # Неправильный формат - лишние пробелы
        """('John works at Google',  {"entities": [(0, 4, "PER"), (12, 18, "ORG")]})""",
        
        # Неправильный формат - неправильные кавычки
        """("John works at Google", {"entities": [(0, 4, "PER"), (12, 18, "ORG")]})""",
        
        # Неправильный формат - неправильные индексы
        """('John works at Google', {"entities": [(0, 20, "PER"), (12, 18, "ORG")]})""",
        
        # Неправильный формат - неправильная метка
        """('John works at Google', {"entities": [(0, 4, "WRONG_LABEL"), (12, 18, "ORG")]})""",
        
        # Неправильный формат - пересекающиеся индексы
        """('John works at Google', {"entities": [(0, 10, "PER"), (5, 18, "ORG")]})""",
        
        # Неправильный формат - отсутствие запятой между элементами
        """('John works at Google', {"entities": [(0, 4, "PER"), (12, 18, "ORG")]})('Jane works at Meta', {"entities": [(0, 4, "PER"), (12, 16, "ORG")]})""",
        
        # Неправильный формат - лишняя разметка и JSON
        """[
            {
                "text": "John works at Google",
                "entities": [
                    {"start": 0, "end": 4, "label": "PER"},
                    {"start": 12, "end": 18, "label": "ORG"}
                ]
            },
            ('Jane works at Meta', {"entities": [(0, 4, "PER"), (12, 16, "ORG")]})
        ]""",
        
        # Неправильный формат - смешанные форматы
        """('John works at Google', {"entities": [(0, 4, "PER"), (12, 18, "ORG")]})
        <div class="entity">('Jane works at Meta', {"entities": [(0, 4, "PER"), (12, 16, "ORG")]})</div>
        {"text": "Bob works at Apple", "entities": [{"start": 0, "end": 3, "label": "PER"}]}"""
    ]
    
    logger.info("Начало тестирования валидатора...")
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\nТест {i}:")
        logger.info(f"Входные данные: {test_case}")
        
        try:
            validated = await processor.validate_chunk(test_case)
            logger.info(f"Результат валидации: {validated}")
            
            if validated != test_case:
                logger.info("Валидатор внес изменения")
            else:
                logger.info("Валидатор не внес изменений")
                
        except Exception as e:
            logger.error(f"Ошибка при валидации: {e}")
        
        await asyncio.sleep(1)  # Пауза между тестами
    
    logger.info("\nТестирование завершено")
    await processor.engine.dispose()

if __name__ == "__main__":
    asyncio.run(test_validator()) 