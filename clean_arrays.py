import logging
import re
import traceback
from collections import Counter
from typing import Tuple, List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCleaner:
    MIN_EXAMPLES_PER_LABEL = 50
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Очищает текст от лишних пробелов и пунктуации по краям"""
        # Заменяем множественные пробелы на один
        text = re.sub(r'\s+', ' ', text)
        # Удаляем пробелы и пунктуацию по краям
        text = text.strip().strip('.,!?:;«»""\'')
        return text

    @staticmethod
    def find_entity_coordinates(text: str, entity_text: str) -> Optional[Tuple[int, int]]:
        """Находит точные координаты сущности в тексте с учетом пробелов"""
        try:
            # Очищаем текст сущности от лишних пробелов
            clean_entity = DataCleaner.clean_text(entity_text)
            if not clean_entity:
                return None
                
            # Ищем все вхождения сущности в тексте
            matches = list(re.finditer(re.escape(clean_entity), text))
            if not matches:
                return None
                
            # Берем первое вхождение
            match = matches[0]
            start, end = match.span()
            
            # Проверяем границы сущности на пробелы
            if (start > 0 and not text[start-1].isspace()) or \
               (end < len(text) and not text[end].isspace() and text[end] not in '.,!?:;»"\''):
                return None
                
            return (start, end)
            
        except Exception as e:
            logger.error(f"Ошибка при поиске координат сущности '{entity_text}': {e}")
            return None

    @staticmethod
    def clean_between_arrays():
        """Очищает файл и подготавливает данные для обучения"""
        try:
            logger.info("Начало очистки файла...")
            
            # Читаем исходный файл
            with open('TEST_DATA.txt', 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
                logger.info(f"Количество строк в TEST_DATA.txt: {len(lines)}")
            
            # Статистика
            original_size = len(content)
            label_counter = Counter()
            invalid_spans = 0
            processed_entities = 0
            
            # Находим все элементы
            pattern = r"\(\'([^\']+)\',\s*\{\'entities\':\s*\[(.*?)\]\}\)"
            matches = re.finditer(pattern, content)
            
            notna_content = []
            
            for match in matches:
                text = DataCleaner.clean_text(match.group(1))
                entities_str = match.group(2)
                
                # Обрабатываем сущности
                entities = []
                if entities_str.strip():
                    entity_pattern = r"\(\'([^\']+)\',\s*\'([^\']+)\'\)"
                    entity_matches = re.finditer(entity_pattern, entities_str)
                    
                    for entity_match in entity_matches:
                        processed_entities += 1
                        entity_text = entity_match.group(1)
                        entity_type = entity_match.group(2).strip().upper()
                        
                        # Пропускаем пустые или некорректные сущности
                        if not entity_text or not entity_type:
                            continue
                            
                        # Находим координаты с проверкой пробелов
                        coords = DataCleaner.find_entity_coordinates(text, entity_text)
                        if coords:
                            entities.append(f"({coords[0]}, {coords[1]}, '{entity_type}')")
                            label_counter[entity_type] += 1
                        else:
                            invalid_spans += 1
                
                # Добавляем только если есть валидные сущности
                if entities:
                    element = f"('{text}', {{'entities': [{', '.join(entities)}]}})"
                    notna_content.append(element)
            
            # Записываем очищенные данные
            notna_cleaned_content = ",\n".join(notna_content)
            with open('TEST_DATA_NOTNA_CLEAN.txt', 'w', encoding='utf-8') as f:
                f.write(notna_cleaned_content)
            
            # Выводим подробную статистику
            logger.info("\n=== Статистика обработки ===")
            logger.info(f"Всего обработано сущностей: {processed_entities}")
            logger.info(f"Некорректных спанов: {invalid_spans}")
            logger.info(f"Итоговых документов: {len(notna_content)}")
            
            logger.info("\n=== Распределение меток ===")
            for label, count in sorted(label_counter.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"- {label}: {count} примеров")
                if count < DataCleaner.MIN_EXAMPLES_PER_LABEL:
                    logger.warning(f"⚠ Недостаточно примеров для метки {label} (нужно минимум {DataCleaner.MIN_EXAMPLES_PER_LABEL})")
            
            logger.info("\n=== Размеры файлов ===")
            logger.info(f"Исходный размер: {original_size:,} байт")
            logger.info(f"Итоговый размер: {len(notna_cleaned_content):,} байт")
            logger.info(f"Процент сжатия: {((original_size - len(notna_cleaned_content)) / original_size * 100):.1f}%")
            
        except Exception as e:
            logger.error(f"Ошибка при очистке файла: {e}")
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    DataCleaner.clean_between_arrays()