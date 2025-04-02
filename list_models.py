import google.generativeai as genai

# Конфигурируем API
genai.configure(api_key='AIzaSyAR3IRvu_WIrMPfbnyL5wyhcgXBW2UCGcU')

# Получаем список моделей
models = genai.list_models()

# Выводим информацию о каждой модели
for model in models:
    print(f"\nМодель: {model.name}")
    print(f"Отображаемое имя: {model.display_name}")
    print(f"Описание: {model.description}")
    print(f"Поддерживаемые генерации: {', '.join(model.supported_generation_methods)}")
    print("-" * 50) 