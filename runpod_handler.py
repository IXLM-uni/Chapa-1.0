import runpod
from sentence_transformers import SentenceTransformer
import torch
import json

# Загружаем модель один раз при запуске сервера
model = SentenceTransformer("all-MiniLM-L6-v2")
model.to("cuda" if torch.cuda.is_available() else "cpu")

def handler(event):
    try:
        # Печатаем входные данные для отладки
        print(f"Received event: {json.dumps(event)}")
        
        # Получаем input из event
        if isinstance(event, dict) and "input" in event:
            input_data = event["input"]
        else:
            # Если event не содержит input, используем его как input
            input_data = event
        
        print(f"Extracted input_data: {json.dumps(input_data)}")
        
        # Обработка одиночного текста
        if isinstance(input_data, dict) and "prompt" in input_data:
            prompt = input_data["prompt"]
            print(f"Processing single prompt: {prompt}")
            embedding = model.encode([prompt])[0].tolist()
            return {"embedding": embedding}
        
        # Обработка массива текстов
        elif isinstance(input_data, dict) and "texts" in input_data:
            texts = input_data["texts"]
            print(f"Processing {len(texts)} texts")
            
            # Проверка на пустой массив
            if not texts:
                return {"error": "Empty texts array"}
            
            # Обработка текстов батчами для экономии памяти
            batch_size = 128
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                batch_embeddings = model.encode(batch).tolist()
                all_embeddings.extend(batch_embeddings)
            
            print(f"Generated {len(all_embeddings)} embeddings")
            return {"embeddings": all_embeddings}
        
        else:
            error_msg = f"Input should contain 'texts' or 'prompt' field. Got: {json.dumps(input_data)}"
            print(error_msg)
            return {"error": error_msg}
            
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        print(error_msg)
        return {"error": error_msg}

# Запускаем serverless с нашим обработчиком
runpod.serverless.start({"handler": handler}) 