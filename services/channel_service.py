from database.requests import DatabaseRequests
from typing import List, Dict, Optional, Tuple, Any

class ChannelService:
    """Сервис для работы с каналами."""
    
    def __init__(self, db: Optional[DatabaseRequests] = None):
        """Инициализация сервиса с базой данных."""
        self.db = db
    
    async def get_user_channels(self, user_id: int) -> List[Dict[str, str]]:
        """
        Получает список каналов пользователя.
        
        Args:
            user_id: Telegram ID пользователя
            
        Returns:
            Список словарей с информацией о каналах: [{"id": "123", "title": "Channel Name"}]
        """
        if not user_id:
            print("ERROR: get_user_channels вызван с пустым user_id")
            return []
        
        try:
            channels = await self.db.get_user_channels(telegram_id=user_id)
            return channels or []
        except Exception as e:
            print(f"ERROR в ChannelService.get_user_channels: {e}")
            import traceback
            print(traceback.format_exc())
            return []
    
    async def add_channel(self, user_id: int, channel_id: int, channel_name: str = "") -> Tuple[bool, str]:
        """
        Добавляет канал пользователю.
        
        Args:
            user_id: Telegram ID пользователя
            channel_id: Telegram ID канала
            channel_name: Название канала
            
        Returns:
            Tuple[success, message]: Успешность операции и сообщение
        """
        try:
            # Проверяем, есть ли уже такой канал у пользователя
            channels = await self.db.get_user_channels(telegram_id=user_id)
            for ch in channels:
                if ch['id'] == str(channel_id):
                    return False, f"Канал '{channel_name}' уже есть в вашем списке."
            
            # Добавляем канал
            added = await self.db.add_channel(
                channel_id=channel_id, 
                telegram_id=user_id, 
                name=channel_name
            )
            
            if added:
                return True, f"Канал '{channel_name}' успешно добавлен!"
            else:
                return False, "Не удалось добавить канал. Попробуйте позже."
                
        except Exception as e:
            print(f"ERROR в ChannelService.add_channel: {e}")
            import traceback
            print(traceback.format_exc())
            return False, f"Произошла ошибка: {str(e)}"
    
    async def delete_channel(self, user_id: int, channel_id: int) -> Tuple[bool, str]:
        """
        Удаляет канал пользователя.
        
        Args:
            user_id: Telegram ID пользователя
            channel_id: Telegram ID канала для удаления
            
        Returns:
            Tuple[success, message]: Успешность операции и сообщение
        """
        try:
            deleted = await self.db.delete_user_channel(str(channel_id), user_id)
            
            if deleted:
                return True, f"Канал успешно удален."
            else:
                return False, "Не удалось удалить канал. Возможно, он не существует или произошла ошибка."
                
        except Exception as e:
            print(f"ERROR в ChannelService.delete_channel: {e}")
            import traceback
            print(traceback.format_exc())
            return False, f"Произошла ошибка: {str(e)}"
    
    async def get_channel_summary(self, user_id: int, channel_id: Optional[int] = None) -> str:
        """
        Получает сводку по каналу пользователя.
        
        Args:
            user_id: Telegram ID пользователя
            channel_id: Telegram ID канала (опционально)
            
        Returns:
            Текст сводки или сообщение об ошибке
        """
        # TODO: Реализовать в будущем
        return "Получение сводки по каналу пока не реализовано."
    
    async def ask_question_about_channel(self, user_id: int, question: str, channel_id: Optional[int] = None) -> str:
        """
        Отвечает на вопрос о содержании канала.
        
        Args:
            user_id: Telegram ID пользователя
            question: Текст вопроса
            channel_id: Telegram ID канала (опционально)
            
        Returns:
            Ответ на вопрос или сообщение об ошибке
        """
        # TODO: Реализовать в будущем
        return "Возможность задавать вопросы по каналу пока не реализована."

    async def channels_to_text(self, user_id: int) -> str:
        """
        Преобразует список каналов пользователя в текстовый формат.
        
        Args:
            user_id: Telegram ID пользователя
            
        Returns:
            Текстовое представление списка каналов
        """
        channels = await self.get_user_channels(user_id)
        
        if not channels:
            return "У вас пока нет добавленных каналов."
        
        channels_text = "\n".join([f"• {ch['title']} (ID: {ch['id']})" for ch in channels])
        return f"Ваши каналы:\n{channels_text}" 