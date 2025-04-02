# bot/middlewares/user_middleware.py

from typing import Callable, Dict, Any, Awaitable
from aiogram import BaseMiddleware
from aiogram.types import Message
from database.requests import DatabaseRequests
from sqlalchemy.ext.asyncio import AsyncSession


class UserMiddleware(BaseMiddleware):
    def __init__(self, db: DatabaseRequests):
        self.db = db
        super().__init__()

    async def __call__(
        self,
        handler: Callable[[Message, Dict[str, Any]], Awaitable[Any]],
        event: Message,
        data: Dict[str, Any]
    ) -> Any:
        """
        Этот middleware проверяет, зарегистрирован ли пользователь в БД, и регистрирует, если нет.
        """
        # Получаем telegram_id пользователя
        user_telegram_id = event.from_user.id
        
        async with self.db.session() as session:
            # Проверяем существование пользователя - передаем telegram_id только как именованный аргумент
            user = await self.db.get_user_by_telegram_id(telegram_id=user_telegram_id)
            
            if not user:
                # Если пользователь не существует, регистрируем его
                await self.db.add_user(session=session, telegram_id=user_telegram_id)
            
            # Добавляем сессию в контекст для использования в хэндлерах
            data["session"] = session
            
            # Вызываем следующий обработчик
            return await handler(event, data)