from sqlalchemy.future import select
from sqlalchemy import delete, update, exists
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.dialects.postgresql import insert as pg_insert
from datetime import datetime
from .models import Users, Channels, UsersChannels, ChannelPosts, Entities, PostEntities
import uuid
from typing import List, Optional, Dict, Tuple, Union
import asyncio
from sqlalchemy import desc
import logging
from sqlalchemy.exc import IntegrityError

logger = logging.getLogger(__name__)

class DatabaseRequests:
    def __init__(self, session_maker: async_sessionmaker):
        self._session_maker = session_maker

    @property
    def session(self) -> async_sessionmaker[AsyncSession]:
        """Возвращает фабрику сессий."""
        return self._session_maker

    async def get_user_by_telegram_id(self, telegram_id: int) -> Optional[Users]:
        """Получает пользователя по telegram_id."""
        async with self.session() as session:
            result = await session.execute(
                select(Users).where(Users.tg_id == telegram_id)
            )
            return result.scalar_one_or_none()

    async def add_user(self, telegram_id: int) -> Users:
        """Добавляет нового пользователя."""
        async with self.session() as session:
            new_user = Users(
                tg_id=telegram_id,
                registration_date=datetime.utcnow()
            )
            session.add(new_user)
            await session.commit()
            await session.refresh(new_user)
            return new_user

    async def check_user_exists(self, telegram_id: int) -> bool:
        """Проверяет существование пользователя."""
        async with self.session() as session:
            result = await session.execute(
                select(Users).where(Users.tg_id == telegram_id)
            )
            user = result.scalar_one_or_none()
            return user is not None

    async def get_or_create_session_id(self, telegram_id: int) -> str:
        """Получает или создает session_id для пользователя."""
        async with self.session() as session:
            user = await self.get_user_by_telegram_id(telegram_id)
            if not user.session_id:
                user.session_id = str(uuid.uuid4())
                await session.commit()
            return user.session_id

    async def _get_channel_by_tg_channel_id(self, tg_channel_id: str) -> Optional[Channels]:
        """Внутренний метод для получения канала по tg_channel_id."""
        async with self.session() as session:
            try:
                tg_channel_id_int = int(tg_channel_id)
            except ValueError:
                return None

            result = await session.execute(
                select(Channels).where(Channels.tg_channel_id == tg_channel_id_int)
            )
            return result.scalar_one_or_none()

    async def add_channel(self, channel_id, client, telegram_id: int, name: str = "") -> Tuple[Union[bool, None], bool, Optional[int]]:
        """
        Добавляет канал и создает связь с пользователем, с предварительной проверкой.

        Args:
            channel_id: ID канала в Telegram (tg_channel_id) или ссылка/юзернейм
            telegram_id: ID пользователя в Telegram
            client: Telethon клиент
            name: Название канала (не используется, если получаем из Telethon)

        Returns:
            Tuple[Union[bool, None], bool, Optional[int]]: Кортеж:
              - status: True (успешно), None (уже подписан), False (ошибка).
              - was_new: True, если канал был новым в БД, иначе False.
              - tg_channel_id: Числовой ID канала или None при ошибке.
        """
        tg_channel_id = None # Инициализируем на случай ранней ошибки
        was_channel_new = False # Флаг нового канала

        try:
            # Получение entity по username или ссылке
            logger.debug(f"Получение entity для идентификатора: {channel_id}")
            channel_entity = await client.get_entity(channel_id)
            if channel_entity is None:
                logger.error(f"Не удалось найти entity для идентификатора: {channel_id}")
                return False, was_channel_new, None # Возвращаем кортеж

            tg_channel_id = channel_entity.id # Присваиваем ID здесь
            channel_title = getattr(channel_entity, 'title', f"Канал {tg_channel_id}")
            channel_username = getattr(channel_entity, 'username', None)
            channel_date = getattr(channel_entity, 'date', None)
            if channel_date:
                channel_date = channel_date.replace(tzinfo=None)

            logger.debug(f"Получен канал: ID={tg_channel_id}, Title='{channel_title}', Username={channel_username}")

            # Пытаемся получить данные канала через LLM (если нужно)
            # Здесь можно добавить логику для channel_metadata, как было раньше,
            # но для исправления ошибки дубликата она не критична.
            # Пока пропустим LLM часть для ясности исправления.
            channel_theme_id = None
            channel_is_new = True # Предполагаем, что новый, если не анализировали
            channel_categories = []
            channel_styles = []
            channel_sources_info = []
            channel_key_themes = []
            channel_main_theme = []
            description = None
            participants_count = None
            telegram_link = f"https://t.me/{channel_username}" if channel_username else None
            # Можно добавить получение полной информации, как раньше, если нужно


            async with self.session() as session:
                # Находим пользователя по telegram_id
                user_result = await session.execute(
                    select(Users).where(Users.tg_id == telegram_id)
                )
                user = user_result.scalar_one_or_none()
                if not user:
                    logger.error(f"Пользователь с telegram_id {telegram_id} не найден.")
                    return False, was_channel_new, tg_channel_id # Возвращаем кортеж
                logger.debug(f"Найден пользователь ID: {user.id} для telegram_id: {telegram_id}")

                # Проверяем существует ли канал в таблице Channels
                channel_result = await session.execute(
                    select(Channels).where(Channels.tg_channel_id == tg_channel_id)
                )
                db_channel = channel_result.scalar_one_or_none()

                # Если канала нет, создаем новый
                if not db_channel:
                    logger.info(f"Канал ID {tg_channel_id} не найден в БД, создаем новый.")
                    was_channel_new = True # <<< Устанавливаем флаг
                    db_channel = Channels(
                        tg_channel_id=tg_channel_id,
                        channel_name=channel_title,
                        date=channel_date,
                        theme_id=channel_theme_id,
                        is_new=channel_is_new,
                        categories=channel_categories,
                        styles=channel_styles,
                        sources_info=channel_sources_info,
                        description=description,
                        participants_count=participants_count,
                        telegram_link=telegram_link,
                        key_themes=channel_key_themes,
                        main_theme=channel_main_theme,
                    )
                    session.add(db_channel)
                    # Используем flush, чтобы получить db_channel.id до коммита
                    await session.flush()
                    logger.info(f"Новый канал ID {tg_channel_id} добавлен в БД с ID {db_channel.id}.")
                else:
                     logger.debug(f"Канал ID {tg_channel_id} уже существует в БД (ID: {db_channel.id}).")


                # --- Предварительная проверка существования связи ---
                logger.debug(f"Проверка наличия связи: user_id={user.id}, channel_id={db_channel.id}")
                # Используем exists() для эффективности
                check_stmt = select(exists().where(
                    UsersChannels.user_id == user.id,
                    UsersChannels.channel_id == db_channel.id
                ))
                check_result = await session.execute(check_stmt)
                already_exists = check_result.scalar()

                if already_exists:
                    logger.info(f"Связь user={user.id}, channel={db_channel.id} уже существует.")
                    # Канал не новый (was_channel_new остается False), но связь есть
                    return None, was_channel_new, tg_channel_id # <<< Пользователь уже подписан
                # --- Конец предварительной проверки ---

                # Если связи нет, пытаемся ее создать
                try:
                    logger.debug(f"Связь не найдена. Попытка добавить: user_id={user.id}, channel_id={db_channel.id}")
                    users_channels = UsersChannels(
                        user_id=user.id,
                        channel_id=db_channel.id
                    )
                    session.add(users_channels)
                    # Теперь IntegrityError здесь маловероятна, но оставим try на всякий случай
                    await session.commit()
                    logger.info(f"Связь между пользователем {user.id} и каналом {db_channel.id} успешно создана.")
                    # Возвращаем True, флаг нового канала и ID
                    return True, was_channel_new, tg_channel_id # Успешное добавление

                except Exception as commit_exc:
                    # Ловим любую ошибку во время commit (если вдруг flush пропустил)
                    logger.error(f"Неожиданная ошибка при commit связи user={user.id}, channel={db_channel.id}: {commit_exc}", exc_info=True)
                    try:
                        await session.rollback() # Откатываем на всякий случай
                    except Exception as rollback_error:
                         logger.error(f"Ошибка при откате после ошибки commit: {rollback_error}", exc_info=True)
                    return False, was_channel_new, tg_channel_id # Возвращаем False при ошибке commit

        except Exception as e:
            # Эта ветка ловит ошибки, возникшие ДО блока async with session
            # (например, client.get_entity)
            logger.error(f"Общая ошибка (до работы с сессией) при добавлении канала ({channel_id}) для пользователя {telegram_id}: {e}", exc_info=True)
            return False, was_channel_new, tg_channel_id # Возвращаем кортеж

    async def _analyze_channel_with_llm(self, client, channel_id: int, message_limit: int = 20) -> dict:
        """
        Анализирует последние сообщения канала и определяет его категории и стили с помощью LLM.
        
        Args:
            client: Telethon клиент
            channel_id: ID канала в Telegram
            message_limit: Количество последних сообщений для анализа
            
        Returns:
            dict: Словарь с категориями канала или None в случае ошибки
        """
        try:
            # Получаем последние сообщения из канала
            messages = []
            try:
                async for message in client.iter_messages(channel_id, limit=message_limit):
                    if message.text:
                        messages.append(message.text)
            except Exception as e:
                if "The channel specified is private" in str(e):
                    print(f"Канал {channel_id} является приватным, доступ ограничен. Пропускаем.")
                    return None
                else:
                    # Пробрасываем другие ошибки
                    raise
            
            if not messages:
                print(f"Не удалось получить сообщения из канала {channel_id}")
                return None
            
            # Объединяем сообщения в один текст для анализа
            combined_text = "\n\n".join(messages)
            
            # Импортируем необходимые модули
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.output_parsers import JsonOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            
            # Создаем промпт-шаблон
            prompt = ChatPromptTemplate.from_messages([
                ("system", """
                Ты - аналитик телеграм-каналов. Твоя задача - проанализировать сообщения из канала
                и определить его тематику, стиль и типы источников информации.

                Определи следующие характеристики:
                1. Категории канала (например: новости, технологии, развлечения, психология). Выбери от 1 до 5 наиболее подходящих категорий из предложенного списка: [Технологии/AI, Гаджеты/Смартфоны, IT-Индустрия, Наука, Образование, Психология, Личностное развитие, Финансы/Экономика/Бизнес, Новости, Политика/Общество, Культура/Искусство, Спорт, Юмор/Развлечения, Путешествия, Городская жизнь, Природа/Экология, Здоровье/Медицина, Игры/Гейминг, Кино/Сериалы, Музыка, Мода/Стиль, Еда/Рецепты, Автомобили, Космос, История, Литература, Маркетинг/Реклама, Криптовалюты, Философия, Сад/Огород, Рукоделие/DIY]. Если ни одна из категорий не подходит, укажи "Другое".
                2. Стили изложения канала (например: информационный, развлекательный, аналитический, разговорный). Выбери от 1 до 5 наиболее подходящих стилей из предложенного списка: [Официальный/Нейтральный, Неофициальный/Разговорный, Аналитический, Юмористический/Ироничный, Информативный/Нейтральный, Эмоциональный/Экспрессивный, Технический/Профессиональный, Рекламный/Продающий, Личный/Интимный, Краткий/Телеграфный]. Если ни один из стилей не подходит, укажи "Другое".
                3. Типы источников информации канала. Выбери от 1 до 3 наиболее подходящих типа из предложенного списка: [Официальное СМИ, Блог/Личный блог, Социальная сеть/Мессенджер, Форум/Сообщество, Официальный источник компании, Агрегатор новостей/Дайджест, Слухи/Инсайдерская информация, Рекламная платформа/Объявление, Учебная платформа/Курс, Мероприятие/Концертная афиша, Экспертный блог, Новостной блог, Тематический блог, Личный канал]. Если ни один из типов не подходит, укажи "Другое".
                4. Ключевые темы канала. Определи до 10 ключевых тем, наиболее часто встречающихся в сообщениях канала. Темы должны быть конкретными и информативными (например: LLM, iPhone 17, коучинг, путешествия по России, выборы 2024).
                5. Основная тема канала. Выбери ОДНУ основную тему, которая наиболее точно характеризует канал в целом.  Основная тема должна быть максимально общей и емкой (например: Новости технологий, Психология и саморазвитие, Личный блог о путешествиях).

                Возврати только JSON-объект с полями:
                - categories: список категорий канала
                - styles_izlozheniya: список стилей изложения канала
                - source_types: список типов источников информации канала
                - key_themes: список ключевых тем канала
                - main_theme: основная тема канала (строка)

                В списках categories, styles_izlozheniya и source_types должно быть от 1 до 5 элементов (для source_types от 1 до 3).
                Поле main_theme должно содержать только одну строку.
                """),
                ("human", "{text}")
            ])
            
            # Создаем LLM
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
            
            # Создаем парсер JSON
            parser = JsonOutputParser()
            
            # Создаем цепочку обработки
            chain = prompt | llm | parser
            
            # Получаем результат и возвращаем его
            response = await chain.ainvoke({"text": combined_text})
            
            # Проверяем наличие всех необходимых полей
            required_fields = ['categories', 'styles', 'sources_info', 'key_themes', 'main_theme']
            for field in required_fields:
                if field not in response:
                    response[field] = []
            
            # Фиксируем sources_info, чтобы там был только один элемент
            if 'sources_info' in response and len(response['sources_info']) > 1:
                # Если в массиве больше одного элемента, оставляем только первый
                response['sources_info'] = [response['sources_info'][0]]
            elif not response.get('sources_info'):
                # Если массив пустой, добавляем "информативно" по умолчанию
                response['sources_info'] = ["информативно"]
            
            print(f"Успешно определены метаданные для канала {channel_id}: {response}")
            return response
            
        except Exception as e:
            print(f"Ошибка при анализе канала с помощью LLM: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def update_channel_metadata(self, channel_id: int) -> bool:
        """
        Обновляет метаданные канала с помощью LLM.
        
        Args:
            channel_id: ID канала в Telegram
            
        Returns:
            bool: Успешно ли обновлены метаданные
        """
        try:
            # Получаем канал из БД
            async with self.session() as session:
                result = await session.execute(
                    select(Channels).where(Channels.tg_channel_id == channel_id)
                )
                channel = result.scalar_one_or_none()
                
                if not channel:
                    print(f"Канал с ID {channel_id} не найден в БД")
                    return False
                
                # Получаем parser из класса
                if not hasattr(self, '_parser') or self._parser is None:
                    print(f"Parser не инициализирован для DatabaseRequests")
                    return False
                
                try:
                    # Получаем метаданные с помощью LLM
                    metadata = await self._analyze_channel_with_llm(
                        client=self._parser.client, 
                        channel_id=channel_id
                    )
                    
                    if not metadata:
                        print(f"Не удалось получить метаданные для канала {channel_id}")
                        return False
                    
                    # Обновляем метаданные в БД
                    channel.categories = metadata.get('categories', [])
                    channel.styles = metadata.get('styles', [])
                    channel.sources_info = metadata.get('sources_info', [])
                    channel.key_themes = metadata.get('key_themes', [])
                    channel.main_theme = metadata.get('main_theme', [])
                    channel.is_new = False
                    
                    await session.commit()
                    print(f"Метаданные для канала {channel_id} успешно обновлены")
                    return True
                    
                except Exception as e:
                    if "The channel specified is private" in str(e):
                        print(f"Канал {channel_id} является приватным, доступ ограничен. Пропускаем.")
                        # Отмечаем канал как обработанный, чтобы не пытаться обновлять его постоянно
                        channel.is_new = False
                        if not channel.categories:
                            channel.categories = ["Приватный канал"]
                        if not channel.styles:
                            channel.styles = ["Недоступен"]
                        if not channel.sources_info:
                            channel.sources_info = ["неинформативно"]
                        if not channel.key_themes:
                            channel.key_themes = ["Недоступно"]
                        if not channel.main_theme:
                            channel.main_theme = ["Приватный канал"]
                        
                        await session.commit()
                        print(f"Канал {channel_id} отмечен как приватный с базовыми метаданными")
                        return False
                    else:
                        # Пробрасываем другие ошибки
                        raise
                
        except Exception as e:
            print(f"Ошибка при обновлении метаданных канала {channel_id}: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def update_all_empty_channel_metadata(self, force_update: bool = False, max_channels: int = None):
        """
        Находит все каналы с пустыми метаданными и обновляет их.
        Используется при запуске бота.
        
        Args:
            force_update: Если True, обновляет метаданные всех каналов, даже если они уже заполнены
            max_channels: Максимальное количество каналов для обработки за один запуск
        """
        try:
            # Проверяем, что парсер инициализирован
            if not hasattr(self, '_parser') or self._parser is None:
                print("Parser не инициализирован для DatabaseRequests. Невозможно обновить метаданные.")
                return
            
            # Проверяем, что клиент доступен
            if not self._parser.client:
                print("Telethon клиент не доступен. Невозможно обновить метаданные.")
                return
                
            # Получаем все каналы из БД
            async with self.session() as session:
                result = await session.execute(
                    select(Channels)
                )
                channels = result.scalars().all()
                
                channels_to_update = []
                
                for channel in channels:
                    # Если force_update=True, обновляем все каналы
                    # Иначе проверяем наличие пустых полей
                    if force_update or (
                        not channel.categories or 
                        not channel.styles or 
                        not channel.sources_info or 
                        not channel.key_themes or 
                        not channel.main_theme
                    ):
                        channels_to_update.append(channel)
                
                # Ограничиваем количество каналов для обработки
                if max_channels and len(channels_to_update) > max_channels:
                    print(f"Ограничиваем обработку до {max_channels} каналов из {len(channels_to_update)}")
                    channels_to_update = channels_to_update[:max_channels]
                
                print(f"Найдено {len(channels_to_update)} каналов для обновления метаданных")
                
                updated_count = 0
                for channel in channels_to_update:
                    print(f"Обновляем метаданные для канала: {channel.channel_name} (ID: {channel.tg_channel_id})")
                    
                    try:
                        # Обновляем метаданные
                        success = await self.update_channel_metadata(channel.tg_channel_id)
                        if success:
                            updated_count += 1
                    except Exception as e:
                        if "The channel specified is private" in str(e):
                            print(f"Канал {channel.channel_name} (ID: {channel.tg_channel_id}) является приватным, пропускаем")
                        else:
                            print(f"Ошибка при обновлении метаданных канала {channel.channel_name}: {e}")
                    
                    # Добавляем небольшую задержку, чтобы не перегружать API
                    await asyncio.sleep(1)
                
                print(f"Обновлено метаданных для {updated_count} каналов из {len(channels_to_update)}")
                
        except Exception as e:
            print(f"Ошибка при массовом обновлении метаданных каналов: {e}")
            import traceback
            traceback.print_exc()

    async def get_user_channels(self, telegram_id: int) -> List[Dict[str, str]]:
        """Получает список каналов пользователя."""
        async with self.session() as session:
            # Прямой запрос с JOIN между Users, UsersChannels и Channels
            result = await session.execute(
                select(Channels)
                .join(UsersChannels, UsersChannels.channel_id == Channels.id)
                .join(Users, Users.id == UsersChannels.user_id)
                .where(Users.tg_id == telegram_id)
            )
            
            channels = result.scalars().all()
            
            # Формируем список каналов
            channels_list = []
            for channel in channels:
                channels_list.append({
                    "id": str(channel.tg_channel_id),
                    "title": channel.channel_name or f"Канал {channel.tg_channel_id}"
                })
            
            return channels_list

    async def delete_user_channel(self, channel_id: str, telegram_id: int) -> bool:
        """Удаляет связь пользователя с каналом."""
        try:
            async with self.session() as session:
                # Получаем пользователя 
                user_result = await session.execute(
                    select(Users).where(Users.tg_id == telegram_id)
                )
                user = user_result.scalar_one_or_none()
                if not user:
                    return False
                
                # Получаем канал
                channel_result = await session.execute(
                    select(Channels).where(Channels.tg_channel_id == int(channel_id))
                )
                channel = channel_result.scalar_one_or_none()
                if not channel:
                    return False
                
                # Удаляем связь
                await session.execute(
                    delete(UsersChannels).where(
                        (UsersChannels.user_id == user.id) & 
                        (UsersChannels.channel_id == channel.id)
                    )
                )
                
                await session.commit()
                return True
        except Exception as e:
            print(f"Ошибка при удалении канала: {e}")
            return False

    async def get_channels_from_db(self):
        """Получает список каналов из базы данных с их статусом"""
        async with self.session() as session:
            result = await session.execute(select(Channels))
            return result.scalars().all()

    async def update_channel_status(self, channel_id: int, is_new: bool = False):
        """Обновляет статус канала после успешного парсинга"""
        async with self.session() as session:
            result = await session.execute(select(Channels).where(Channels.tg_channel_id == channel_id))
            channel = result.scalar_one_or_none()
            if channel:
                channel.is_new = is_new
                await session.commit()
                print(f"Статус канала {channel_id} обновлен на {'новый' if is_new else 'не новый'}")

    async def get_last_message_id_from_db(self, channel_id: int) -> int | None:
        """Получает ID последнего сообщения, сохраненного в БД для канала."""
        async with self.session() as session:
            # Сначала проверяем, есть ли сообщения в БД для этого канала
            result = await session.execute(
                select(ChannelPosts.message_id)
                .where(ChannelPosts.peer_id == channel_id)
                .order_by(desc(ChannelPosts.message_id))
                .limit(1)
            )
            last_message_id = result.scalar_one_or_none()
            
            # Если сообщений нет, проверяем, существует ли канал в таблице Channels
            if last_message_id is None:
                channel_result = await session.execute(
                    select(Channels).where(Channels.tg_channel_id == channel_id)
                )
                channel = channel_result.scalar_one_or_none()
                
                # Если канал существует, возвращаем 0 чтобы начать парсинг с начала
                if channel:
                    print(f"DEBUG: get_last_message_id_from_db: Канал {channel_id} существует, но сообщений нет. Начинаем с ID=0")
                    return 0
            
            print(f"DEBUG: get_last_message_id_from_db: Для канала {channel_id} последний message_id в БД: {last_message_id}")
            return last_message_id

    def set_parser(self, parser):
        """Устанавливает экземпляр парсера для использования."""
        self._parser = parser

    async def get_all_channels_with_last_messages(self):
        """Получает все каналы и их последние сообщения, исключая недоступные"""
        async with self.session() as session:
            # Получаем только доступные каналы
            channels = await session.execute(
                select(Channels.tg_channel_id)
                .where(Channels.is_unavailable == False)
            )
            channel_ids = [row[0] for row in channels]
            
            # Получаем последние сообщения для доступных каналов
            last_messages = {}
            for channel_id in channel_ids:
                last_message = await session.execute(
                    select(ChannelPosts.message_id)
                    .where(ChannelPosts.peer_id == channel_id)
                    .order_by(ChannelPosts.message_id.desc())
                    .limit(1)
                )
                last_message_id = last_message.scalar_one_or_none()
                if last_message_id:
                    last_messages[channel_id] = last_message_id
                else:
                    last_messages[channel_id] = None
                    
            return last_messages

    async def get_all_message_embeddings(self):
        """
        Асинхронно получает все id, peer_id, текст и эмбеддинги из базы данных одним запросом.

        Returns:
            List[Tuple[int, int, str, List[float]]]: Список кортежей (id, peer_id, message, embedding).
                                                     Возвращает пустой список, если эмбеддинги не найдены.
        """
        try:
            async with self.session() as session:
                # Получаем только непустые эмбеддинги, добавляем peer_id и message
                query = select(ChannelPosts.id, ChannelPosts.peer_id, ChannelPosts.message, ChannelPosts.embedding).where(
                    ChannelPosts.embedding != None,
                    ChannelPosts.message != None, # Добавляем проверку на непустое сообщение
                    ChannelPosts.message != ''
                )
                result = await session.execute(query)
                message_embeddings_data = result.all()

            if not message_embeddings_data:
                print("В базе данных не найдено сообщений с эмбеддингами")
                return []

            # Фильтруем пустые эмбеддинги и проверяем peer_id
            valid_embeddings = []
            for msg_id, peer_id, message, embedding in message_embeddings_data:
                # Проверяем, что peer_id не None
                if peer_id is None:
                    print(f"Пропущен эмбеддинг для сообщения ID: {msg_id} из-за отсутствия peer_id")
                    continue
                if embedding and len(embedding) > 0:
                    # Текст (message) пока просто пробрасываем дальше
                    valid_embeddings.append((msg_id, peer_id, message, embedding))
                else:
                    print(f"Пропущен пустой эмбеддинг для сообщения ID: {msg_id}")

            print(f"Загружено {len(valid_embeddings)} валидных эмбеддингов с peer_id и текстом из {len(message_embeddings_data)} записей")
            return valid_embeddings

        except Exception as e:
            print(f"Ошибка при получении эмбеддингов из БД: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def get_document_by_id(self, msg_id: int) -> Optional[dict]:
        """
        Получает документ по ID из базы данных.
        
        Args:
            msg_id: ID сообщения
            
        Returns:
            Optional[dict]: Документ или None
        """
        try:
            async with self.session() as session:
                query = select(ChannelPosts).where(ChannelPosts.id == msg_id)
                result = await session.execute(query)
                post = result.scalars().first()
                
                if not post:
                    print(f"Документ с ID {msg_id} не найден")
                    return None
                
                # Получаем ссылку на сообщение напрямую из столбца message_link
                message_link = post.message_link
                
                return {
                    "id": post.id,
                    "text": post.message,
                    "author": post.post_author,
                    "date": post.date,
                    "channel_id": post.peer_id,
                    "message_link": message_link,  # Добавляем ссылку на сообщение
                    "message_id": post.message_id  # Добавляем ID сообщения
                }
                
        except Exception as e:
            print(f"Ошибка при получении документа с ID {msg_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def get_channel_link(self, channel_id: int) -> Optional[str]:
        """
        Получает ссылку на телеграм-канал по его ID.
        
        Args:
            channel_id: ID канала в Telegram
            
        Returns:
            Optional[str]: Ссылка на канал или None
        """
        try:
            async with self.session() as session:
                query = select(Channels).where(Channels.tg_channel_id == channel_id)
                result = await session.execute(query)
                channel = result.scalar_one_or_none()
                
                if not channel:
                    logger.warning(f"Канал с ID {channel_id} не найден в базе данных")
                    return None
                
                # Возвращаем ссылку на канал, если она есть
                if channel.telegram_link:
                    return channel.telegram_link
                
                # Если ссылки нет, но есть username, формируем ссылку
                if hasattr(channel, 'username') and channel.username:
                    return f"https://t.me/{channel.username}"
                
                # Если нет ни ссылки, ни username, возвращаем стандартную ссылку с ID
                return f"https://t.me/c/{channel_id}"
                
        except Exception as e:
            logger.error(f"Ошибка при получении ссылки на канал с ID {channel_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def get_message_link(self, msg_id: int) -> Optional[str]:
        """
        Получает прямую ссылку на сообщение по его ID.
        
        Args:
            msg_id: ID сообщения в таблице ChannelPosts
            
        Returns:
            Optional[str]: Ссылка на сообщение или None
        """
        try:
            async with self.session() as session:
                query = select(ChannelPosts).where(ChannelPosts.id == msg_id)
                result = await session.execute(query)
                post = result.scalar_one_or_none()
                
                if not post:
                    logger.warning(f"Сообщение с ID {msg_id} не найдено в базе данных")
                    return None
                
                # Возвращаем прямую ссылку на сообщение, если она есть
                if post.message_link:
                    return post.message_link
                
                # Если нет прямой ссылки, но есть peer_id и message_id, формируем ссылку
                if post.peer_id and post.message_id:
                    # Проверяем, есть ли информация о канале с username
                    channel_link = await self.get_channel_link(post.peer_id)
                    if channel_link and channel_link.startswith("https://t.me/"):
                        # Получаем username из ссылки на канал
                        username = channel_link.split("https://t.me/")[1].split("/")[0]
                        if username:
                            return f"https://t.me/{username}/{post.message_id}"
                    
                    # Если канал без username, используем формат с peer_id
                    return f"https://t.me/c/{post.peer_id}/{post.message_id}"
                
                return None
                
        except Exception as e:
            logger.error(f"Ошибка при получении ссылки на сообщение с ID {msg_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def get_channel_by_tg_id(self, tg_channel_id):
        """Получает канал по его telegram_id"""
        async with self.session() as session:
            channel = await session.execute(
                select(Channels).where(Channels.tg_channel_id == tg_channel_id)
            )
            return channel.scalar_one_or_none()

    async def add_entities_for_post(self, post_id: int, lemmas: List[str]) -> None:
        """Добавляет сущности (леммы) для поста в базу данных, игнорируя дубликаты."""
        if not lemmas:
            return
        
        try:
            async with self.session() as session:
                # 1. Получаем или создаем все нужные сущности (Entities)
                entities_to_link = []
                for lemma in lemmas:
                    # Используем ON CONFLICT DO NOTHING для Entities, если лемма уже есть
                    stmt_entity = pg_insert(Entities).values(lemma=lemma)
                    stmt_entity = stmt_entity.on_conflict_do_nothing(index_elements=['lemma'])
                    await session.execute(stmt_entity)
                    
                    # Получаем ID существующей или только что созданной сущности
                    result = await session.execute(select(Entities.id).where(Entities.lemma == lemma))
                    entity_id = result.scalar_one_or_none()
                    if entity_id:
                        entities_to_link.append(entity_id)
                
                # Убираем дубликаты entity_id, если они есть
                unique_entity_ids = list(set(entities_to_link))
                
                if not unique_entity_ids:
                    return

                # 2. Создаем связи в PostEntities, игнорируя конфликты
                values_to_insert = [{'post_id': post_id, 'entity_id': eid} for eid in unique_entity_ids]
                
                stmt_post_entity = pg_insert(PostEntities).values(values_to_insert)
                # Игнорируем конфликты по первичному ключу (post_id, entity_id)
                stmt_post_entity = stmt_post_entity.on_conflict_do_nothing(index_elements=['post_id', 'entity_id'])
                
                await session.execute(stmt_post_entity)
                await session.commit()
                # Убираем лог, так как он может быть слишком частым
                # print(f"Добавлены/проигнорированы сущности для поста {post_id}")
                
        except Exception as e:
            # Логируем ошибку, но не прерываем весь процесс из-за одного поста
            logger.error(f"Ошибка при добавлении сущностей для поста {post_id}: {e}", exc_info=True)
            # Не пробрасываем ошибку дальше, чтобы цикл обработки мог продолжиться
            # raise

    async def get_entity_ids_by_lemmas(self, lemmas: List[str]) -> List[int]:
        """Получает ID сущностей по списку лемм."""
        if not lemmas:
            return []
        try:
            async with self.session() as session:
                stmt = select(Entities.id).where(Entities.lemma.in_(lemmas))
                result = await session.execute(stmt)
                return result.scalars().all()
        except Exception as e:
            logger.error(f"Ошибка при получении ID сущностей по леммам {lemmas}: {e}", exc_info=True)
            return []

    async def get_post_ids_by_entity_ids(self, entity_ids: List[int]) -> List[int]:
        """Получает список ID постов, связанных с заданными ID сущностей."""
        if not entity_ids:
            return []
        try:
            async with self.session() as session:
                stmt = select(PostEntities.post_id).where(PostEntities.entity_id.in_(entity_ids)).distinct()
                result = await session.execute(stmt)
                # Преобразуем результат в список int
                post_ids = [pid for pid, in result.fetchall()]
                return post_ids
        except Exception as e:
            logger.error(f"Ошибка при получении ID постов по ID сущностей {entity_ids}: {e}", exc_info=True)
            return []

    async def get_unprocessed_posts(self, batch_size: int) -> List[ChannelPosts]:
        """Получает порцию необработанных постов (is_processed=False)."""
        try:
            async with self.session() as session:
                stmt = (
                    select(ChannelPosts)
                    .where(
                        ChannelPosts.is_processed == False, 
                        ChannelPosts.message != None,
                        ChannelPosts.message != ''
                    )
                    .order_by(ChannelPosts.id)
                    .limit(batch_size)
                )
                result = await session.execute(stmt)
                return result.scalars().all()
        except Exception as e:
            logger.error(f"Ошибка при получении необработанных постов: {e}", exc_info=True)
            return []

    async def update_post_embedding(self, post_id: int, embedding: List[float]) -> bool:
        """Обновляет эмбеддинг для указанного поста."""
        try:
            async with self.session() as session:
                stmt = (
                    update(ChannelPosts)
                    .where(ChannelPosts.id == post_id)
                    .values(embedding=embedding)
                )
                await session.execute(stmt)
                await session.commit()
                return True
        except Exception as e:
            logger.error(f"Ошибка при обновлении эмбеддинга для поста {post_id}: {e}", exc_info=True)
            return False

    async def mark_posts_as_processed(self, post_ids: List[int]) -> bool:
        """Помечает список постов как обработанные (is_processed=True)."""
        if not post_ids:
            return True # Нечего делать
        try:
            async with self.session() as session:
                stmt = (
                    update(ChannelPosts)
                    .where(ChannelPosts.id.in_(post_ids))
                    .values(is_processed=True)
                )
                await session.execute(stmt)
                await session.commit()
                return True
        except Exception as e:
            logger.error(f"Ошибка при пометке постов {post_ids} как обработанных: {e}", exc_info=True)
            return False