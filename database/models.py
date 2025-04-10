# database/models.py
from sqlalchemy import (Column, Integer, String, DateTime, ForeignKey, Float, 
                       ARRAY, BigInteger, Text, JSON, Boolean, UniqueConstraint, 
                       PrimaryKeyConstraint)
from sqlalchemy.orm import relationship, DeclarativeBase
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncAttrs

class Base(AsyncAttrs, DeclarativeBase):
    pass

# Справочники (Reference Tables)
class Users(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, nullable=False, index=True)
    tg_id = Column(BigInteger, unique=True, nullable=False)
    registration_date = Column(DateTime, default=datetime.utcnow)
    session_id = Column(String, unique=True, nullable=True)
    
    # Добавляем отношения
    chat_history = relationship("ChatMessageHistory", 
                               back_populates="user",
                               primaryjoin="Users.session_id==ChatMessageHistory.session_id")
    
    # Связь с каналами через промежуточную таблицу
    channels = relationship("Channels", 
                           secondary="users_channels",
                           back_populates="users")

class Themes(Base):
    __tablename__ = 'themes'

    id = Column(Integer, primary_key=True, nullable=False, index=True)
    theme_name = Column(String, unique=True, nullable=False)
    theme_prompt = Column(String, nullable=True)
    
    # Связь с каналами
    channels = relationship("Channels", back_populates="theme")

class Channels(Base):
    __tablename__ = 'channels'
    
    id = Column(Integer, primary_key=True, nullable=False, index=True)
    tg_channel_id = Column(BigInteger, unique=True, nullable=False)
    channel_name = Column(String, nullable=True)
    date = Column(DateTime, nullable=True)  # Поле для даты создания канала
    theme_id = Column(Integer, ForeignKey('themes.id'))
    is_new = Column(Boolean, default=True)
    is_unavailable = Column(Boolean, default=False)  # Новое поле для отметки недоступных каналов
    categories = Column(ARRAY(String))  # Правильное объявление массива
    styles = Column(ARRAY(String))
    sources_info = Column(ARRAY(String))
    description = Column(Text, nullable=True)  # Поле для описания канала
    participants_count = Column(Integer, nullable=True)  # Поле для количества участников
    telegram_link = Column(String, nullable=True)  # Поле для ссылки на канал
    key_themes = Column(ARRAY(String))
    main_theme = Column(ARRAY(String))
    
    # Добавляем обратную связь с channel_messages
    channel_messages = relationship("ChannelPosts", back_populates="channel", cascade="all, delete-orphan")
    
    # Связь с другими таблицами
    theme = relationship("Themes", back_populates="channels")    
    # Связь с пользователями через промежуточную таблицу
    users = relationship("Users", 
                       secondary="users_channels",
                       back_populates="channels")

    def __repr__(self):
        return f"<Channel(id={self.id}, name={self.channel_name})>"

# Новая промежуточная таблица для связи Users и Channels
class UsersChannels(Base):
    __tablename__ = 'users_channels'
    
    id = Column(Integer, primary_key=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    channel_id = Column(BigInteger, ForeignKey('channels.tg_channel_id'), nullable=False)
    
    # Добавим timestamp для отслеживания времени добавления канала
    added_at = Column(DateTime, default=datetime.utcnow)
    
    # Создаем уникальный индекс, чтобы предотвратить дублирование
    __table_args__ = (
        UniqueConstraint('user_id', 'channel_id', name='uix_user_channel'),
    )

# --- Добавляем новые модели ---

class Entities(Base):
    """Таблица-словарь для хранения уникальных сущностей (лемм)."""
    __tablename__ = 'entities'

    id = Column(Integer, primary_key=True, index=True, nullable=False)
    lemma = Column(Text, unique=True, nullable=False, index=True)
    embedding = Column(ARRAY(Float), nullable=True)

    posts = relationship("ChannelPosts",
                         secondary="post_entities",
                         back_populates="entities",
                         lazy="selectin")

    def __repr__(self):
        return f"<Entity(id={self.id}, lemma='{self.lemma}')>"

class PostEntities(Base):
    """Промежуточная таблица для связи 'многие-ко-многим'
       между постами (ChannelPosts) и сущностями (Entities)."""
    __tablename__ = 'post_entities'

    post_id = Column(Integer, ForeignKey('channel_messages.id', ondelete="CASCADE"), primary_key=True, nullable=False)
    entity_id = Column(Integer, ForeignKey('entities.id', ondelete="CASCADE"), primary_key=True, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint('post_id', 'entity_id', name='pk_post_entity'),
    )

# --- Обновляем существующую модель ChannelPosts ---

# Таблицы фактов (Fact Tables)
class ChannelPosts(Base):
    __tablename__ = 'channel_messages'

    id = Column(Integer, primary_key=True, nullable=False, index=True)
    message_id = Column(BigInteger, nullable=False, index=True)
    peer_id = Column(BigInteger, ForeignKey('channels.tg_channel_id'), nullable=False)
    date = Column(DateTime, nullable=True)
    message = Column(Text, nullable=True)
    views = Column(Integer, default=0)
    forwards = Column(Integer, default=0)
    post_author = Column(String, nullable=True)
    embedding = Column(ARRAY(Float), nullable=True)
    key_words = Column(ARRAY(String), nullable=True)
    message_link = Column(String, nullable=True)
    is_processed = Column(Boolean, default=False, nullable=False, index=True)
    
    channel = relationship("Channels", back_populates="channel_messages")
    
    # --- Новое отношение для связи с сущностями ---
    entities = relationship("Entities",
                            secondary="post_entities",
                            back_populates="posts",
                            lazy="selectin")
    
    __table_args__ = (
        UniqueConstraint('peer_id', 'message_id', name='uix_channel_message'),
    )

# Определяем ChatMessageHistory после всех остальных моделей
class ChatMessageHistory(Base):
    __tablename__ = 'chat_message_history'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String, ForeignKey('users.session_id'), nullable=False, index=True)
    message = Column(Text, nullable=False)
    
    # Связь с пользователем через session_id
    user = relationship("Users", back_populates="chat_history")