import logging  # Импорт модуля для логирования
from datetime import datetime  # Импорт функции для получения текущего времени
from sqlalchemy import Column, Integer, String, DateTime, create_engine  # Импорт необходимых классов для описания столбцов и создания движка БД
from sqlalchemy.orm import declarative_base, sessionmaker  # Импорт базового класса и фабрики сессий

# Настройка логирования: формат, уровень логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Формат сообщений
    level=logging.INFO  # Уровень логирования: INFO
)
logger = logging.getLogger(__name__)  # Создаем логгер для текущего модуля

# Создаем базовый класс для описания моделей SQLAlchemy
Base = declarative_base()

# Описание модели для хранения информации об обработанных изображениях
class ClassifiedImage(Base):
    __tablename__ = "classified_images"  # Имя таблицы в базе данных

    # Описание столбцов таблицы
    id = Column(Integer, primary_key=True, autoincrement=True)  # Идентификатор записи, автоинкремент
    user_id = Column(String(255), nullable=False)  # ID пользователя Telegram (обязательное поле)
    image_path = Column(String(1024), nullable=False)  # Путь к сохраненному изображению (обязательное поле)
    classification_result = Column(String(2048), nullable=False)  # Результат классификации (обязательное поле)
    created_at = Column(DateTime, default=datetime.utcnow)  # Время создания записи, по умолчанию текущее UTC время

# Создаем SQLite движок. Путь к файлу базы данных можно изменить при необходимости.
engine = create_engine("sqlite:///db.sqlite3", echo=False)

# Создаем фабрику сессий для работы с базой данных
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """
    Инициализация базы данных:
      - Создает таблицы, если они еще не созданы.
    """
    Base.metadata.create_all(engine)  # Создает все таблицы, описанные в моделях, в движке
    logger.info("Таблицы базы данных успешно созданы или проверены.")

# Если модуль запущен напрямую, инициализировать базу данных
if __name__ == "__main__":
    init_db()