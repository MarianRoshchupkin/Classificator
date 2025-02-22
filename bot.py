import logging  # Модуль для логирования
import os  # Модуль для работы с операционной системой
import io  # Модуль для работы с потоками ввода/вывода
import time  # Модуль для работы со временем
import numpy as np  # Библиотека для работы с массивами и матрицами
from PIL import Image  # Библиотека для обработки изображений
from dotenv import load_dotenv  # Загрузка переменных окружения из файла .env
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton  # Импорт классов для работы с Telegram API
from telegram.constants import ParseMode  # Импорт константы ParseMode для форматирования сообщений
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters  # Импорт необходимых обработчиков команд и сообщений
from ultralytics import YOLO  # Импорт модели YOLOv8 из пакета ultralytics
from models import init_db, SessionLocal, ClassifiedImage  # Импорт функций и классов из файла models.py

# Загрузка переменных окружения из файла .env
load_dotenv()
# Получаем токен бота из переменной окружения
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Настройка логирования: форматирование сообщений и уровень логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Формат лог-сообщений
    level=logging.INFO  # Уровень логирования
)
logger = logging.getLogger(__name__)  # Создаем логгер для текущего модуля

# Загрузка модели YOLOv8
try:
    model = YOLO("yolov8n.pt")  # Загружаем модель из файла "yolov8n.pt"
    logger.info("Модель YOLOv8 успешно загружена.")  # Логируем успешную загрузку
except Exception as e:
    logger.error(f"Не удалось загрузить модель YOLOv8: {e}")  # Логируем ошибку загрузки
    raise e  # Прерываем выполнение, если модель не загрузилась

def predict_objects(image_bytes: bytes) -> (io.BytesIO, str):
    """
    Обнаруживает объекты на изображении с помощью YOLOv8.

    Функция:
      1. Открывает изображение из байтов.
      2. Преобразует его в numpy-массив.
      3. Запускает модель YOLOv8 для детекции объектов.
      4. Рисует прямоугольники вокруг обнаруженных объектов.
      5. Формирует текстовое описание обнаруженных объектов.

    Возвращает:
      - Буфер BytesIO с аннотированным изображением.
      - Строку с перечнем обнаруженных объектов и их оценками.
    """
    # Открываем изображение из байтов и конвертируем его в RGB
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Преобразуем изображение в numpy-массив для обработки моделью
    image_np = np.array(img)

    # Запускаем модель для обнаружения объектов
    results = model(image_np)

    # Рендерим результаты детекции: функция plot() возвращает numpy-массив с аннотациями
    annotated_frame = results[0].plot()
    # Преобразуем аннотированный массив в изображение
    annotated_image = Image.fromarray(annotated_frame)

    # Сохраняем аннотированное изображение в буфер памяти в формате JPEG
    output_buffer = io.BytesIO()
    annotated_image.save(output_buffer, format="JPEG")
    output_buffer.seek(0)  # Перемещаем указатель в начало буфера

    # Формируем текстовое описание обнаруженных объектов
    detection_text = "Обнаруженные объекты:\n"
    # Проверяем, есть ли обнаруженные объекты в результатах
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        # Перебираем каждый найденный объект
        for box in results[0].boxes:
            cls_index = int(box.cls.item())  # Получаем индекс класса объекта
            conf = float(box.conf.item())  # Получаем уверенность (confidence)
            # Если модель содержит имена классов, используем их, иначе преобразуем индекс в строку
            label = model.names[cls_index] if hasattr(model, "names") else str(cls_index)
            # Добавляем информацию об объекте в текстовое описание
            detection_text += f"- {label}: {conf:.2f}\n"
    else:
        # Если объектов не обнаружено, устанавливаем соответствующее сообщение
        detection_text = "Объекты не обнаружены."

    # Возвращаем буфер с аннотированным изображением и текстовое описание
    return output_buffer, detection_text

def get_main_menu():
    """Создает главное меню с кнопками /help и /history."""
    return ReplyKeyboardMarkup(
        [[KeyboardButton("/help"), KeyboardButton("/history")]],  # Кнопки для вызова справки и истории
        resize_keyboard=True,  # Автоматическое изменение размера клавиатуры
        one_time_keyboard=False  # Клавиатура остается доступной после выбора
    )

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает команду /start и отправляет приветственное сообщение."""
    await update.message.reply_text(
        "Привет! Пришли мне изображение, и я определю объекты на нем с помощью YOLOv8.",  # Приветственное сообщение
        reply_markup=get_main_menu()  # Отправляем главное меню с кнопками
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает команду /help и отправляет справку по командам бота."""
    help_text = (
        "/start - Запустить бота\n"
        "/help - Показать справку\n"
        "/history - Показать историю отправленных изображений\n"
        "Просто пришли изображение для обнаружения объектов."
    )
    await update.message.reply_text(help_text, reply_markup=get_main_menu())

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает входящие фотографии, выполняет детекцию объектов и отправляет результаты."""
    try:
        # Получаем фотографию наивысшего разрешения из списка присланных фото
        photo = update.message.photo[-1]
        # Получаем файл фотографии из Telegram
        photo_file = await photo.get_file()
        # Скачиваем файл в виде массива байтов
        file_bytes = await photo_file.download_as_bytearray()

        # Запускаем детекцию объектов по полученным байтам изображения
        output_buffer, detection_text = predict_objects(file_bytes)

        # Создаем директорию для сохранения аннотированных изображений, если ее нет
        os.makedirs("classified_images", exist_ok=True)
        # Формируем имя файла с использованием ID пользователя и текущего времени
        file_name = f"classified_{update.effective_user.id}_{int(time.time())}.jpg"
        # Определяем полный путь для сохранения файла
        file_path = os.path.join("classified_images", file_name)
        # Сохраняем аннотированное изображение на диск
        with open(file_path, "wb") as f:
            f.write(output_buffer.getvalue())

        # Сохраняем информацию о классификации в базе данных
        user_id = str(update.effective_user.id)  # Преобразуем ID пользователя в строку
        db = SessionLocal()  # Создаем сессию для работы с базой данных
        # Создаем новую запись с данными о классификации
        record = ClassifiedImage(
            user_id=user_id,
            image_path=file_path,
            classification_result=detection_text
        )
        db.add(record)  # Добавляем запись в сессию
        db.commit()  # Сохраняем изменения в базе данных
        db.close()  # Закрываем сессию

        # Отправляем пользователю аннотированное изображение с результатами детекции
        await update.message.reply_photo(
            photo=output_buffer,
            caption=detection_text,
            parse_mode=ParseMode.MARKDOWN  # Форматирование текста с помощью Markdown
        )
    except Exception as e:
        # Логируем ошибку, если что-то пошло не так при обработке фото
        logger.error(f"Ошибка при обработке фото: {e}")
        await update.message.reply_text("Произошла ошибка при обработке изображения.")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает текстовые сообщения, не являющиеся командами."""
    await update.message.reply_text("Пожалуйста, пришли изображение для обнаружения объектов или используй /help для справки.")

async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает команду /history: выводит историю отправленных изображений и результатов."""
    try:
        # Получаем ID пользователя, преобразованный в строку
        user_id = str(update.effective_user.id)
        # Создаем сессию для работы с базой данных
        db = SessionLocal()
        # Извлекаем все записи, связанные с данным пользователем
        records = db.query(ClassifiedImage).filter(ClassifiedImage.user_id == user_id).all()
        db.close()  # Закрываем сессию

        # Если записей нет, информируем пользователя
        if not records:
            await update.message.reply_text("У вас пока нет сохраненной истории.")
            return

        # Для каждой записи отправляем пользователю соответствующее изображение и информацию
        for record in records:
            with open(record.image_path, "rb") as photo_file:
                await update.message.reply_photo(
                    photo=photo_file,
                    caption=f"Результат:\n{record.classification_result}\nДата: {record.created_at}"
                )
    except Exception as e:
        # Логируем ошибку при получении истории
        logger.error(f"Ошибка при получении истории: {e}")
        await update.message.reply_text("Произошла ошибка при получении истории.")

def main():
    # Инициализируем базу данных (создаем таблицы, если их еще нет)
    init_db()  # Вызов функции для инициализации базы данных

    """Инициализирует Telegram-бота и регистрирует обработчики команд."""
    # Создаем объект приложения для бота с указанным токеном
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Регистрируем обработчик для команды /start
    application.add_handler(CommandHandler("start", start))
    # Регистрируем обработчик для команды /help
    application.add_handler(CommandHandler("help", help_command))
    # Регистрируем обработчик для команды /history
    application.add_handler(CommandHandler("history", history_command))
    # Регистрируем обработчик для входящих фотографий
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    # Регистрируем обработчик для текстовых сообщений, не являющихся командами
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # Запускаем бота в режиме polling (опроса сервера Telegram на наличие новых сообщений)
    application.run_polling()

# Если скрипт запущен напрямую, вызываем main()
if __name__ == "__main__":
    main()