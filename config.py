import os
from pathlib import Path

# Пути к данным
#DATA_DIR = 'RAW_DATA'
DATA_DIR = 'test_data'
CACHE_DIR = 'cache'
SUBMISSION_FILE = 'submission.csv'
LOG_FILE = 'recommendation_system.log'

# Параметры модели
KNN_PARAMS = {
    'n_neighbors': [30, 50, 70],
    'metric': ['cosine', 'euclidean'],
    'algorithm': ['auto', 'brute']
}

# Доля данных для обработки (1.0 - все данные, 0.1 - 10% и т.д.)
DATA_SAMPLE_RATIO = 0.1

# Валидация параметров
if not (0 < DATA_SAMPLE_RATIO <= 1):
    raise ValueError("DATA_SAMPLE_RATIO должен быть между 0 и 1")

# Создание директорий
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)