import pandas as pd
import numpy as np
from pathlib import Path
from utils.cache import cache_step
from config import DATA_DIR, DATA_SAMPLE_RATIO


@cache_step('load_data')
def load_data():
    """Загрузка данных с обработкой ошибок"""
    try:
        import pyarrow  # Проверка наличия библиотеки
    except ImportError:
        raise ImportError(
            "Для работы с Parquet файлами требуется pyarrow. "
            "Установите: pip install pyarrow"
        )

    required_files = {
        'text_features': 'text_features.pq',
        'cat_features': 'cat_features.pq',
        'clickstream': 'clickstream.pq',
        'events': 'events.pq',
        'test_users': 'test_users.pq',
        'submit_example': 'submit_example.csv'
    }

    data = {}
    for name, filename in required_files.items():
        filepath = Path(DATA_DIR) / filename
        if not filepath.exists():
            raise FileNotFoundError(f'Файл {filepath} не найден')

        try:
            if filename.endswith('.pq'):
                df = pd.read_parquet(filepath, engine='pyarrow')  # Явно указываем движок
            else:
                df = pd.read_csv(filepath)

            if DATA_SAMPLE_RATIO < 1.0 and name in ['clickstream', 'events', 'test_users']:
                df = df.sample(frac=DATA_SAMPLE_RATIO, random_state=42)

            data[name] = df

        except Exception as e:
            raise IOError(f'Ошибка загрузки {filename}: {str(e)}')

    return data