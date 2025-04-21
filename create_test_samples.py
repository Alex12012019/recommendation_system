# create_test_samples.py
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Конфигурация
RAW_DATA_DIR = "RAW_DATA"  # Папка с исходными данными
TEST_DATA_DIR = "test_data"  # Куда сохранять уменьшенные данные
SAMPLE_RATIO = 0.01  # 1% записей (можно настроить)
RANDOM_SEED = 42  # Для воспроизводимости


def create_test_samples():
    """Создает тестовые версии файлов из RAW_DATA."""
    os.makedirs(TEST_DATA_DIR, exist_ok=True)

    for file in Path(RAW_DATA_DIR).iterdir():
        if file.suffix in (".pq", ".parquet", ".csv"):
            print(f"Обработка {file.name}...")

            # Чтение файла
            if file.suffix == ".csv":
                df = pd.read_csv(file)
            else:
                df = pd.read_parquet(file)

            # Выбор подмножества записей
            sample_size = max(1, int(len(df) * SAMPLE_RATIO))  # Минимум 1 строка
            df_sample = df.sample(sample_size, random_state=RANDOM_SEED)

            # Сохранение
            output_path = Path(TEST_DATA_DIR) / file.name
            if file.suffix == ".csv":
                df_sample.to_csv(output_path, index=False)
            else:
                df_sample.to_parquet(output_path, engine="pyarrow", compression="snappy")

            print(f"Сохранено {len(df_sample)} строк в {output_path}")


if __name__ == "__main__":
    create_test_samples()