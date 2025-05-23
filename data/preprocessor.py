import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from utils.cache import cache_step
from config import DATA_SAMPLE_RATIO
import logging
import ast

logger = logging.getLogger(__name__)


def validate_clean_params(params):
    """Валидация и очистка параметров"""
    try:
        if isinstance(params, str):
            params = ast.literal_eval(params)  # Безопасный eval
        return params if isinstance(params, list) else []
    except (ValueError, SyntaxError):
        return []


@cache_step('preprocess_data')
def preprocess_data(data):

    """
    Предобрабатывает входные данные для системы рекомендаций.

    Ожидаемый формат входных данных:
    - data: pandas.DataFrame с обязательными столбцами:
        - 'user_id': уникальный идентификатор пользователя (int или str)
        - 'item_id': уникальный идентификатор объекта рекомендации (int или str)
        - 'interaction_value': числовое значение взаимодействия (float или int)
        - Дополнительные столбцы с признаками объектов (например, 'feature_1', 'feature_2', ...)

    Возвращает:
    - preprocessed_data: pandas.DataFrame с очищенными и подготовленными данными для дальнейшей обработки.
    """

    """Полная предобработка данных с проверкой качества"""
    try:
        # 1. Объединение данных о товарах
        logger.info("Merging item data...")
        items = data['cat_features'].merge(
            data['text_features'],
            on='item',
            how='left',
            validate='one_to_one'
        )

        # 2. Проверка и очистка clean_params
        logger.info("Validating clean_params...")
        items['clean_params'] = items['clean_params'].apply(validate_clean_params)

        empty_params = items['clean_params'].apply(len) == 0
        if empty_params.any():
            empty_count = empty_params.sum()
            logger.warning(f"Found {empty_count} items with empty clean_params")
            items.loc[empty_params, 'clean_params'] = [{'attr': 'default', 'value': 'empty'}] * empty_count

        # 3. Фильтрация контактных событий
        logger.info("Filtering contact events...")
        contact_events = data['events'][data['events']['is_contact'] == 1]['event'].unique()
        clickstream = data['clickstream'][data['clickstream']['event'].isin(contact_events)].copy()

        # 4. Создание матрицы взаимодействий
        logger.info("Creating interactions matrix...")
        interactions = clickstream.groupby(['cookie', 'item']).size().reset_index(name='weight')

        # 5. Разделение пользователей с учетом выборки
        logger.info("Splitting test users...")
        test_users = data['test_users']['cookie'].unique()

        if DATA_SAMPLE_RATIO < 1.0:
            sample_size = max(1, int(len(test_users) * DATA_SAMPLE_RATIO))
            # Исправлено: используем RandomState для воспроизводимости
            rng = np.random.RandomState(42)
            test_users = rng.choice(
                test_users,
                size=sample_size,
                replace=False
            )
            logger.info(f"Sampled {len(test_users)} test users ({DATA_SAMPLE_RATIO * 100}%)")

        # Разделение на train/test
        train_users, test_users_split = train_test_split(
            test_users,
            test_size=0.2,
            random_state=42,
            shuffle=True
        )

        return {
            'items': items,
            'interactions': interactions,
            'train_users': train_users,
            'test_users_split': test_users_split
        }

    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise


@cache_step('filter_features')
def filter_features(preprocessed_data):
    """
    Очищает и фильтрует признаки объектов (например, товаров).

    Аргументы:
    - df: pandas.DataFrame, содержащий признаки объектов.
        Обязательные столбцы:
            - 'item_id': уникальный идентификатор объекта.
        Остальные столбцы — категориальные или текстовые признаки объектов.
    - min_df: int (по умолчанию = 2), минимальное количество вхождений значения признака,
        чтобы оно было оставлено (редкие значения будут удалены).

    Возвращает:
    - pandas.DataFrame с очищенными и отфильтрованными признаками объектов.
    """

    """Фильтрация и уменьшение размерности признаков"""
    items = preprocessed_data['items'].copy()

    # 1. Фильтрация clean_params
    logger.info("Filtering clean_params...")
    param_stats = {}
    for params in items['clean_params']:
        try:
            param_list = eval(params) if isinstance(params, str) else params
            for p in param_list:
                if isinstance(p, dict):
                    key = (str(p.get('attr', '')), str(p.get('value', '')))
                    param_stats[key] = param_stats.get(key, 0) + 1
        except:
            continue

    top_pairs = sorted(param_stats.items(), key=lambda x: -x[1])[:1000]
    keep_pairs = {pair for pair, _ in top_pairs}

    filtered_params = []
    for params in items['clean_params']:
        try:
            param_list = eval(params) if isinstance(params, str) else params
            filtered = [
                p for p in param_list
                if isinstance(p, dict) and
                   (str(p.get('attr', '')), str(p.get('value', ''))) in keep_pairs
            ]
            filtered_params.append(str(filtered) if filtered else '[]')
        except:
            filtered_params.append('[]')

    items['clean_params'] = filtered_params

    # 2. Уменьшение размерности title_projection
    logger.info("Reducing title dimensions...")
    titles = items['title_projection'].values
    titles_matrix = np.vstack(titles)
    pca = PCA(n_components=16)
    reduced_titles = pca.fit_transform(titles_matrix)
    items['title_projection'] = [reduced_titles[i] for i in range(len(titles))]

    preprocessed_data['items'] = items
    return preprocessed_data

