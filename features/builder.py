import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import FeatureHasher
from sklearn.decomposition import PCA
from utils.cache import cache_step
import logging
import ast
from scipy.sparse import csr_matrix
import numpy as np
from tqdm import tqdm


logger = logging.getLogger(__name__)


@cache_step('filter_features')
def filter_features(preprocessed_data):
    """Полная фильтрация и уменьшение размерности признаков"""
    logger.info("Starting feature filtering...")
    items = preprocessed_data['items'].copy()

    # 1. Фильтрация clean_params
    param_stats = {}
    for params in items['clean_params']:
        try:
            param_list = ast.literal_eval(params) if isinstance(params, str) else params
            for p in param_list:
                if isinstance(p, dict):
                    key = (str(p.get('attr', '')), str(p.get('value', '')))
                    param_stats[key] = param_stats.get(key, 0) + 1
        except:
            continue

    # Выбор топ-1000 самых частых пар
    top_pairs = sorted(param_stats.items(), key=lambda x: -x[1])[:1000]
    keep_pairs = {pair for pair, _ in top_pairs}

    # Фильтрация параметров
    filtered_params = []
    for params in items['clean_params']:
        try:
            param_list = ast.literal_eval(params) if isinstance(params, str) else params
            filtered = [
                p for p in param_list
                if isinstance(p, dict) and
                   (str(p.get('attr', '')), str(p.get('value', ''))) in keep_pairs
            ]
            filtered_params.append(str(filtered) if filtered else '[{"attr":"none","value":"none"}]')
        except:
            filtered_params.append('[{"attr":"none","value":"none"}]')

    items['clean_params'] = filtered_params

    # 2. Уменьшение размерности title_projection
    titles = items['title_projection'].values
    titles_matrix = np.vstack(titles)
    pca = PCA(n_components=16)
    reduced_titles = pca.fit_transform(titles_matrix)
    items['title_projection'] = [reduced_titles[i] for i in range(len(titles))]

    preprocessed_data['items'] = items
    return preprocessed_data


@cache_step('build_features')
def build_features(preprocessed_data):
    """Полное построение матрицы признаков"""
    try:
        logger.info("Starting feature building...")
        items = preprocessed_data['items']

        # 1. Обработка параметров
        logger.info("Processing item parameters...")
        hasher = FeatureHasher(n_features=2 ** 12, input_type='pair', dtype=np.float32)

        param_pairs = []
        for params in items['clean_params']:
            try:
                param_list = ast.literal_eval(params) if isinstance(params, str) else params
                pairs = [
                    (str(p.get('attr', '')), str(p.get('value', '')))
                    for p in param_list if isinstance(p, dict)
                ]
                param_pairs.append(pairs if pairs else [('none', 'none')])
            except:
                param_pairs.append([('none', 'none')])
                logger.warning("Failed to parse params, using default")

        params_features = hasher.transform(param_pairs)

        # 2. Категориальные признаки
        logger.info("Processing categorical features...")
        cat_features = csr_matrix(items[['location', 'category']].astype(np.float32).values)

        # 3. Текстовые признаки
        logger.info("Processing text features...")
        title_features = csr_matrix(np.vstack(items['title_projection'].values))

        # 4. Комбинирование признаков
        logger.info("Combining all features...")
        item_features = hstack([cat_features, params_features, title_features]).tocsr()

        logger.info(f"Built feature matrix: {item_features.shape}")

        return {
            'item_features': item_features,
            'item_ids': items['item'].values,
            'node_mapping': items.set_index('item')['node'].to_dict(),
            'item_meta': items.set_index('item')[['location', 'category', 'node']],
            'item_id_to_idx': {item_id: idx for idx, item_id in enumerate(items['item'].values)},
            'item_idx_to_id': {idx: item_id for idx, item_id in enumerate(items['item'].values)}
        }

    except Exception as e:
        logger.error(f"Feature building failed: {str(e)}", exc_info=True)
        raise


@cache_step('build_interaction_matrix')
def build_interaction_matrix(preprocessed_data, feature_data):
    """Построение матрицы взаимодействий пользователь-товар."""
    try:
        logger.info("Building user-item interaction matrix...")

        interactions = preprocessed_data['interactions']
        train_users = set(preprocessed_data['train_users'])

        if interactions.empty or not train_users:
            logger.warning("Empty interactions or train_users.")
            return None

        # Фильтрация взаимодействий только от обучающих пользователей
        train_interactions = interactions[interactions['cookie'].isin(train_users)]

        # Маппинг пользователей и товаров в индексы
        user_to_idx = {user: idx for idx, user in enumerate(train_interactions['cookie'].unique())}
        item_to_idx = feature_data['item_id_to_idx']
        item_ids = feature_data['item_ids']  # предполагается, что порядок соответствует item_to_idx

        # Подготовка списков для разреженной матрицы
        rows, cols, data = [], [], []

        for row in tqdm(train_interactions.itertuples(index=False),
                        total=len(train_interactions),
                        desc="Processing interactions"):
            if row.item in item_to_idx:
                rows.append(user_to_idx[row.cookie])
                cols.append(item_to_idx[row.item])
                data.append(row.weight)

        # Создание разреженной матрицы в формате CSR
        user_item_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(user_to_idx), len(item_to_idx)),
            dtype=np.float32
        )

        logger.info(f"Built interaction matrix with shape {user_item_matrix.shape}")
        logger.info(f"Number of training interactions: {len(train_interactions)}")
        logger.info(f"Unique users: {len(user_to_idx)}, Unique items: {len(item_to_idx)}")

        return {
            'user_item_matrix': user_item_matrix,
            'user_ids': np.array(list(user_to_idx.keys())),
            'item_ids': item_ids,
            'interactions': interactions,
            'user_to_idx': user_to_idx,
            'item_to_idx': item_to_idx
        }

    except Exception as e:
        logger.error(f"Failed to build interaction matrix: {str(e)}", exc_info=True)
        raise
