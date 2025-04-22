import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from utils.cache import cache_step
from config import KNN_PARAMS
import logging

logger = logging.getLogger(__name__)


def custom_recall_at_k(y_true, y_pred, k=40):
    """Кастомная метрика Recall@k"""
    y_pred = y_pred[:k]
    intersection = len(set(y_true) & set(y_pred))
    return intersection / len(y_true) if len(y_true) > 0 else 0.0


@cache_step('optimize_knn')
def optimize_knn(feature_data, interactions):
    """Оптимизация гиперпараметров KNN"""
    sample_size = min(5000, feature_data['item_features'].shape[0])
    indices = np.random.choice(feature_data['item_features'].shape[0], sample_size, replace=False)
    X = feature_data['item_features'][indices]

    item_ids = feature_data['item_ids'][indices]
    y = []
    for item_id in item_ids:
        interacted_users = interactions[interactions['item'] == item_id]['cookie'].unique()
        y.append(interacted_users)

    knn = NearestNeighbors(n_jobs=-1)
    scorer = make_scorer(custom_recall_at_k, needs_proba=False, k=40)

    gs = GridSearchCV(
        estimator=knn,
        param_grid=KNN_PARAMS,
        scoring=scorer,
        cv=3,
        verbose=2,
        n_jobs=1
    )

    gs.fit(X, y)
    return gs.best_estimator_


import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def generate_recommendations(interaction_matrix, user_ids, item_ids, knn_model):
    """
    Генерирует рекомендации для каждого пользователя на основе KNN модели.

    interaction_data: pd.DataFrame, user-item матрица взаимодействий (например, cookie x node)
    feature_data: pd.DataFrame, признаки объявлений или товаров
    knn_model: обученная модель KNN
    top_k: количество рекомендаций на пользователя
    """

    recommendations = {}
    for user_idx, user_id in enumerate(user_ids):
        distances, indices = knn_model.kneighbors(
            interaction_matrix[user_idx], n_neighbors=40
        )
        recommended_items = [item_ids[i] for i in indices.flatten()]
        recommendations[user_id] = recommended_items
    return recommendations