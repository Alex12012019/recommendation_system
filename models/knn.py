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

def generate_recommendations(interaction_data, feature_data, knn_model, top_k=40):
    """
    Генерирует рекомендации для каждого пользователя на основе KNN модели.

    interaction_data: pd.DataFrame, user-item матрица взаимодействий (например, cookie x node)
    feature_data: pd.DataFrame, признаки объявлений или товаров
    knn_model: обученная модель KNN
    top_k: количество рекомендаций на пользователя
    """

    recommendations = []

    if isinstance(interaction_data, pd.DataFrame) and interaction_data.empty:
        print("Interaction data is empty.")
        return pd.DataFrame(columns=['cookie', 'recommended_node', 'score'])

    # Пройтись по каждому пользователю (cookie)
    for user_idx, user_id in enumerate(interaction_data.index):
        user_vector = interaction_data.iloc[user_idx].values.reshape(1, -1)

        try:
            distances, indices = knn_model.kneighbors(user_vector, n_neighbors=top_k)
        except Exception as e:
            print(f"Ошибка при генерации для пользователя {user_id}: {e}")
            continue

        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            node_id = interaction_data.columns[idx]
            recommendations.append({
                'cookie': user_id,
                'recommended_node': node_id,
                'score': 1 / (1 + dist)  # Чем меньше расстояние — тем выше score
            })

    if not recommendations:
        print("Нет сгенерированных рекомендаций.")
        return pd.DataFrame(columns=['cookie', 'recommended_node', 'score'])

    return pd.DataFrame(recommendations)
