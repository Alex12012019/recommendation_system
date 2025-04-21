import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from utils.cache import cache_step
from config import KNN_PARAMS


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


def generate_recommendations(interaction_data, feature_data, knn_model):
    """Генерация рекомендаций"""
    user_item_matrix = interaction_data['user_item_matrix']
    user_ids = interaction_data['user_ids']
    item_features = feature_data['item_features']
    item_ids = feature_data['item_ids']
    node_mapping = feature_data['node_mapping']
    item_id_to_idx = feature_data['item_id_to_idx']
    item_idx_to_id = {v: k for k, v in item_id_to_idx.items()}

    item_popularity = interaction_data['interactions']['item'].value_counts().to_dict()
    recommendations = []

    for user_idx, user_id in enumerate(user_ids):
        user_row = user_item_matrix[user_idx]
        interacted_items = [
            item_idx_to_id[col]
            for col in user_row.indices
            if col in item_idx_to_id
        ]

        if len(interacted_items) == 0:
            popular_item = max(item_popularity.items(), key=lambda x: x[1])[0]
            rec_node = node_mapping.get(popular_item, 0)
        else:
            first_item = interacted_items[0]
            if first_item in item_id_to_idx:
                idx = item_id_to_idx[first_item]
                query = item_features[idx].reshape(1, -1)
                _, indices = knn_model.kneighbors(query, n_neighbors=1)
                recommended_item = item_ids[indices[0][0]]
                rec_node = node_mapping.get(recommended_item, 0)
            else:
                rec_node = 0

        recommendations.append({
            'cookie': int(user_id),
            'node': int(rec_node)
        })

    return pd.DataFrame(recommendations)