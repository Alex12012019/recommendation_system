import numpy as np
import pandas as pd

def evaluate_recall(recommendations, interactions, test_users, node_mapping):
    """Оценка Recall@40"""
    test_users_set = set(test_users)
    user_interactions = interactions.groupby('cookie')['item'].apply(set).to_dict()

    scores = []
    for _, row in recommendations.iterrows():
        user_id = row['cookie']
        if user_id not in test_users_set:
            continue

        rec_node = row['node']
        true_items = user_interactions.get(user_id, set())

        if true_items:
            true_nodes = {node_mapping[item] for item in true_items if item in node_mapping}
            recall = 1 if rec_node in true_nodes else 0
            scores.append(recall)

    return np.mean(scores) if scores else 0