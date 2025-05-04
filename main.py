import sys
import argparse
from pathlib import Path
import logging
from utils.logging import setup_logger
import pandas as pd

# Инициализация логгера ДО всех импортов
logger = setup_logger()

# Остальные импорты
from data.loader import load_data
from data.preprocessor import preprocess_data, filter_features
from features.builder import build_features, build_interaction_matrix
from models.knn import optimize_knn, generate_recommendations
from evaluation.metrics import evaluate_recall
from config import SUBMISSION_FILE, DATA_SAMPLE_RATIO


def main():
    try:
        logger.info("=== Starting recommendation system ===")

        # 1. Загрузка данных
        data = load_data()
        if not data:
            raise ValueError("No data loaded")

        # 2. Предобработка
        logger.info("Preprocessing data...")
        preprocessed_data = preprocess_data(data)
        preprocessed_data = filter_features(preprocessed_data)

        # 3. Построение признаков
        logger.info("Build features...")
        feature_data = build_features(preprocessed_data)

        # 5. Построение матрицы взаимодействий
        logger.info("Building interaction matrix...")
        interaction_result = build_interaction_matrix(preprocessed_data, feature_data)
        # Извлекаем необходимые данные из результата
        interaction_matrix = interaction_result["user_item_matrix"]
        user_ids = interaction_result["user_ids"]
        item_ids = interaction_result["item_ids"]

        # 6. Оптимизация KNN
        logger.info("Optimizing KNN model...")
        knn_model = optimize_knn(feature_data, preprocessed_data['interactions'])

        # 7. Генерация и оценка рекомендаций
        logger.info("generate_recommendations...")
        recommendations = generate_recommendations(
            interaction_matrix, user_ids, item_ids, knn_model
        )

        # 8. Генерация финальных рекомендаций
        logger.info("Generating final recommendations...")
        final_recommendations = generate_recommendations(interaction_matrix, user_ids, item_ids, knn_model)

        # Преобразуем словарь рекомендаций в DataFrame
        recommendations_df = pd.DataFrame([
            {"cookie": user_id, "node": items[0] if items else None}  # node = первый рекомендованный item
            for user_id, items in final_recommendations.items()
        ])

        # Удаляем пользователей без рекомендаций (если такие есть)
        print(recommendations_df)
        recommendations_df = recommendations_df.dropna()
        print(recommendations_df)

        # 9. Обработка новых пользователей
        all_test_users = set(data['test_users']['cookie'].unique())
        final_recommendations_df = recommendations_df.rename(columns={'user_id': 'cookie'})

        print(type(final_recommendations_df))
        print(final_recommendations_df)
        print(len(final_recommendations_df))

        existing_users = set(final_recommendations_df['cookie'])
        new_users = all_test_users - existing_users

        if new_users:
            logger.info(f"Adding {len(new_users)} new users...")
            popular_item = preprocessed_data['interactions']['item'].value_counts().index[0]
            popular_node = feature_data['node_mapping'].get(popular_item, 0)

            new_users_recs = pd.DataFrame({
                'cookie': list(new_users),
                'node': [popular_node] * len(new_users)
            })

            recommendations_df = pd.concat([recommendations_df, new_users_recs])

        recommendations_df.to_csv('submission.csv', index=False)

        # Оценка метрики recall
        recall = evaluate_recall(
            recommendations=recommendations_df,
            interactions=preprocessed_data["interactions"],
            test_users=preprocessed_data["test_users_split"],
            node_mapping=feature_data["node_mapping"],
        )

        print(f"Recall: {recall:.4f}")
        logger.info(f"Recall@40: {recall:.4f}")

        # 6. Вывод рекомендаций (например, в лог или файл)
        logger.info("Recommendations generated.")
        for user_id, recommended_items in final_recommendations.items():
            logger.info(
                f"User {user_id}: Recommended items {recommended_items[:10]}")  # Выводим только первые 10 товаров

        # 9. Обработка новых пользователей
        all_test_users = set(data['test_users']['cookie'].unique())

        print(type(final_recommendations))
        print(final_recommendations)

        existing_users = set(final_recommendations['cookie'])
        new_users = all_test_users - existing_users

        if new_users:
            logger.info(f"Adding {len(new_users)} new users...")
            popular_item = preprocessed_data['interactions']['item'].value_counts().index[0]
            popular_node = feature_data['node_mapping'].get(popular_item, 0)

            new_users_recs = pd.DataFrame({
                'cookie': list(new_users),
                'node': [popular_node] * len(new_users)
            })
            final_recommendations = pd.concat([final_recommendations, new_users_recs])

        # 10. Сохранение результатов
        final_recommendations = final_recommendations.astype({
            'cookie': 'int64',
            'node': 'int64'
        })
        final_recommendations.to_csv('submission.csv', index=False)
        logger.info("=== Results saved to submission.csv ===")

    except Exception as e:
        logger.error(f"Critical error: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("=== System shutdown ===")

if __name__ == '__main__':
    main()