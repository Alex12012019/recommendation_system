import sys
import argparse
from pathlib import Path
import logging
from utils.logging import setup_logger

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
        preprocessed_data = preprocess_data(data)
        preprocessed_data = filter_features(preprocessed_data)

        # 3. Построение признаков
        feature_data = build_features(preprocessed_data)

        # 5. Построение матрицы взаимодействий
        interaction_result = build_interaction_matrix(preprocessed_data)
        interaction_data = interaction_result["interaction_matrix"]
        feature_data = interaction_result["feature_matrix"]

        # 6. Оптимизация KNN
        knn_model = optimize_knn(feature_data, preprocessed_data['interactions'])

        # 7. Генерация и оценка рекомендаций
        recommendations = generate_recommendations(interaction_data, feature_data, knn_model)
        recall = evaluate_recall(
            recommendations,
            preprocessed_data['interactions'],
            preprocessed_data['test_users_split'],
            feature_data['node_mapping']
        )

        # 8. Генерация финальных рекомендаций
        final_recommendations = generate_recommendations(interaction_data, feature_data, knn_model)

        # 9. Обработка новых пользователей
        all_test_users = set(data['test_users']['cookie'].unique())

        #print("Columns:", final_recommendations.columns)
        #print("Head:\n", final_recommendations.head())
        print(type(final_recommendations))
        print(final_recommendations)
        print(final_recommendations.columns)

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