"""
evaluate.py
-----------
Evaluates recommender quality using a train/test split:
  - Precision@K / Recall@K  for ranking quality (top-N recommendations)
  - RMSE                    for the SVD rating-prediction accuracy
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.hybrid import HybridRecommender
from src.collaborative import CollaborativeRecommender


def precision_recall_at_k(recommended_ids, relevant_ids, k=10):
    recommended_k = recommended_ids[:k]
    if not recommended_k:
        return 0.0, 0.0
    hits = len(set(recommended_k) & set(relevant_ids))
    precision = hits / len(recommended_k)
    recall = hits / len(relevant_ids) if relevant_ids else 0.0
    return precision, recall


def evaluate_hybrid(products_df: pd.DataFrame, ratings_df: pd.DataFrame,
                     k: int = 10, relevance_threshold: float = 4.0, test_size: float = 0.2):
    train_df, test_df = train_test_split(ratings_df, test_size=test_size, random_state=42,
                                          stratify=None)

    recommender = HybridRecommender(products_df, train_df)

    relevant_by_user = (
        test_df[test_df.rating >= relevance_threshold]
        .groupby("user_id")["product_id"].apply(list).to_dict()
    )

    precisions, recalls = [], []
    evaluated_users = 0
    for user_id, relevant_items in relevant_by_user.items():
        if user_id not in train_df.user_id.values:
            continue  # skip users unseen in training (true cold-start, handled separately)
        recs = recommender.recommend(user_id, top_n=k)
        recommended_ids = recs.product_id.tolist()
        p, r = precision_recall_at_k(recommended_ids, relevant_items, k=k)
        precisions.append(p)
        recalls.append(r)
        evaluated_users += 1

    avg_precision = np.mean(precisions) if precisions else 0.0
    avg_recall = np.mean(recalls) if recalls else 0.0
    return {
        "precision_at_k": avg_precision,
        "recall_at_k": avg_recall,
        "k": k,
        "evaluated_users": evaluated_users,
    }


def evaluate_svd_rmse(ratings_df: pd.DataFrame, test_size: float = 0.2):
    train_df, test_df = train_test_split(ratings_df, test_size=test_size, random_state=42)
    cf = CollaborativeRecommender(train_df).fit_svd()

    errors = []
    for row in test_df.itertuples():
        if row.user_id in cf._uid_to_idx and row.product_id in cf._iid_to_idx:
            pred = cf.predict_rating(row.user_id, row.product_id)
            errors.append((pred - row.rating) ** 2)

    rmse = np.sqrt(np.mean(errors)) if errors else float("nan")
    return {"rmse": rmse, "n_test_points_evaluated": len(errors), "n_test_points_total": len(test_df)}


if __name__ == "__main__":
    products_df = pd.read_csv("data/products.csv")
    ratings_df = pd.read_csv("data/ratings.csv")

    print("Evaluating SVD rating-prediction accuracy (RMSE)...")
    rmse_result = evaluate_svd_rmse(ratings_df)
    print(rmse_result)

    print("\nEvaluating Hybrid recommender ranking quality (Precision@K / Recall@K)...")
    rank_result = evaluate_hybrid(products_df, ratings_df, k=10)
    print(rank_result)
