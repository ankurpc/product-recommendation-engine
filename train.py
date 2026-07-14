"""
train.py
--------
Loads the dataset, builds all recommender models, runs a quick evaluation,
and saves the fitted HybridRecommender to disk (pickle) for reuse by the app.

Run:
    python train.py
"""

import pickle
import pandas as pd

from src.hybrid import HybridRecommender
from src.evaluate import evaluate_svd_rmse, evaluate_hybrid


def main():
    print("Loading data...")
    products_df = pd.read_csv("data/products.csv")
    ratings_df = pd.read_csv("data/ratings.csv")
    print(f"  {len(products_df)} products, {len(ratings_df)} ratings, "
          f"{ratings_df.user_id.nunique()} users")

    print("\nTraining Hybrid Recommender (Content-Based + Collaborative Filtering)...")
    recommender = HybridRecommender(products_df, ratings_df)

    print("\nSaving model to models/hybrid_recommender.pkl ...")
    with open("models/hybrid_recommender.pkl", "wb") as f:
        pickle.dump(recommender, f)

    print("\nRunning evaluation...")
    rmse_result = evaluate_svd_rmse(ratings_df)
    print(f"  SVD RMSE: {rmse_result['rmse']:.4f} "
          f"(evaluated on {rmse_result['n_test_points_evaluated']} held-out ratings)")

    rank_result = evaluate_hybrid(products_df, ratings_df, k=10)
    print(f"  Precision@10: {rank_result['precision_at_k']:.4f}")
    print(f"  Recall@10:    {rank_result['recall_at_k']:.4f}")
    print(f"  (evaluated across {rank_result['evaluated_users']} users)")

    print("\nDone. Model saved and ready to use in app.py / recommend.py")


if __name__ == "__main__":
    main()
