"""
recommend.py
------------
Simple command-line interface to get recommendations for a user.

Usage:
    python recommend.py --user_id 5 --top_n 10
    python recommend.py --user_id 5 --mode content
    python recommend.py --user_id 5 --mode collaborative
    python recommend.py --list_users
"""

import argparse
import pickle
import os
import pandas as pd

from src.hybrid import HybridRecommender


def load_or_train_model(products_df, ratings_df, model_path="models/hybrid_recommender.pkl"):
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    print("No saved model found — training a new one (this may take a few seconds)...")
    model = HybridRecommender(products_df, ratings_df)
    os.makedirs("models", exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return model


def main():
    parser = argparse.ArgumentParser(description="Product Recommendation Engine CLI")
    parser.add_argument("--user_id", type=int, help="User ID to generate recommendations for")
    parser.add_argument("--top_n", type=int, default=10, help="Number of recommendations")
    parser.add_argument("--mode", choices=["hybrid", "content", "collaborative"], default="hybrid",
                         help="Which recommendation strategy to use")
    parser.add_argument("--list_users", action="store_true", help="List sample user IDs and exit")
    args = parser.parse_args()

    products_df = pd.read_csv("data/products.csv")
    ratings_df = pd.read_csv("data/ratings.csv")
    users_df = pd.read_csv("data/users.csv")

    if args.list_users:
        print(users_df.head(20).to_string(index=False))
        return

    if args.user_id is None:
        parser.error("--user_id is required (or use --list_users to see sample IDs)")

    model = load_or_train_model(products_df, ratings_df)

    user_row = users_df[users_df.user_id == args.user_id]
    if not user_row.empty:
        print(f"\nUser: {user_row.user_name.values[0]} "
              f"(preferred category: {user_row.preferred_category.values[0]})")

    if args.mode == "hybrid":
        recs = model.recommend(args.user_id, top_n=args.top_n)
    elif args.mode == "content":
        recs = model.cb.recommend_for_user(args.user_id, ratings_df, top_n=args.top_n)
    else:  # collaborative
        raw = model.cf.recommend_svd(args.user_id, top_n=args.top_n)
        pids = [pid for pid, _ in raw]
        recs = products_df[products_df.product_id.isin(pids)].copy()
        score_map = dict(raw)
        recs["score"] = recs.product_id.map(score_map)
        recs = recs.sort_values("score", ascending=False)

    print(f"\nTop {args.top_n} recommendations ({args.mode}):\n")
    cols = [c for c in ["product_name", "category", "brand", "price", "score"] if c in recs.columns]
    print(recs[cols].to_string(index=False))


if __name__ == "__main__":
    main()
