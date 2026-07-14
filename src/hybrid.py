"""
hybrid.py
---------
Hybrid Recommender: combines Content-Based and Collaborative Filtering scores
into a single weighted ranking. Also handles the classic "cold-start" problem —
new users with no ratings fall back to content-based/popularity recommendations.
"""

import pandas as pd
import numpy as np

from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender


class HybridRecommender:
    def __init__(self, products_df: pd.DataFrame, ratings_df: pd.DataFrame,
                 content_weight: float = 0.4, collab_weight: float = 0.6):
        self.products_df = products_df
        self.ratings_df = ratings_df
        self.content_weight = content_weight
        self.collab_weight = collab_weight

        self.cb = ContentBasedRecommender(products_df).fit()
        self.cf = CollaborativeRecommender(ratings_df).fit_svd()
        self.cf.fit_item_based()

        # popularity fallback for brand-new / cold-start users
        pop = (ratings_df.groupby("product_id")
               .agg(avg_rating=("rating", "mean"), n_ratings=("rating", "count"))
               .reset_index())
        pop["popularity_score"] = pop["avg_rating"] * np.log1p(pop["n_ratings"])
        self.popularity = pop.sort_values("popularity_score", ascending=False)

    def _normalize(self, series: pd.Series) -> pd.Series:
        if series.max() == series.min():
            return series * 0 + 0.5
        return (series - series.min()) / (series.max() - series.min())

    def recommend(self, user_id: int, top_n: int = 10):
        n_user_ratings = len(self.ratings_df[self.ratings_df.user_id == user_id])

        # Cold start: brand new user -> popularity-based recommendations
        if n_user_ratings == 0:
            top = self.popularity.head(top_n).merge(self.products_df, on="product_id")
            top["score"] = top["popularity_score"]
            top["source"] = "popularity (cold-start)"
            return top[["product_id", "product_name", "category", "brand", "price", "score", "source"]]

        # Content-based candidates
        cb_recs = self.cb.recommend_for_user(user_id, self.ratings_df, top_n=30)
        cb_scores = dict(zip(cb_recs.product_id, cb_recs.score)) if not cb_recs.empty else {}

        # Collaborative candidates (SVD predicted ratings)
        cf_recs = self.cf.recommend_svd(user_id, top_n=30)
        cf_scores = dict(cf_recs)

        all_pids = set(cb_scores) | set(cf_scores)
        if not all_pids:
            # fallback if both are empty (e.g. very few ratings)
            all_pids = set(self.products_df.product_id) - set(
                self.ratings_df[self.ratings_df.user_id == user_id].product_id)

        cb_series = pd.Series({pid: cb_scores.get(pid, 0.0) for pid in all_pids})
        cf_series = pd.Series({pid: cf_scores.get(pid, self.ratings_df.rating.mean()) for pid in all_pids})

        cb_norm = self._normalize(cb_series)
        cf_norm = self._normalize(cf_series)

        final_scores = (self.content_weight * cb_norm + self.collab_weight * cf_norm).sort_values(ascending=False)

        top_pids = final_scores.head(top_n).index.tolist()
        result = self.products_df[self.products_df.product_id.isin(top_pids)].copy()
        result["score"] = result["product_id"].map(final_scores)
        result = result.sort_values("score", ascending=False).reset_index(drop=True)
        result["source"] = "hybrid (content + collaborative)"
        return result[["product_id", "product_name", "category", "brand", "price", "score", "source"]]
