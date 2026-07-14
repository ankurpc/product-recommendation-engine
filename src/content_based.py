"""
content_based.py
-----------------
Content-Based Filtering recommender.

Approach:
  1. Build a combined "content soup" for every product (category + brand + description).
  2. Vectorize with TF-IDF.
  3. Compute cosine similarity between all products.
  4. For a given user, build a profile from products they rated highly,
     and recommend the most similar *unseen* products.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self, products_df: pd.DataFrame):
        self.products_df = products_df.reset_index(drop=True).copy()
        self.products_df["content"] = (
            self.products_df["category"] + " " +
            self.products_df["brand"] + " " +
            self.products_df["product_name"] + " " +
            self.products_df["description"]
        )
        self._pid_to_idx = {pid: idx for idx, pid in enumerate(self.products_df["product_id"])}
        self._idx_to_pid = {idx: pid for pid, idx in self._pid_to_idx.items()}

        self.vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
        self.tfidf_matrix = None
        self.similarity_matrix = None

    def fit(self):
        """Build TF-IDF matrix and cosine similarity matrix over all products."""
        self.tfidf_matrix = self.vectorizer.fit_transform(self.products_df["content"])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        return self

    def similar_products(self, product_id: int, top_n: int = 10):
        """Return top-N products most similar to a given product."""
        if product_id not in self._pid_to_idx:
            return pd.DataFrame()
        idx = self._pid_to_idx[product_id]
        scores = list(enumerate(self.similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx][:top_n]

        result = self.products_df.iloc[[i for i, _ in scores]].copy()
        result["similarity_score"] = [s for _, s in scores]
        return result.reset_index(drop=True)

    def recommend_for_user(self, user_id: int, ratings_df: pd.DataFrame,
                            top_n: int = 10, rating_threshold: float = 4.0):
        """
        Build a user profile as the weighted-average TF-IDF vector of products
        they rated >= rating_threshold, then recommend the closest unseen products.
        """
        user_ratings = ratings_df[ratings_df.user_id == user_id]
        liked = user_ratings[user_ratings.rating >= rating_threshold]

        if liked.empty:
            # fall back to all rated products, weighted by rating
            liked = user_ratings
            if liked.empty:
                return pd.DataFrame()

        idxs = [self._pid_to_idx[pid] for pid in liked.product_id if pid in self._pid_to_idx]
        weights = liked.set_index("product_id").loc[
            [self._idx_to_pid[i] for i in idxs], "rating"
        ].values

        if len(idxs) == 0:
            return pd.DataFrame()

        user_profile = np.asarray(
            self.tfidf_matrix[idxs].multiply(weights.reshape(-1, 1)).mean(axis=0)
        )
        sims = cosine_similarity(user_profile, self.tfidf_matrix).flatten()

        seen_pids = set(user_ratings.product_id)
        candidate_idxs = [i for i in range(len(self.products_df))
                           if self._idx_to_pid[i] not in seen_pids]
        candidate_scores = sorted(
            ((i, sims[i]) for i in candidate_idxs),
            key=lambda x: x[1], reverse=True
        )[:top_n]

        result = self.products_df.iloc[[i for i, _ in candidate_scores]].copy()
        result["score"] = [s for _, s in candidate_scores]
        return result.drop(columns=["content"]).reset_index(drop=True)
