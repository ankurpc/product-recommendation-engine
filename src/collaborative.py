"""
collaborative.py
-----------------
Collaborative Filtering recommender with two strategies:

  1. Item-Based CF   : cosine similarity between item rating-vectors (memory-based).
  2. Matrix Factorization : Truncated SVD on the user-item matrix (model-based),
                             which also lets us predict ratings for unseen pairs.

Both operate purely on the user-item ratings matrix — no product metadata needed,
which is what makes this "collaborative" (as opposed to content-based).
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD


class CollaborativeRecommender:
    def __init__(self, ratings_df: pd.DataFrame, n_factors: int = 20):
        self.ratings_df = ratings_df.copy()
        self.n_factors = n_factors

        self.user_item_matrix = None
        self.user_ids = None
        self.item_ids = None
        self._uid_to_idx = {}
        self._iid_to_idx = {}
        self._idx_to_iid = {}

        self.item_similarity = None
        self.svd_model = None
        self.user_factors = None
        self.item_factors = None
        self.predicted_matrix = None
        self.global_mean = None

    def _build_matrix(self):
        self.user_ids = sorted(self.ratings_df.user_id.unique())
        self.item_ids = sorted(self.ratings_df.product_id.unique())
        self._uid_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self._iid_to_idx = {p: i for i, p in enumerate(self.item_ids)}
        self._idx_to_iid = {i: p for p, i in self._iid_to_idx.items()}

        matrix = np.zeros((len(self.user_ids), len(self.item_ids)))
        for row in self.ratings_df.itertuples():
            matrix[self._uid_to_idx[row.user_id], self._iid_to_idx[row.product_id]] = row.rating
        self.user_item_matrix = matrix
        self.global_mean = self.ratings_df.rating.mean()

    # ------------------------------------------------------------------
    # Item-based memory CF
    # ------------------------------------------------------------------
    def fit_item_based(self):
        if self.user_item_matrix is None:
            self._build_matrix()
        self.item_similarity = cosine_similarity(self.user_item_matrix.T)
        return self

    def recommend_item_based(self, user_id: int, top_n: int = 10):
        if user_id not in self._uid_to_idx:
            return []
        uidx = self._uid_to_idx[user_id]
        user_ratings = self.user_item_matrix[uidx]
        rated_idxs = np.where(user_ratings > 0)[0]

        if len(rated_idxs) == 0:
            return []

        scores = np.zeros(len(self.item_ids))
        sim_sums = np.zeros(len(self.item_ids))
        for iidx in rated_idxs:
            sims = self.item_similarity[iidx]
            scores += sims * user_ratings[iidx]
            sim_sums += np.abs(sims)

        with np.errstate(divide="ignore", invalid="ignore"):
            pred_scores = np.where(sim_sums > 0, scores / sim_sums, 0)

        pred_scores[rated_idxs] = -np.inf  # exclude already-rated items
        top_idxs = np.argsort(pred_scores)[::-1][:top_n]
        return [(self._idx_to_iid[i], float(pred_scores[i])) for i in top_idxs if pred_scores[i] > -np.inf]

    # ------------------------------------------------------------------
    # Matrix Factorization (SVD)
    # ------------------------------------------------------------------
    def fit_svd(self):
        if self.user_item_matrix is None:
            self._build_matrix()

        # mean-center per user (only over rated items) to handle sparsity better
        matrix = self.user_item_matrix.copy()
        user_means = np.true_divide(matrix.sum(axis=1), (matrix != 0).sum(axis=1) + 1e-9)
        self._user_means = user_means

        centered = matrix.copy()
        for i in range(matrix.shape[0]):
            mask = matrix[i] > 0
            centered[i, mask] = matrix[i, mask] - user_means[i]

        n_components = min(self.n_factors, min(matrix.shape) - 1)
        n_components = max(n_components, 1)
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = self.svd_model.fit_transform(centered)
        self.item_factors = self.svd_model.components_

        predicted = self.user_factors @ self.item_factors
        for i in range(predicted.shape[0]):
            predicted[i] += user_means[i]
        self.predicted_matrix = np.clip(predicted, 1, 5)
        return self

    def recommend_svd(self, user_id: int, top_n: int = 10):
        if user_id not in self._uid_to_idx or self.predicted_matrix is None:
            return []
        uidx = self._uid_to_idx[user_id]
        preds = self.predicted_matrix[uidx].copy()
        rated_idxs = np.where(self.user_item_matrix[uidx] > 0)[0]
        preds[rated_idxs] = -np.inf
        top_idxs = np.argsort(preds)[::-1][:top_n]
        return [(self._idx_to_iid[i], float(preds[i])) for i in top_idxs if preds[i] > -np.inf]

    def predict_rating(self, user_id: int, product_id: int) -> float:
        """Predicted rating for a user-product pair using the SVD model."""
        if user_id not in self._uid_to_idx or product_id not in self._iid_to_idx:
            return float(self.global_mean) if self.global_mean else 3.0
        uidx = self._uid_to_idx[user_id]
        iidx = self._iid_to_idx[product_id]
        return float(self.predicted_matrix[uidx, iidx])
