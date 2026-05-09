"""
RecommendationService — Task 3 similar product recommendations.

Similarity measure: cosine similarity on TF-IDF vectors of product descriptions.
Matrix is computed once at startup; lookups are O(1) dict access.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
from data_access.product_repository import ProductRepository


class RecommendationService:
    def __init__(self, product_repo: ProductRepository) -> None:
        self._repo = product_repo
        self._product_ids: list[str] = []
        self._sim_matrix: np.ndarray | None = None
        self._id_to_idx: dict[str, int] = {}
        self._build_index()

    # ── internal ──────────────────────────────────────────────────────────────

    def _build_index(self) -> None:
        pairs = self._repo.get_all_descriptions()
        if not pairs:
            return
        self._product_ids = [pid for pid, _ in pairs]
        self._id_to_idx = {pid: i for i, pid in enumerate(self._product_ids)}
        texts = [text for _, text in pairs]

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=5000,
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        self._sim_matrix = cosine_similarity(tfidf_matrix, dense_output=True)
        print(f"[RecommendationService] built {self._sim_matrix.shape} similarity matrix")

    # ── public API ────────────────────────────────────────────────────────────

    def similar(self, product_id: str, top_n: int | None = None) -> list[dict]:
        """Return up to top_n products most similar to product_id (excluding itself)."""
        top_n = top_n or config.SIMILAR_PRODUCTS_TOP_N

        idx = self._id_to_idx.get(product_id)
        if idx is None or self._sim_matrix is None:
            return []

        scores = self._sim_matrix[idx].copy()
        scores[idx] = -1.0  # exclude self

        top_indices = np.argsort(scores)[::-1][:top_n]
        results = []
        for i in top_indices:
            if scores[i] <= 0:
                break
            pid = self._product_ids[i]
            product = self._repo.get_by_id(pid)
            if product:
                results.append(product)
        return results
