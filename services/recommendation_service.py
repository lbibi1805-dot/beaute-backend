"""
RecommendationService — product-level recommendations.

Two complementary signals:

1. Content-based "Similar items": cosine similarity on TF-IDF of product
   descriptions. Captures "products that look alike."

2. Author-based "Customers also bought" (collaborative filtering): item-item
   Jaccard similarity over a verified-buyer × product matrix. Captures
   "products commonly co-reviewed by the same buyers." Verified buyers only
   to reduce seeding/bot noise (mirrors notebook §15.2).

Both indices are built once at startup; lookups are O(1) dict access or
O(n_products) array operations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
from data_access.product_repository import ProductRepository


class RecommendationService:
    def __init__(self, product_repo: ProductRepository) -> None:
        self._repo = product_repo
        # content-based index
        self._product_ids: list[str] = []
        self._sim_matrix: np.ndarray | None = None
        self._id_to_idx: dict[str, int] = {}
        # collaborative-filtering index
        self._cf_product_ids: list[str] = []
        self._cf_id_to_idx: dict[str, int] = {}
        self._cf_co_occur: np.ndarray | None = None
        self._cf_counts: np.ndarray | None = None

        self._build_content_index()
        self._build_cf_index()

    # ── content-based (existing) ─────────────────────────────────────────────

    def _build_content_index(self) -> None:
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
        print(f"[RecommendationService] content sim matrix: {self._sim_matrix.shape}")

    def similar(self, product_id: str, top_n: int | None = None) -> list[dict]:
        """Content-based top-N similar products (excluding self)."""
        top_n = top_n or config.SIMILAR_PRODUCTS_TOP_N
        idx = self._id_to_idx.get(product_id)
        if idx is None or self._sim_matrix is None:
            return []
        scores = self._sim_matrix[idx].copy()
        scores[idx] = -1.0
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

    # ── collaborative filtering (NEW) ────────────────────────────────────────

    def _build_cf_index(self) -> None:
        df = pd.read_csv(config.RAW_CSV_PATH, low_memory=False)
        df = df.dropna(subset=["author", "product_id", "is_a_buyer"]).copy()
        df["is_a_buyer"] = df["is_a_buyer"].astype(str).str.upper().eq("TRUE")
        verified = df[df["is_a_buyer"]].drop_duplicates(subset=["author", "product_id"])

        if verified.empty:
            print("[RecommendationService] no verified-buyer rows — CF disabled")
            return

        verified = verified.copy()
        verified["product_id"] = verified["product_id"].astype(str)

        authors  = verified["author"].unique()
        products = verified["product_id"].unique()
        author_idx  = {a: i for i, a in enumerate(authors)}
        product_idx = {p: i for i, p in enumerate(products)}

        rows = verified["author"].map(author_idx).values
        cols = verified["product_id"].map(product_idx).values
        data = np.ones(len(verified), dtype=np.float32)

        ap = sparse.coo_matrix(
            (data, (rows, cols)),
            shape=(len(authors), len(products)),
        ).tocsr()

        self._cf_product_ids = products.tolist()
        self._cf_id_to_idx   = {p: i for i, p in enumerate(self._cf_product_ids)}
        self._cf_counts      = np.asarray(ap.sum(axis=0)).ravel()
        co = (ap.T @ ap).toarray()
        np.fill_diagonal(co, 0)
        self._cf_co_occur = co
        print(f"[RecommendationService] CF matrix: {co.shape}  "
              f"products={len(products)}  authors={len(authors)}")

    def cooccurring_products(self, product_id: str, top_k: int | None = None) -> list[dict]:
        """
        Author-based top-K co-reviewed products (Jaccard similarity, verified
        buyers only). Returns full product dicts enriched with cf_jaccard
        and cf_shared_reviewers fields for the UI.
        """
        top_k = top_k or config.COOCCURRING_PRODUCTS_TOP_N
        if self._cf_co_occur is None:
            return []
        idx = self._cf_id_to_idx.get(str(product_id))
        if idx is None:
            return []

        co = self._cf_co_occur[idx]
        counts = self._cf_counts
        union = counts[idx] + counts - co
        jacc = np.where(union > 0, co / union, 0.0)
        jacc[idx] = 0.0

        top = np.argsort(-jacc)[:top_k]
        results = []
        for i in top:
            if jacc[i] <= 0:
                break
            pid = self._cf_product_ids[i]
            product = self._repo.get_by_id(pid)
            if product is None:
                continue
            enriched = {**product, "cf_jaccard": round(float(jacc[i]), 4),
                        "cf_shared_reviewers": int(co[i])}
            results.append(enriched)
        return results
