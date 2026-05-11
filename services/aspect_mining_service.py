"""
AspectMiningService — extracts top complaint terms from negative reviews.

Mirrors the methodology in data/task2_3.ipynb §15.4:
  1. Concatenate review_title + review_text, lowercase.
  2. Fit a TF-IDF (unigrams + bigrams) over the catalogue.
  3. Compute trust-weighted mean TF-IDF for negative (rating<=2) and
     positive (rating>=4) cohorts.
  4. Discriminative score = neg_mean - pos_mean. High score => more
     characteristic of negative reviews.

Cached in-memory at service init. Exposes catalogue-wide and per-brand and
per-product top-N complaint terms.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import config


class AspectMiningService:
    NEG_RATING_THRESHOLD = 2   # ratings <= this count as "negative"

    def __init__(self) -> None:
        self._vec = None
        self._feat_names: list[str] = []
        self._pos_mean: np.ndarray | None = None
        self._brand_scores: dict[str, np.ndarray] = {}
        self._product_scores: dict[str, np.ndarray] = {}
        self._global_score: np.ndarray | None = None
        self._build_index()

    # ── internal ──────────────────────────────────────────────────────────────

    def _build_index(self) -> None:
        df = pd.read_csv(config.RAW_CSV_PATH, low_memory=False)
        df = df.dropna(subset=["review_rating", "is_a_buyer"]).copy()
        df["review_rating"] = df["review_rating"].astype(float)
        df["is_a_buyer"]    = df["is_a_buyer"].astype(str).str.upper().eq("TRUE")
        df["weight"]        = np.where(df["is_a_buyer"], 1.0, config.NON_BUYER_TRUST_WEIGHT)
        df["text"] = (
            df["review_title"].fillna("").astype(str) + " "
            + df["review_text"].fillna("").astype(str)
        ).str.lower()

        neg_mask = df["review_rating"] <= self.NEG_RATING_THRESHOLD
        pos_mask = df["review_rating"] >= config.RECOMMEND_RATING_THRESHOLD

        neg = df[neg_mask]
        pos = df[pos_mask]
        if len(neg) == 0 or len(pos) == 0:
            print("[AspectMiningService] insufficient data — skipping build")
            return

        # Fit on union of cohorts
        self._vec = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=20,
            max_df=0.5,
            stop_words="english",
            sublinear_tf=True,
        )
        self._vec.fit(pd.concat([neg["text"], pos["text"]], ignore_index=True))
        self._feat_names = list(self._vec.get_feature_names_out())

        X_neg = self._vec.transform(neg["text"])
        X_pos = self._vec.transform(pos["text"])

        self._pos_mean = self._weighted_mean(X_pos, pos["weight"].values)
        neg_mean       = self._weighted_mean(X_neg, neg["weight"].values)
        self._global_score = neg_mean - self._pos_mean

        # Per-brand
        for brand, bn in neg.groupby("brand_name"):
            if len(bn) < 30:
                continue
            Xb = self._vec.transform(bn["text"])
            mean_b = self._weighted_mean(Xb, bn["weight"].values)
            self._brand_scores[brand] = mean_b - self._pos_mean

        # Per-product (only products with >=10 negative reviews)
        for product_id, pn in neg.groupby("product_id"):
            if len(pn) < 10:
                continue
            Xp = self._vec.transform(pn["text"])
            mean_p = self._weighted_mean(Xp, pn["weight"].values)
            self._product_scores[str(product_id)] = mean_p - self._pos_mean

        print(f"[AspectMiningService] vocab={len(self._feat_names):,}  "
              f"brands={len(self._brand_scores)}  products={len(self._product_scores)}")

    @staticmethod
    def _weighted_mean(X, weights: np.ndarray) -> np.ndarray:
        w = weights.reshape(-1, 1)
        num = np.asarray((X.multiply(w)).sum(axis=0)).ravel()
        denom = float(weights.sum())
        return num / denom if denom > 0 else num

    def _top_n(self, scores: np.ndarray, n: int) -> list[dict]:
        if scores is None or len(scores) == 0:
            return []
        idx = np.argsort(-scores)[:n]
        return [
            {"term": self._feat_names[i], "score": round(float(scores[i]), 4)}
            for i in idx if scores[i] > 0
        ]

    # ── public API ────────────────────────────────────────────────────────────

    def top_complaints(self, limit: int = 20) -> list[dict]:
        """Catalogue-wide most-discriminative complaint terms."""
        return self._top_n(self._global_score, limit)

    def top_complaints_by_brand(self, brand: str, limit: int = 10) -> list[dict]:
        scores = self._brand_scores.get(brand)
        return self._top_n(scores, limit) if scores is not None else []

    def top_complaints_for_product(self, product_id: str, limit: int = 10) -> list[dict]:
        scores = self._product_scores.get(str(product_id))
        return self._top_n(scores, limit) if scores is not None else []

    def available_brands(self) -> list[str]:
        return sorted(self._brand_scores.keys())
