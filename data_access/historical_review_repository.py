"""
HistoricalReviewRepository — exposes the bundled cosmetics CSV reviews as if
they came from the same store as live app reviews.

Reviews are loaded once at startup and indexed by product_id. Each row is
converted to the dict shape returned by ReviewRepository.to_dict() so the
service layer can merge them with live SQLite reviews seamlessly.

Labels are derived from the rating using the same threshold the ML target
uses (>= RECOMMEND_RATING_THRESHOLD → "Recommended"). CSV rows have no
AI prediction, so ai_label == final_label and overridden = False.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

import config
from enums.label import RecommendLabel


class HistoricalReviewRepository:
    CSV_ID_PREFIX = "csv-"

    def __init__(self) -> None:
        self._all: list[dict] = []
        self._by_product: dict[str, list[dict]] = defaultdict(list)
        self._by_id: dict[str, dict] = {}
        self._load()

    # ── internal ──────────────────────────────────────────────────────────────

    def _load(self) -> None:
        df = pd.read_csv(config.RAW_CSV_PATH, low_memory=False)
        df = df.dropna(subset=["product_id", "review_id", "review_rating"]).copy()

        df["is_a_buyer_bool"] = df["is_a_buyer"].astype(str).str.upper().eq("TRUE")
        df["created_at"] = pd.to_datetime(df["review_date"],
                                          format="%d/%m/%Y %H:%M",
                                          errors="coerce")

        rec_value = RecommendLabel.RECOMMEND.value
        not_rec_value = RecommendLabel.NOT_RECOMMEND.value
        threshold = config.RECOMMEND_RATING_THRESHOLD

        for row in df.itertuples(index=False):
            product_id  = str(getattr(row, "product_id"))
            review_id   = f"{self.CSV_ID_PREFIX}{int(getattr(row, 'review_id'))}"
            rating      = int(getattr(row, "review_rating"))
            label_value = rec_value if rating >= threshold else not_rec_value
            created_at  = getattr(row, "created_at")
            created_iso = created_at.isoformat() if pd.notna(created_at) else None

            record = {
                "review_id":         review_id,
                "product_id":        product_id,
                "title":             _str_or_default(getattr(row, "review_title"), ""),
                "description":       _str_or_default(getattr(row, "review_text"), ""),
                "rating":            rating,
                "ai_label":          label_value,
                "final_label":       label_value,
                "overridden":        False,
                "is_verified_buyer": bool(getattr(row, "is_a_buyer_bool")),
                "created_at":        created_iso,
                "review_url":        f"/api/reviews/{review_id}",
                "source":            "historical",
                "author":            _str_or_default(getattr(row, "author"), ""),
                "brand_name":        _str_or_default(getattr(row, "brand_name"), ""),
            }
            self._all.append(record)
            self._by_product[product_id].append(record)
            self._by_id[review_id] = record

        # Sort each product's reviews newest-first to match SQLite ordering.
        for product_id, reviews in self._by_product.items():
            reviews.sort(
                key=lambda r: r["created_at"] or "",
                reverse=True,
            )
        print(f"[HistoricalReviewRepository] loaded {len(self._all):,} CSV reviews "
              f"across {len(self._by_product):,} products")

    # ── public API ────────────────────────────────────────────────────────────

    def all(self) -> list[dict]:
        return self._all

    def get_by_product_id(self, product_id: str) -> list[dict]:
        return list(self._by_product.get(str(product_id), []))

    def get_by_id(self, review_id: str) -> dict | None:
        return self._by_id.get(review_id)


def _str_or_default(value, default: str) -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text else default
