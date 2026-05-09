"""
ProductRepository — loads the cosmetics CSV and exposes a queryable product catalog.

Products are derived by grouping reviews on product_id.
The repository builds an in-memory list/dict at startup and exposes
get/search helpers used by the service layer.
"""
from __future__ import annotations

import pandas as pd

import config


class ProductRepository:
    def __init__(self) -> None:
        self._products: list[dict] = []
        self._by_id: dict[str, dict] = {}
        self._load()

    # ── internal ──────────────────────────────────────────────────────────────

    def _load(self) -> None:
        df = pd.read_csv(config.RAW_CSV_PATH, low_memory=False)

        # Derive one record per unique product — keep first occurrence of each
        # product_id for stable attribute values.
        product_cols = [c for c in df.columns if c not in (
            "review_id", "review_title", "review_text",
            "review_rating", "review_votes", "is_a_buyer",
        )]

        products_df = (
            df[product_cols]
            .drop_duplicates(subset=["product_id"])
            .reset_index(drop=True)
        )

        for _, row in products_df.iterrows():
            product_id = str(row.get("product_id", ""))
            # Build a clean stats snapshot from all reviews for this product
            reviews_subset = df[df["product_id"] == row["product_id"]]
            avg_rating = round(float(reviews_subset["review_rating"].mean()), 2) \
                if "review_rating" in df.columns else 0.0
            review_count = int(len(reviews_subset))

            record = {
                "product_id":   product_id,
                "product_name": str(row.get("product_name", "")),
                "brand_name":   str(row.get("brand_name", "")),
                "product_title":str(row.get("product_title", row.get("product_name", ""))),
                "price":        self._safe_float(row.get("price", 0)),
                "category":     str(row.get("product_type", row.get("category", "Beauty"))),
                "image_url":    str(row.get("image_url", "")),
                "avg_rating":   avg_rating,
                "review_count": review_count,
                # description composed from available text columns
                "description":  self._build_description(row),
            }
            self._products.append(record)
            self._by_id[product_id] = record

        print(f"[ProductRepository] loaded {len(self._products)} products")

    @staticmethod
    def _safe_float(val) -> float:
        try:
            return round(float(val), 2)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _build_description(row: pd.Series) -> str:
        for col in ("product_description", "description", "product_title", "product_name"):
            val = row.get(col, "")
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""

    # ── public API ────────────────────────────────────────────────────────────

    def all(self) -> list[dict]:
        return self._products

    def get_by_id(self, product_id: str) -> dict | None:
        return self._by_id.get(product_id)

    def get_all_descriptions(self) -> list[tuple[str, str]]:
        """Return list of (product_id, description_text) for the similarity engine."""
        return [(p["product_id"], p["description"] or p["product_name"]) for p in self._products]
