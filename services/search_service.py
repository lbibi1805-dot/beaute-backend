"""
SearchService — Task 1 search requirement.

Supports case-insensitive, partial-match keyword search over brand_name,
product_name, and description. Handles similar keyword forms by normalising
both query and data to lowercase before comparison.

Supports optional filter params for Task 4 (brand, category, price range).
"""
from __future__ import annotations

import re

from data_access.product_repository import ProductRepository


class SearchService:
    def __init__(self, product_repo: ProductRepository) -> None:
        self._repo = product_repo

    # ── public API ────────────────────────────────────────────────────────────

    def search(
        self,
        query: str = "",
        brand: str | None = None,
        category: str | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        sort: str | None = None,
    ) -> dict:
        """
        Return {count, products} matching all supplied filters.
        All string comparisons are case-insensitive.

        sort options: price_asc, price_desc, rating_asc, rating_desc
        """
        products = self._repo.all()

        if query:
            products = self._keyword_filter(products, query)

        if brand:
            brands_lower = {b.strip().lower() for b in brand.split(",")}
            products = [p for p in products if p["brand_name"].lower() in brands_lower]

        if category:
            cats_lower = {c.strip().lower() for c in category.split(",")}
            products = [p for p in products if p["category"].lower() in cats_lower]

        if min_price is not None:
            products = [p for p in products if p["price"] >= min_price]

        if max_price is not None:
            products = [p for p in products if p["price"] <= max_price]

        _SORT_KEY = {
            "price_asc":    (lambda p: p["price"],      False),
            "price_desc":   (lambda p: p["price"],      True),
            "rating_asc":   (lambda p: p["avg_rating"], False),
            "rating_desc":  (lambda p: p["avg_rating"], True),
        }
        if sort and sort in _SORT_KEY:
            key_fn, reverse = _SORT_KEY[sort]
            products = sorted(products, key=key_fn, reverse=reverse)

        return {"count": len(products), "products": products}

    def distinct_brands(self) -> list[str]:
        return sorted({p["brand_name"] for p in self._repo.all() if p["brand_name"]})

    def distinct_categories(self) -> list[str]:
        return sorted({p["category"] for p in self._repo.all() if p["category"]})

    def price_range(self) -> dict:
        prices = [p["price"] for p in self._repo.all() if p["price"] > 0]
        return {
            "min": round(min(prices), 2) if prices else 0,
            "max": round(max(prices), 2) if prices else 0,
        }

    # ── internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _keyword_filter(products: list[dict], query: str) -> list[dict]:
        """
        Match products whose brand, name, or description contains the query.
        Supports partial matches and is case-insensitive.
        We strip all non-alphanumeric chars before comparing so that
        "Maybeline New York" matches "maybelline" approximately.
        """
        normalised_query = _normalise(query)
        result = []
        for p in products:
            haystack = _normalise(
                f"{p['brand_name']} {p['product_name']} {p['description']}"
            )
            if normalised_query in haystack:
                result.append(p)
        return result


def _normalise(text: str) -> str:
    """Lowercase and collapse non-alphanumeric runs to a single space."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
