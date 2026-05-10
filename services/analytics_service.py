"""
AnalyticsService — aggregates review label data for the Admin Dashboard.

All aggregation is done server-side; the frontend receives pre-computed totals.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

from database import Review
from data_access.product_repository import ProductRepository


class AnalyticsService:
    def __init__(self, product_repo: ProductRepository) -> None:
        self._product_repo = product_repo

    # ── public API ────────────────────────────────────────────────────────────

    def overview(self, limit: int = 100, days: int = 30) -> dict:
        """
        Return label counts for the most recent `limit` reviews created
        within the last `days` days.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        rows = (
            Review.query
            .filter(Review.created_at >= cutoff)
            .order_by(Review.created_at.desc())
            .limit(limit)
            .all()
        )

        buy_count     = sum(1 for r in rows if r.final_label == "Buy")
        not_buy_count = len(rows) - buy_count
        buy_rate      = round(buy_count / len(rows) * 100, 1) if rows else 0.0

        return {
            "total_reviews":    len(rows),
            "buy_count":        buy_count,
            "not_buy_count":    not_buy_count,
            "buy_rate_percent": buy_rate,
            "time_window_days": days,
            "limit":            limit,
        }

    def brands(self, limit: int = 10, min_reviews: int = 5) -> list[dict]:
        """
        Return top brands ranked by buy rate, with at least `min_reviews` reviews.
        """
        all_reviews = Review.query.all()

        # Aggregate per product first, then join brand
        product_stats: dict[str, dict] = {}
        for r in all_reviews:
            if r.product_id not in product_stats:
                product_stats[r.product_id] = {"buy": 0, "total": 0}
            product_stats[r.product_id]["total"] += 1
            if r.final_label == "Buy":
                product_stats[r.product_id]["buy"] += 1

        # Group by brand
        brand_stats: dict[str, dict] = {}
        for product_id, stats in product_stats.items():
            product = self._product_repo.get_by_id(product_id)
            brand = product["brand_name"] if product else "Unknown"
            if brand not in brand_stats:
                brand_stats[brand] = {"buy": 0, "total": 0}
            brand_stats[brand]["buy"]   += stats["buy"]
            brand_stats[brand]["total"] += stats["total"]

        # Filter, compute rate, sort
        result = []
        for brand, stats in brand_stats.items():
            if stats["total"] < min_reviews:
                continue
            buy_rate = round(stats["buy"] / stats["total"] * 100, 1)
            result.append({
                "brand_name":       brand,
                "total_reviews":    stats["total"],
                "buy_count":        stats["buy"],
                "buy_rate_percent": buy_rate,
            })

        result.sort(key=lambda x: x["buy_rate_percent"], reverse=True)
        return result[:limit]
