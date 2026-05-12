"""
AnalyticsService — aggregates review label data for the Admin Dashboard.

Live metrics (Model Trust, brand stats from app-submitted reviews) come from
the SQLAlchemy `reviews` table. Historical analytics (sentiment trends over
time, price-tier behaviour) are computed from the bundled raw CSV at startup
so the dashboard has rich context even before users submit live reviews.

Methodology mirrors data/task2_3.ipynb:
  - "Recommended" iff review_rating >= RECOMMEND_RATING_THRESHOLD
  - is_a_buyer used as trust weight (verified=1.0, non-buyer=0.3)
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd

import config
from database import Review
from data_access.historical_review_repository import HistoricalReviewRepository
from data_access.product_repository import ProductRepository
from enums.label import RecommendLabel


class AnalyticsService:
    def __init__(
        self,
        product_repo: ProductRepository,
        historical_repo: HistoricalReviewRepository | None = None,
    ) -> None:
        self._product_repo = product_repo
        self._historical_repo = historical_repo
        self._historical = self._load_historical()

    # ── historical CSV-based pre-computation (one-time at startup) ───────────

    def _load_historical(self) -> dict:
        """Pre-compute monthly trends + price-tier stats from the bundled CSV."""
        df = pd.read_csv(config.RAW_CSV_PATH, low_memory=False)
        df = df.dropna(subset=["review_rating", "is_a_buyer", "brand_name"]).copy()

        df["is_a_buyer"]  = df["is_a_buyer"].astype(str).str.upper().eq("TRUE")
        df["review_rating"] = df["review_rating"].astype(float)
        df["recommended"] = (df["review_rating"] >= config.RECOMMEND_RATING_THRESHOLD).astype(int)
        df["weight"]      = np.where(df["is_a_buyer"], 1.0, config.NON_BUYER_TRUST_WEIGHT)

        # Parse review_date (DD/MM/YYYY HH:MM)
        df["date"] = pd.to_datetime(df["review_date"], format="%d/%m/%Y %H:%M", errors="coerce")
        dated = df.dropna(subset=["date"]).copy()
        dated["month"] = dated["date"].dt.to_period("M").dt.to_timestamp()

        print(f"[AnalyticsService] loaded {len(df):,} historical reviews "
              f"({len(dated):,} dated)")
        return {"df": df, "dated": dated}

    # ── live overview (SQLAlchemy reviews) ────────────────────────────────────

    def overview(self, limit: int = 100, days: int = 30) -> dict:
        """
        Return metrics over the combined corpus of historical CSV reviews and
        live SQLite reviews. The `days` window still constrains which live
        reviews are included (so admins can see "recent" app activity in the
        Model Trust block); CSV reviews are always included in full.

        Model Trust stats (override rate, AI-vs-rating agreement, confusion
        matrix) come from live reviews only because CSV rows have no AI
        prediction or override flag.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        live_rows = (
            Review.query
            .filter(Review.created_at >= cutoff)
            .order_by(Review.created_at.desc())
            .limit(limit)
            .all()
        )

        rec     = RecommendLabel.RECOMMEND.value

        # ── live (SQLite) aggregates ──
        live_n           = len(live_rows)
        live_rec         = sum(1 for r in live_rows if r.final_label == rec)
        live_verified    = sum(1 for r in live_rows if r.is_verified_buyer)
        live_override    = sum(1 for r in live_rows if r.overridden)
        live_weights     = [1.0 if r.is_verified_buyer else config.NON_BUYER_TRUST_WEIGHT for r in live_rows]
        live_weighted_rec_sum = sum(
            w * (1.0 if r.final_label == rec else 0.0)
            for w, r in zip(live_weights, live_rows)
        )
        live_weight_sum = sum(live_weights)

        # AI-vs-rating confusion (live only — CSV has no AI prediction)
        tp = fp = tn = fn = 0
        for r in live_rows:
            ai_pos    = r.ai_label == rec
            truth_pos = r.rating >= config.RECOMMEND_RATING_THRESHOLD
            if   ai_pos and truth_pos:     tp += 1
            elif ai_pos and not truth_pos: fp += 1
            elif (not ai_pos) and truth_pos: fn += 1
            else: tn += 1

        # ── historical (CSV) aggregates ──
        hist_df = self._historical.get("df")
        hist_n        = int(len(hist_df)) if hist_df is not None else 0
        hist_rec      = int(hist_df["recommended"].sum()) if hist_n else 0
        hist_verified = int(hist_df["is_a_buyer"].sum()) if hist_n else 0
        hist_weight_sum       = float(hist_df["weight"].sum()) if hist_n else 0.0
        hist_weighted_rec_sum = float((hist_df["recommended"] * hist_df["weight"]).sum()) if hist_n else 0.0

        # ── combined metrics ──
        total_n            = live_n + hist_n
        if total_n == 0:
            return self._empty_overview(limit, days)

        recommend_count     = live_rec + hist_rec
        not_recommend_count = total_n - recommend_count
        recommend_rate      = round(recommend_count / total_n * 100, 1)

        verified_count = live_verified + hist_verified
        verified_share = round(verified_count / total_n * 100, 1)

        weight_sum     = (live_weight_sum + hist_weight_sum) or 1.0
        weighted_rate  = round((live_weighted_rec_sum + hist_weighted_rec_sum) / weight_sum * 100, 1)

        # Override / agreement still computed against live reviews only.
        override_rate = round(live_override / live_n * 100, 1) if live_n else 0.0
        agreement     = round((tp + tn) / live_n * 100, 1)   if live_n else 0.0

        return {
            "total_reviews":                  total_n,
            "recommend_count":                recommend_count,
            "not_recommend_count":            not_recommend_count,
            "recommend_rate_percent":         recommend_rate,
            "recommend_rate_weighted_percent": weighted_rate,
            "override_count":                 live_override,
            "override_rate_percent":          override_rate,
            "verified_buyer_share_percent":   verified_share,
            "ai_rating_agreement_percent":    agreement,
            "confusion":                      {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "time_window_days":               days,
            "limit":                          limit,
            "live_review_count":              live_n,
            "historical_review_count":        hist_n,
        }

    def _empty_overview(self, limit: int, days: int) -> dict:
        return {
            "total_reviews":                  0,
            "recommend_count":                0,
            "not_recommend_count":            0,
            "recommend_rate_percent":         0.0,
            "recommend_rate_weighted_percent": 0.0,
            "override_count":                 0,
            "override_rate_percent":          0.0,
            "verified_buyer_share_percent":   0.0,
            "ai_rating_agreement_percent":    0.0,
            "confusion":                      {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
            "time_window_days":               days,
            "limit":                          limit,
            "live_review_count":              0,
            "historical_review_count":        0,
        }

    # ── brand stats (SQLAlchemy) ──────────────────────────────────────────────

    def brands(self, limit: int = 10, min_reviews: int = 5) -> list[dict]:
        """
        Top brands by recommendation rate over the combined corpus of
        historical CSV reviews and live SQLite reviews.
        """
        rec = RecommendLabel.RECOMMEND.value

        brand_stats: dict[str, dict] = defaultdict(lambda: {
            "rec": 0, "total": 0, "rating_sum": 0.0, "verified": 0
        })

        # ── historical CSV reviews ──
        hist_df = self._historical.get("df")
        if hist_df is not None and len(hist_df) > 0:
            agg = hist_df.groupby("brand_name").agg(
                total       = ("review_rating", "size"),
                rating_sum  = ("review_rating", "sum"),
                rec         = ("recommended",   "sum"),
                verified    = ("is_a_buyer",    "sum"),
            )
            for brand, row in agg.iterrows():
                s = brand_stats[str(brand)]
                s["total"]      += int(row["total"])
                s["rating_sum"] += float(row["rating_sum"])
                s["rec"]        += int(row["rec"])
                s["verified"]   += int(row["verified"])

        # ── live SQLite reviews ──
        for r in Review.query.all():
            product = self._product_repo.get_by_id(r.product_id)
            brand   = product["brand_name"] if product else "Unknown"
            s = brand_stats[brand]
            s["total"]      += 1
            s["rating_sum"] += float(r.rating)
            if r.final_label == rec:
                s["rec"] += 1
            if r.is_verified_buyer:
                s["verified"] += 1

        result = []
        for brand, s in brand_stats.items():
            if s["total"] < min_reviews:
                continue
            rate           = round(s["rec"] / s["total"] * 100, 1)
            avg_rating     = round(s["rating_sum"] / s["total"], 2)
            verified_share = round(s["verified"] / s["total"] * 100, 1)
            result.append({
                "brand_name":              brand,
                "total_reviews":           s["total"],
                "recommend_count":         s["rec"],
                "recommend_rate_percent":  rate,
                "avg_rating":              avg_rating,
                "verified_buyer_share_percent": verified_share,
            })
        result.sort(key=lambda x: x["recommend_rate_percent"], reverse=True)
        return result[:limit]

    # ── sentiment trends over time (historical CSV) ──────────────────────────

    def trends(self, top_n_brands: int = 5, min_reviews_per_month: int = 5) -> dict:
        """
        Monthly weighted recommendation rate for the top-N brands by review
        count. Returns:
            {
              "brands":  ["Olay", "Lakme", ...],
              "series":  {brand: [{month: "2021-03", rate: 0.86, n_reviews: 142}, ...]}
            }
        """
        dated = self._historical["dated"]
        top_brands = dated["brand_name"].value_counts().head(top_n_brands).index.tolist()
        series = {}
        for brand in top_brands:
            sub = dated[dated["brand_name"] == brand]
            grp = sub.groupby("month")
            points = []
            for month, g in grp:
                n = len(g)
                if n < min_reviews_per_month:
                    continue
                w_sum = g["weight"].sum()
                rate  = (g["recommended"] * g["weight"]).sum() / w_sum if w_sum > 0 else 0.0
                points.append({
                    "month":     month.strftime("%Y-%m"),
                    "rate":      round(float(rate), 4),
                    "n_reviews": int(n),
                })
            points.sort(key=lambda p: p["month"])
            series[brand] = points
        return {"brands": top_brands, "series": series}

    # ── price-tier behaviour (historical CSV) ────────────────────────────────

    def price_tiers(self) -> list[dict]:
        """
        Trust-weighted recommend rate / avg rating / verified-buyer share per
        price tier (Budget / Mid / Premium).
        """
        df = self._historical["df"].dropna(subset=["price"]).copy()
        df["price"] = df["price"].astype(float)

        result = []
        for label, lo, hi in config.PRICE_TIERS:
            mask = (df["price"] >= lo) & (df["price"] < hi)
            sub = df[mask]
            n = len(sub)
            if n == 0:
                result.append({
                    "tier":                          label,
                    "n_reviews":                     0,
                    "weighted_recommend_rate":       0.0,
                    "recommend_rate_percent":        0.0,
                    "weighted_avg_rating":           0.0,
                    "verified_buyer_share_percent":  0.0,
                })
                continue
            w_sum = sub["weight"].sum() or 1.0
            wrec  = (sub["recommended"] * sub["weight"]).sum() / w_sum
            wavg  = (sub["review_rating"] * sub["weight"]).sum() / w_sum
            result.append({
                "tier":                          label,
                "n_reviews":                     int(n),
                "weighted_recommend_rate":       round(float(wrec), 4),
                "recommend_rate_percent":        round(float(wrec) * 100, 1),
                "weighted_avg_rating":           round(float(wavg), 3),
                "verified_buyer_share_percent":  round(float(sub["is_a_buyer"].mean()) * 100, 1),
            })
        return result
