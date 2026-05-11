"""
Admin routes — /api/admin/*
All endpoints require Admin role (enforced by @require_admin decorator).
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request, current_app

from utils.auth_guard import require_admin

admin_bp = Blueprint("admin", __name__, url_prefix="/api/admin")


def _services():
    return current_app.config["SERVICES"]


# ── GET /api/admin/analytics/overview ────────────────────────────────────────
@admin_bp.get("/analytics/overview")
@require_admin
def analytics_overview():
    """
    Returns live review metrics + Model Trust stats (override rate,
    AI-vs-rating agreement, confusion matrix, verified-buyer share).

    Query params:
        limit (int, default 100) — max number of recent reviews to consider
        days  (int, default 30)  — time window in days
    """
    svc   = _services()["analytics"]
    limit = _int_or(request.args.get("limit"), 100)
    days  = _int_or(request.args.get("days"),  30)
    return jsonify(svc.overview(limit=limit, days=days)), 200


# ── GET /api/admin/analytics/brands ──────────────────────────────────────────
@admin_bp.get("/analytics/brands")
@require_admin
def analytics_brands():
    """
    Top brands by recommendation rate (from live app reviews).

    Query params:
        limit       (int, default 10) — max brands to return
        min_reviews (int, default 5)  — minimum reviews to include a brand
    """
    svc         = _services()["analytics"]
    limit       = _int_or(request.args.get("limit"),       10)
    min_reviews = _int_or(request.args.get("min_reviews"),  5)
    return jsonify({"brands": svc.brands(limit=limit, min_reviews=min_reviews)}), 200


# ── GET /api/admin/analytics/trends ──────────────────────────────────────────
@admin_bp.get("/analytics/trends")
@require_admin
def analytics_trends():
    """
    Monthly weighted recommendation rate for top-N brands (historical CSV).

    Query params:
        top_n_brands         (int, default 5)
        min_reviews_per_month (int, default 5)
    """
    svc                  = _services()["analytics"]
    top_n_brands         = _int_or(request.args.get("top_n_brands"), 5)
    min_reviews_per_month = _int_or(request.args.get("min_reviews_per_month"), 5)
    return jsonify(svc.trends(
        top_n_brands=top_n_brands,
        min_reviews_per_month=min_reviews_per_month,
    )), 200


# ── GET /api/admin/analytics/price-tiers ──────────────────────────────────────
@admin_bp.get("/analytics/price-tiers")
@require_admin
def analytics_price_tiers():
    """Per-tier (Budget / Mid / Premium) weighted satisfaction metrics."""
    svc = _services()["analytics"]
    return jsonify({"tiers": svc.price_tiers()}), 200


# ── GET /api/admin/analytics/complaints ───────────────────────────────────────
@admin_bp.get("/analytics/complaints")
@require_admin
def analytics_complaints():
    """
    Top complaint terms from negative reviews (mined via discriminative
    weighted TF-IDF).

    Query params:
        brand (str, optional) — if provided, return brand-specific complaints
        limit (int, default 20)
    """
    svc   = _services()["aspect_mining"]
    brand = request.args.get("brand", "").strip() or None
    limit = _int_or(request.args.get("limit"), 20)
    if brand:
        return jsonify({"brand": brand, "complaints": svc.top_complaints_by_brand(brand, limit=limit)}), 200
    return jsonify({
        "brand": None,
        "complaints": svc.top_complaints(limit=limit),
        "available_brands": svc.available_brands(),
    }), 200


# ── DELETE /api/admin/reviews/<review_id> ────────────────────────────────────
@admin_bp.delete("/reviews/<review_id>")
@require_admin
def delete_review(review_id: str):
    """Hard-delete a review by UUID."""
    repo    = _services()["review_repo"]
    deleted = repo.delete_by_id(review_id)
    if not deleted:
        return jsonify({"error": "Review not found"}), 404
    return "", 204


# ── helpers ───────────────────────────────────────────────────────────────────
def _int_or(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
