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
    Query params:
        limit       (int, default 10) — max brands to return
        min_reviews (int, default 5)  — minimum reviews to include a brand
    """
    svc         = _services()["analytics"]
    limit       = _int_or(request.args.get("limit"),       10)
    min_reviews = _int_or(request.args.get("min_reviews"),  5)
    return jsonify({"brands": svc.brands(limit=limit, min_reviews=min_reviews)}), 200


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
