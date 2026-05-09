"""
Review routes — /api/reviews/* endpoints.

Exposes a URL per saved review so that a review can be fetched directly,
satisfying the Task 2 "accessible through a URL" requirement.
"""
from __future__ import annotations

from flask import Blueprint, jsonify, current_app


reviews_bp = Blueprint("reviews", __name__, url_prefix="/api/reviews")


def _services():
    return current_app.config["SERVICES"]


# ── GET /api/reviews/<review_id> ──────────────────────────────────────────────
@reviews_bp.get("/<review_id>")
def get_review(review_id: str):
    svc = _services()["review"]
    review = svc.get_review(review_id)
    if review is None:
        return jsonify({"error": "Review not found"}), 404
    review.setdefault("review_url", f"/api/reviews/{review_id}")
    return jsonify(review), 200
