"""
Product routes — all /api/products/* endpoints.
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request, current_app


products_bp = Blueprint("products", __name__, url_prefix="/api/products")


def _services():
    return current_app.config["SERVICES"]


# ── GET /api/products ─────────────────────────────────────────────────────────
@products_bp.get("")
def list_products():
    """
    Search / filter products.
    Query params:
        q           — keyword (brand, name, description)
        brand       — comma-separated brand names
        category    — comma-separated category names
        min_price   — float
        max_price   — float
    """
    svc = _services()["search"]
    q         = request.args.get("q", "").strip()
    brand     = request.args.get("brand", "").strip() or None
    category  = request.args.get("category", "").strip() or None
    min_price = _float_or_none(request.args.get("min_price"))
    max_price = _float_or_none(request.args.get("max_price"))

    result = svc.search(query=q, brand=brand, category=category,
                        min_price=min_price, max_price=max_price)
    return jsonify(result), 200


# ── GET /api/products/filters ─────────────────────────────────────────────────
@products_bp.get("/filters")
def filter_options():
    """Return distinct brands, categories, and price range for the filter sidebar."""
    svc = _services()["search"]
    return jsonify({
        "brands":     svc.distinct_brands(),
        "categories": svc.distinct_categories(),
        "price_range": svc.price_range(),
    }), 200


# ── GET /api/products/<product_id> ────────────────────────────────────────────
@products_bp.get("/<product_id>")
def get_product(product_id: str):
    repo = _services()["product_repo"]
    product = repo.get_by_id(product_id)
    if product is None:
        return jsonify({"error": "Product not found"}), 404
    return jsonify(product), 200


# ── GET /api/products/<product_id>/reviews ────────────────────────────────────
@products_bp.get("/<product_id>/reviews")
def get_product_reviews(product_id: str):
    svc = _services()["review"]
    reviews = svc.get_reviews_for_product(product_id)
    return jsonify({"reviews": reviews}), 200


# ── POST /api/products/<product_id>/reviews ───────────────────────────────────
@products_bp.post("/<product_id>/reviews")
def create_review(product_id: str):
    """
    Body (JSON):
        title          — string (required)
        description    — string (required)
        rating         — int 1-5 (required)
        label_override — "Buy" | "Not Buy" (optional)
    """
    data = request.get_json(silent=True) or {}
    title       = str(data.get("title", "")).strip()
    description = str(data.get("description", "")).strip()
    rating_raw  = data.get("rating")
    label_override = data.get("label_override")

    if not title or not description:
        return jsonify({"error": "title and description are required"}), 400

    try:
        rating = int(rating_raw)
        if not 1 <= rating <= 5:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"error": "rating must be an integer between 1 and 5"}), 400

    svc = _services()["review"]
    saved = svc.create_review(
        product_id=product_id,
        title=title,
        description=description,
        rating=rating,
        label_override=label_override,
    )
    return jsonify(saved), 201


# ── GET /api/products/<product_id>/similar ────────────────────────────────────
@products_bp.get("/<product_id>/similar")
def similar_products(product_id: str):
    svc = _services()["recommendation"]
    products = svc.similar(product_id)
    return jsonify({"products": products}), 200


# ── helpers ───────────────────────────────────────────────────────────────────

def _float_or_none(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None
