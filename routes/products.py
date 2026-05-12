"""
Product routes — all /api/products/* endpoints.
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request, current_app

from utils.helpers import float_or_none as _float_or_none, int_or as _int_or


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
        sort        — price_asc | price_desc | rating_asc | rating_desc
    """
    svc = _services()["search"]
    q         = request.args.get("q", "").strip()
    brand     = request.args.get("brand", "").strip() or None
    category  = request.args.get("category", "").strip() or None
    min_price = _float_or_none(request.args.get("min_price"))
    max_price = _float_or_none(request.args.get("max_price"))
    sort      = request.args.get("sort", "").strip() or None

    result = svc.search(query=q, brand=brand, category=category,
                        min_price=min_price, max_price=max_price, sort=sort)
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
        label_override — "Recommended" | "Not Recommended" (optional)
    """
    data = request.get_json(silent=True) or {}
    title       = str(data.get("title", "")).strip()
    description = str(data.get("description", "")).strip()
    rating_raw  = data.get("rating")
    label_override = data.get("label_override")

    if not title or not description:
        return jsonify({"error": "title and description are required"}), 400
    if len(title) > 256:
        return jsonify({"error": "title must be 256 characters or fewer"}), 400
    if len(description) > 5000:
        return jsonify({"error": "description must be 5000 characters or fewer"}), 400

    try:
        rating = int(rating_raw)
        if not 1 <= rating <= 5:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"error": "rating must be an integer between 1 and 5"}), 400

    # Extract author from Bearer token (optional — guests may submit without a token)
    author: str | None = None
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        author = _services()["auth"].get_username(token)

    svc = _services()["review"]
    saved = svc.create_review(
        product_id=product_id,
        title=title,
        description=description,
        rating=rating,
        label_override=label_override,
        is_verified_buyer=(author is not None),
        author=author,
    )
    return jsonify(saved), 201


# ── DELETE /api/products/<product_id>/reviews/<review_id> ─────────────────────────────
@products_bp.delete("/<product_id>/reviews/<review_id>")
def delete_own_review(product_id: str, review_id: str):
    """
    Allows an authenticated user to delete their own review.
    Admins may delete any review via DELETE /api/admin/reviews/<id>.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"error": "Authentication required"}), 401

    token    = auth_header[7:].strip()
    username = _services()["auth"].get_username(token)
    if not username:
        return jsonify({"error": "Invalid token"}), 401

    repo = _services()["review_repo"]
    ok, reason = repo.delete_by_id_as_user(review_id, username)
    if not ok:
        if reason == "not_found":
            return jsonify({"error": "Review not found"}), 404
        return jsonify({"error": "You can only delete your own reviews"}), 403
    return "", 204


# ── GET /api/products/<product_id>/similar ────────────────────────────────────
@products_bp.get("/<product_id>/similar")
def similar_products(product_id: str):
    """Content-based similar products (TF-IDF cosine on descriptions)."""
    svc = _services()["recommendation"]
    products = svc.similar(product_id)
    return jsonify({"products": products}), 200


# ── GET /api/products/<product_id>/cooccurring ────────────────────────────────
@products_bp.get("/<product_id>/cooccurring")
def cooccurring_products(product_id: str):
    """Author-based 'Customers also bought' (Jaccard on verified-buyer reviews)."""
    svc = _services()["recommendation"]
    products = svc.cooccurring_products(product_id)
    return jsonify({"products": products}), 200


# ── GET /api/products/<product_id>/complaints ─────────────────────────────────
@products_bp.get("/<product_id>/complaints")
def product_complaints(product_id: str):
    """Top complaint terms mined from this product's negative reviews."""
    svc = _services()["aspect_mining"]
    limit = _int_or(request.args.get("limit"), 10)
    return jsonify({"complaints": svc.top_complaints_for_product(product_id, limit=limit)}), 200

