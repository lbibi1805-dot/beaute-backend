"""
ReviewRepository — persists and retrieves reviews via SQLite (SQLAlchemy).

All database operations happen within the active Flask application context.
The schema is created automatically by app.py on first startup via db.create_all().
"""
from __future__ import annotations

import uuid

from database import db, Review


class ReviewRepository:
    # ── public API ────────────────────────────────────────────────────────────

    def save(self, review: dict) -> dict:
        """Assign a UUID, persist to SQLite, and return the saved record as dict."""
        record = Review(
            id                = str(uuid.uuid4()),
            product_id        = review["product_id"],
            author            = review.get("author"),
            title             = review["title"],
            description       = review["description"],
            rating            = review["rating"],
            ai_label          = review["ai_label"],
            final_label       = review["final_label"],
            overridden        = review.get("overridden", False),
            is_verified_buyer = review.get("is_verified_buyer", True),
        )
        db.session.add(record)
        db.session.commit()
        return record.to_dict()

    def get_by_id(self, review_id: str) -> dict | None:
        record = db.session.get(Review, review_id)
        return record.to_dict() if record else None

    def get_by_product_id(self, product_id: str) -> list[dict]:
        records = Review.query.filter_by(product_id=product_id, is_deleted=False).order_by(Review.created_at.desc()).all()
        return [r.to_dict() for r in records]

    def all(self) -> list[dict]:
        return [r.to_dict() for r in Review.query.filter_by(is_deleted=False).order_by(Review.created_at.desc()).all()]

    def delete_by_id(self, review_id: str) -> bool:
        """Soft-delete a review. Returns True if found and marked deleted, False if not found."""
        record = db.session.get(Review, review_id)
        if record is None or record.is_deleted:
            return False
        record.is_deleted = True
        db.session.commit()
        return True

    def delete_by_id_as_user(self, review_id: str, username: str) -> tuple[bool, str]:
        """
        Soft-delete a review only if it belongs to `username`.
        Returns (True, "") on success, (False, "not_found") or (False, "forbidden").
        """
        record = db.session.get(Review, review_id)
        if record is None or record.is_deleted:
            return False, "not_found"
        if record.author != username:
            return False, "forbidden"
        record.is_deleted = True
        db.session.commit()
        return True, ""
