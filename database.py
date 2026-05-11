"""
SQLAlchemy database instance and ORM models.

Import `db` and call `db.create_all()` inside the Flask app context to
initialise the schema. The SQLite file is stored at config.DB_PATH.
"""
from __future__ import annotations

from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Review(db.Model):
    """Persisted customer review with ML-generated and user-confirmed labels."""

    __tablename__ = "reviews"

    id                 = db.Column(db.String(36), primary_key=True)       # UUID
    product_id         = db.Column(db.String(64), nullable=False, index=True)
    title              = db.Column(db.String(256), nullable=False)
    description        = db.Column(db.Text, nullable=False)
    rating             = db.Column(db.Integer, nullable=False)
    ai_label           = db.Column(db.String(32), nullable=False)         # RecommendLabel.value
    final_label        = db.Column(db.String(32), nullable=False)
    overridden         = db.Column(db.Boolean, nullable=False, default=False)
    is_verified_buyer  = db.Column(db.Boolean, nullable=False, default=True)
    created_at         = db.Column(
        db.DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self) -> dict:
        return {
            "review_id":         self.id,
            "product_id":        self.product_id,
            "title":             self.title,
            "description":       self.description,
            "rating":            self.rating,
            "ai_label":          self.ai_label,
            "final_label":       self.final_label,
            "overridden":        self.overridden,
            "is_verified_buyer": self.is_verified_buyer,
            "created_at":        self.created_at.isoformat() if self.created_at else None,
            "review_url":        f"/api/reviews/{self.id}",
        }
