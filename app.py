"""
Flask application entry point.

Start the server:
    python app.py

Prerequisites (run once from the backend/ directory):
    python ml/train_and_export.py
"""
from __future__ import annotations

from flask import Flask
from flask_cors import CORS

import config
from database import db
from data_access.product_repository import ProductRepository
from data_access.review_repository import ReviewRepository
from services.auth_service import AuthService
from services.predict_service import PredictService
from services.review_service import ReviewService
from services.search_service import SearchService
from services.recommendation_service import RecommendationService
from services.analytics_service import AnalyticsService
from routes.auth import auth_bp
from routes.products import products_bp
from routes.reviews import reviews_bp
from routes.admin import admin_bp


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)  # allow all origins for local development

    # ── SQLAlchemy configuration ─────────────────────────────────────────────
    app.config["SQLALCHEMY_DATABASE_URI"]        = config.DB_URI
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)

    with app.app_context():
        db.create_all()   # creates tables if they do not exist yet

    # ── boot services (one instance each, shared via app.config) ────────────
    product_repo   = ProductRepository()
    review_repo    = ReviewRepository()
    auth_svc       = AuthService()
    predict_svc    = PredictService()
    review_svc     = ReviewService(review_repo, predict_svc)
    search_svc     = SearchService(product_repo)
    recommend_svc  = RecommendationService(product_repo)
    analytics_svc  = AnalyticsService(product_repo)

    app.config["SERVICES"] = {
        "product_repo":    product_repo,
        "review_repo":     review_repo,
        "auth":            auth_svc,
        "review":          review_svc,
        "search":          search_svc,
        "recommendation":  recommend_svc,
        "analytics":       analytics_svc,
    }

    # ── register blueprints ──────────────────────────────────────────────────
    app.register_blueprint(auth_bp)
    app.register_blueprint(products_bp)
    app.register_blueprint(reviews_bp)
    app.register_blueprint(admin_bp)

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(host="0.0.0.0", port=config.FLASK_PORT, debug=config.FLASK_DEBUG)


if __name__ == "__main__":
    application = create_app()
    application.run(host="0.0.0.0", port=config.FLASK_PORT, debug=config.FLASK_DEBUG)
