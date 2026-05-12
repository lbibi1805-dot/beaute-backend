"""
ReviewService — orchestrates review creation.

Workflow:
  1. Combine title + description -> run ML prediction via PredictService.
  2. Allow caller to optionally override the predicted label.
  3. Persist the review via ReviewRepository.
  4. Return the saved record with its UUID and review_url.

App-submitted reviews default to `is_verified_buyer=True` because the user
checked out via the cart flow. Future work could derive this from a real
purchase event.
"""
from __future__ import annotations

from enums.label import RecommendLabel
from data_access.historical_review_repository import HistoricalReviewRepository
from data_access.review_repository import ReviewRepository
from services.predict_service import PredictService


class ReviewService:
    def __init__(
        self,
        review_repo: ReviewRepository,
        predict_service: PredictService,
        historical_repo: HistoricalReviewRepository | None = None,
    ) -> None:
        self._repo = review_repo
        self._predict = predict_service
        self._historical = historical_repo

    # ── public API ────────────────────────────────────────────────────────────

    def create_review(
        self,
        product_id: str,
        title: str,
        description: str,
        rating: int,
        label_override: str | None = None,
        is_verified_buyer: bool = True,
        author: str | None = None,
    ) -> dict:
        """
        Predict label, optionally override, persist, and return the full record.

        label_override — if supplied and a valid RecommendLabel value, it
                         replaces the model prediction before saving.
        """
        combined_text = f"{title} {description}".strip()
        ai_label: RecommendLabel = self._predict.predict(combined_text)

        final_label: RecommendLabel = ai_label
        if label_override:
            try:
                final_label = RecommendLabel(label_override)
            except ValueError:
                pass  # ignore invalid override values

        review = {
            "product_id":        product_id,
            "author":            author,
            "title":             title,
            "description":       description,
            "rating":            rating,
            "ai_label":          ai_label.value,
            "final_label":       final_label.value,
            "overridden":        (final_label != ai_label),
            "is_verified_buyer": bool(is_verified_buyer),
        }
        saved = self._repo.save(review)
        saved["review_url"] = f"/api/reviews/{saved['review_id']}"
        return saved

    def get_review(self, review_id: str) -> dict | None:
        if self._historical and review_id.startswith(HistoricalReviewRepository.CSV_ID_PREFIX):
            return self._historical.get_by_id(review_id)
        return self._repo.get_by_id(review_id)

    def get_reviews_for_product(
        self,
        product_id: str,
        historical_limit: int = 50,
    ) -> list[dict]:
        # Newest live (app-submitted) reviews first, then historical CSV
        # reviews — keeps user-generated content prominent while still
        # surfacing the rich historical context bundled with the dataset.
        # CSV reviews are capped to keep the product page responsive: some
        # products have hundreds of historical reviews in the dataset.
        live = self._repo.get_by_product_id(product_id)
        for r in live:
            r.setdefault("review_url", f"/api/reviews/{r['review_id']}")
            r.setdefault("source", "live")

        if self._historical is None:
            return live
        historical = self._historical.get_by_product_id(product_id)
        if historical_limit and historical_limit > 0:
            historical = historical[:historical_limit]
        return live + historical
