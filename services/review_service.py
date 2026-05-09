"""
ReviewService — orchestrates review creation (Task 2).

Workflow:
  1. Combine title + description → run ML prediction via PredictService.
  2. Allow caller to optionally override the predicted label.
  3. Persist the review via ReviewRepository.
  4. Return the saved record with its UUID and review_url.
"""
from __future__ import annotations

from enums.label import BuyLabel
from data_access.review_repository import ReviewRepository
from services.predict_service import PredictService


class ReviewService:
    def __init__(
        self,
        review_repo: ReviewRepository,
        predict_service: PredictService,
    ) -> None:
        self._repo = review_repo
        self._predict = predict_service

    # ── public API ────────────────────────────────────────────────────────────

    def create_review(
        self,
        product_id: str,
        title: str,
        description: str,
        rating: int,
        label_override: str | None = None,
    ) -> dict:
        """
        Predict label, optionally override, persist, and return the full record.

        label_override — if supplied and a valid BuyLabel value, it replaces
                         the model prediction before saving.
        """
        combined_text = f"{title} {description}".strip()
        ai_label: BuyLabel = self._predict.predict(combined_text)

        # Honour explicit override if it is a valid BuyLabel
        final_label: BuyLabel = ai_label
        if label_override:
            try:
                final_label = BuyLabel(label_override)
            except ValueError:
                pass  # ignore invalid override values

        review = {
            "product_id":   product_id,
            "title":        title,
            "description":  description,
            "rating":       rating,
            "ai_label":     ai_label.value,
            "final_label":  final_label.value,
            "overridden":   (final_label != ai_label),
        }
        saved = self._repo.save(review)
        saved["review_url"] = f"/api/reviews/{saved['review_id']}"
        return saved

    def get_review(self, review_id: str) -> dict | None:
        return self._repo.get_by_id(review_id)

    def get_reviews_for_product(self, product_id: str) -> list[dict]:
        reviews = self._repo.get_by_product_id(product_id)
        for r in reviews:
            r.setdefault("review_url", f"/api/reviews/{r['review_id']}")
        return reviews
