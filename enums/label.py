"""
RecommendLabel enum — binary recommendation label produced by the ML model.

Target methodology (matches data/task2_3.ipynb): a review is labelled
'Recommended' when its star rating is >= 4 (the spec phrase "the review
represents the recommendation of buying an item"); otherwise 'Not Recommended'.

`is_a_buyer` is used as a per-sample trust weight during training (verified
buyers contribute weight 1.0; non-buyers are down-weighted), NOT as the label.
"""
from enum import Enum


class RecommendLabel(str, Enum):
    RECOMMEND     = "Recommended"
    NOT_RECOMMEND = "Not Recommended"

    @classmethod
    def from_int(cls, value: int) -> "RecommendLabel":
        """Convert model output (1 = recommend, 0 = not recommend) to RecommendLabel."""
        return cls.RECOMMEND if value == 1 else cls.NOT_RECOMMEND

    @classmethod
    def from_rating(cls, rating: int) -> "RecommendLabel":
        """Derive the ground-truth label from a star rating (>= 4 -> Recommended)."""
        return cls.RECOMMEND if int(rating) >= 4 else cls.NOT_RECOMMEND
