"""
PredictService — Task 2 ML inference.

Loads the LogisticRegression model and TfidfVectorizer saved by
ml/train_and_export.py and exposes a single predict() method.
Both artifacts are loaded once at startup (singleton pattern via module-level
initialisation in app.py).
"""
from __future__ import annotations

import sys

import joblib

import config
from enums.label import BuyLabel


class PredictService:
    def __init__(self) -> None:
        if not config.MODEL_PKL_PATH.exists() or not config.VECTORIZER_PKL_PATH.exists():
            sys.exit(
                "[ERROR] ML artifacts not found. "
                "Run 'python backend/ml/train_and_export.py' first."
            )
        self._model = joblib.load(config.MODEL_PKL_PATH)
        self._vectorizer = joblib.load(config.VECTORIZER_PKL_PATH)
        print("[PredictService] model and vectorizer loaded")

    def predict(self, text: str) -> BuyLabel:
        """
        Vectorise text and return a BuyLabel prediction.
        text should be the review_title + ' ' + review_description.
        """
        vec = self._vectorizer.transform([text])
        result = int(self._model.predict(vec)[0])
        return BuyLabel.from_int(result)
