"""
PredictService — ML inference with Task 1 preprocessing baked in.

Loads the LinearSVC model and CountVectorizer saved by
ml/train_and_export.py and exposes a single predict() method that returns
a RecommendLabel.

CRITICAL: training data is already cleaned via the Task 1 pipeline (regex
tokenisation, lowercasing, length filter, stop-word removal, top-20 doc-freq
removal, vocab restriction). Raw user-submitted text must go through the same
cleaning before vectorisation, otherwise capitalised words / punctuation /
stop-words never match the vocabulary and the model sees an effectively
empty feature vector.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import joblib

import config
from enums.label import RecommendLabel


_TOKEN_PATTERN = re.compile(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")


class PredictService:
    def __init__(self) -> None:
        if not config.MODEL_PKL_PATH.exists() or not config.VECTORIZER_PKL_PATH.exists():
            sys.exit(
                "[ERROR] ML artifacts not found. "
                "Run 'python backend/ml/train_and_export.py' first."
            )
        self._model = joblib.load(config.MODEL_PKL_PATH)
        self._vectorizer = joblib.load(config.VECTORIZER_PKL_PATH)
        self._vocab = set(self._vectorizer.vocabulary_.keys())
        self._stopwords = self._load_stopwords(config.STOPWORDS_TXT_PATH)
        print(f"[PredictService] model + vectorizer loaded "
              f"(vocab={len(self._vocab):,}, stopwords={len(self._stopwords):,})")

    @staticmethod
    def _load_stopwords(path: Path) -> set[str]:
        if not path.exists():
            return set()
        with open(path, encoding="utf-8") as f:
            return {line.strip().lower() for line in f if line.strip()}

    def _clean(self, text: str) -> str:
        """Apply Task 1 cleaning pipeline so inference matches training."""
        if not text:
            return ""
        tokens = (t.lower() for t in _TOKEN_PATTERN.findall(text))
        kept = [
            t for t in tokens
            if len(t) >= 2 and t not in self._stopwords and t in self._vocab
        ]
        return " ".join(kept)

    def predict(self, text: str) -> RecommendLabel:
        """Clean text + vectorise + predict -> RecommendLabel."""
        cleaned = self._clean(text)
        vec = self._vectorizer.transform([cleaned])
        result = int(self._model.predict(vec)[0])
        return RecommendLabel.from_int(result)
