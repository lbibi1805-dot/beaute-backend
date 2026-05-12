"""
PredictService — fused ML inference for the recommendation label.

Loads THREE base models trained by ml/train_and_export.py and combines their
"Recommended" probabilities via weighted soft voting:

    p_fused = Σ (w_i · p_i) / Σ w_i
    label   = Recommended  if p_fused >= FUSION_THRESHOLD else Not Recommended

Each base model consumes a different feature representation of the SAME cleaned
text, satisfying the "different type of data" wording in the Milestone 2 brief.
Weights are the Milestone 1 macro-F1 scores (see config.FUSION_WEIGHTS).

CRITICAL: training data is cleaned via the Task 1 pipeline (regex tokenisation,
lowercasing, length filter, stop-word removal, vocab restriction). Raw user-
submitted text MUST go through the same cleaning before vectorisation,
otherwise capitalised words / punctuation / stop-words never match the
vocabulary and the models see an effectively empty feature vector.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import joblib
import numpy as np

import config
from enums.label import RecommendLabel


_TOKEN_PATTERN = re.compile(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")
_EMBED_DIM = 300


class PredictService:
    def __init__(self) -> None:
        required = [
            config.MODEL_PKL_PATH,
            config.VECTORIZER_PKL_PATH,
            config.MODEL_UNWEIGHTED_PKL_PATH,
            config.MODEL_WEIGHTED_PKL_PATH,
            config.TFIDF_WEIGHT_VECTORIZER_PKL_PATH,
            config.FASTTEXT_VOCAB_PKL_PATH,
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            sys.exit(
                "[ERROR] ML artifacts not found: " + ", ".join(missing) +
                ". Run 'python ml/train_and_export.py' first."
            )

        # Path A — BoW
        self._bow_clf = joblib.load(config.MODEL_PKL_PATH)
        self._bow_vec = joblib.load(config.VECTORIZER_PKL_PATH)
        self._vocab = set(self._bow_vec.vocabulary_.keys())

        # Path B & C — FastText lookup
        self._fasttext_vocab: dict[str, np.ndarray] = joblib.load(config.FASTTEXT_VOCAB_PKL_PATH)

        # Path B — unweighted
        self._unw_clf = joblib.load(config.MODEL_UNWEIGHTED_PKL_PATH)

        # Path C — weighted
        self._wgt_clf = joblib.load(config.MODEL_WEIGHTED_PKL_PATH)
        self._wgt_vec = joblib.load(config.TFIDF_WEIGHT_VECTORIZER_PKL_PATH)

        self._stopwords = self._load_stopwords(config.STOPWORDS_TXT_PATH)
        self._w_bow = config.FUSION_WEIGHTS["bow"]
        self._w_unw = config.FUSION_WEIGHTS["unweighted"]
        self._w_wgt = config.FUSION_WEIGHTS["weighted"]
        self._w_sum = self._w_bow + self._w_unw + self._w_wgt

        print(
            f"[PredictService] fused predictor ready "
            f"(vocab={len(self._vocab):,}, stopwords={len(self._stopwords):,}, "
            f"fasttext_tokens={len(self._fasttext_vocab):,}, "
            f"weights bow={self._w_bow:.4f} unw={self._w_unw:.4f} wgt={self._w_wgt:.4f})"
        )

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

    def _embed_unweighted(self, tokens: list[str]) -> np.ndarray:
        """Mean of FastText rows for in-vocab tokens; zero vector if none."""
        vecs = [self._fasttext_vocab[t] for t in tokens if t in self._fasttext_vocab]
        if not vecs:
            return np.zeros(_EMBED_DIM, dtype=np.float32)
        return np.mean(np.stack(vecs, axis=0), axis=0)

    def _embed_weighted(self, cleaned_text: str) -> np.ndarray:
        """TF-IDF-weighted mean of FastText rows; zero vector if no mass."""
        tfidf_row = self._wgt_vec.transform([cleaned_text])
        if tfidf_row.nnz == 0:
            return np.zeros(_EMBED_DIM, dtype=np.float32)
        index_to_token = self._wgt_vec.get_feature_names_out()
        coo = tfidf_row.tocoo()
        weighted_sum = np.zeros(_EMBED_DIM, dtype=np.float32)
        weight_total = 0.0
        for col, weight in zip(coo.col, coo.data):
            token = index_to_token[col]
            vec = self._fasttext_vocab.get(token)
            if vec is None:
                continue
            weighted_sum += float(weight) * vec
            weight_total += float(weight)
        if weight_total == 0.0:
            return np.zeros(_EMBED_DIM, dtype=np.float32)
        return weighted_sum / weight_total

    def _proba_recommend(self, clf, X) -> float:
        """Return probability of the 'Recommended' (class 1) label."""
        proba = clf.predict_proba(X)[0]
        # classes_ may be ordered [0, 1] or [1, 0]; pick the column for class 1
        classes = list(getattr(clf, "classes_", [0, 1]))
        idx = classes.index(1) if 1 in classes else 1
        return float(proba[idx])

    def predict(self, text: str) -> RecommendLabel:
        """Clean text, run all three base models, fuse via weighted soft voting."""
        cleaned = self._clean(text)
        if not cleaned:
            # No usable signal — fall back to the dataset's majority class.
            return RecommendLabel.RECOMMEND

        # Path A: BoW
        bow_X = self._bow_vec.transform([cleaned])
        p_bow = self._proba_recommend(self._bow_clf, bow_X)

        # Path B: unweighted FastText
        unw_X = self._embed_unweighted(cleaned.split()).reshape(1, -1)
        p_unw = self._proba_recommend(self._unw_clf, unw_X)

        # Path C: weighted FastText
        wgt_X = self._embed_weighted(cleaned).reshape(1, -1)
        p_wgt = self._proba_recommend(self._wgt_clf, wgt_X)

        p_fused = (
            self._w_bow * p_bow + self._w_unw * p_unw + self._w_wgt * p_wgt
        ) / self._w_sum

        label = 1 if p_fused >= config.FUSION_THRESHOLD else 0
        return RecommendLabel.from_int(label)
