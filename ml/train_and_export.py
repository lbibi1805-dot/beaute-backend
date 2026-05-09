"""
Phase 0 — one-time model training + export.

Run this script once before starting the Flask server.
Run from the backend/ directory:
    python ml/train_and_export.py

Outputs (saved in ml/):
    ml/model.pkl              — trained LogisticRegression
    ml/tfidf_vectorizer.pkl   — fitted TfidfVectorizer (must stay paired with model.pkl)
"""

import sys
from pathlib import Path

# ── resolve paths via config so they match the running backend ───────────────
BACKEND_DIR = Path(__file__).resolve().parent.parent   # backend/
sys.path.insert(0, str(BACKEND_DIR))

import config

PROCESSED_CSV  = config.PROCESSED_CSV_PATH
VOCAB_TXT      = config.VOCAB_TXT_PATH
MODEL_OUT      = config.MODEL_PKL_PATH
VECTORIZER_OUT = config.VECTORIZER_PKL_PATH

# ── imports ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
import joblib

RANDOM_STATE = 42


def load_vocab(path: Path) -> dict:
    vocab = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            word, idx = line.rsplit(":", 1)
            vocab[word] = int(idx)
    return vocab


def main():
    print("=== Milestone 2 — Model Export ===\n")

    # 1. Load cleaned reviews from Milestone 1 output
    if not PROCESSED_CSV.exists():
        sys.exit(f"[ERROR] processed.csv not found at {PROCESSED_CSV}")
    if not VOCAB_TXT.exists():
        sys.exit(f"[ERROR] vocab.txt not found at {VOCAB_TXT}")

    reviews = pd.read_csv(PROCESSED_CSV)
    reviews["review_text"] = reviews["review_text"].fillna("").astype(str)
    print(f"Loaded {len(reviews):,} reviews from processed.csv")

    # 2. Load vocabulary (identical to Task 1 vocab.txt format: word:index)
    vocab = load_vocab(VOCAB_TXT)
    print(f"Loaded {len(vocab):,} vocabulary entries from vocab.txt")

    # 3. Build TF-IDF matrix using the exact Task 1 vocabulary
    #    (same approach as task2_3.py §7)
    #    sublinear_tf=True  → 1+log(tf) scaling (more numerically stable)
    #    vocabulary=vocab   → columns align with vocab.txt indices
    vectorizer = TfidfVectorizer(
        vocabulary=vocab,
        lowercase=False,
        token_pattern=r"(?u)\S+",
        sublinear_tf=True,
        norm="l2",
    )
    X = vectorizer.fit_transform(reviews["review_text"].tolist())
    print(f"TF-IDF matrix: {X.shape}, nnz={X.nnz:,}")

    # 4. Build target label
    # is_a_buyer may be stored as bool (True/False) or string ("True"/"False")
    # depending on how processed.csv was written; handle both gracefully.
    is_buyer_col = reviews["is_a_buyer"]
    y = is_buyer_col.map(
        {True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0}
    ).fillna(0).astype(int).values
    pos_rate = y.mean()
    print(f"Class balance: is_a_buyer=True {pos_rate:.1%} / False {1 - pos_rate:.1%}")

    # 5. Quick 5-fold CV sanity check (mirrors task2_3.py Task 3 evaluation)
    print("\nRunning 5-fold CV sanity check …")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    clf_eval = LogisticRegression(
        max_iter=2000, solver="lbfgs", n_jobs=-1,
        class_weight="balanced", random_state=RANDOM_STATE,
    )
    scores = cross_validate(
        clf_eval, X, y, cv=skf,
        scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
        n_jobs=1,
    )
    print(f"  Accuracy  : {scores['test_accuracy'].mean():.4f} ± {scores['test_accuracy'].std():.4f}")
    print(f"  Macro-F1  : {scores['test_macro_f1'].mean():.4f} ± {scores['test_macro_f1'].std():.4f}")

    # 6. Train final model on the full dataset
    print("\nTraining final model on full dataset …")
    clf = LogisticRegression(
        max_iter=2000, solver="lbfgs", n_jobs=-1,
        class_weight="balanced", random_state=RANDOM_STATE,
    )
    clf.fit(X, y)
    print("Training complete.")

    # 7. Persist artifacts
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_OUT)
    joblib.dump(vectorizer, VECTORIZER_OUT)
    print(f"\nSaved model       → {MODEL_OUT}")
    print(f"Saved vectorizer  → {VECTORIZER_OUT}")
    print("\n[DONE] Run 'python backend/app.py' to start the Flask server.")


if __name__ == "__main__":
    main()
