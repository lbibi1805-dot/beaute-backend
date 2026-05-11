"""
Phase 0 — one-time model training + export.

Methodology (mirrors data/task2_3.ipynb):
  - Target: binary recommendation = (review_rating >= 4)
  - Sample weights: verified buyers = 1.0; non-buyers = NON_BUYER_TRUST_WEIGHT
    (down-weighted to reduce influence of potential seeding/bot reviews)
  - Pipeline: BoW (Task 1 vocabulary) -> LinearSVC

Run from the backend/ directory:
    python ml/train_and_export.py

Outputs (saved in ml/):
    ml/model.pkl              - trained LinearSVC
    ml/tfidf_vectorizer.pkl   - fitted CountVectorizer (paired with model.pkl)
"""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent   # backend/
sys.path.insert(0, str(BACKEND_DIR))

import config

PROCESSED_CSV  = config.PROCESSED_CSV_PATH
VOCAB_TXT      = config.VOCAB_TXT_PATH
MODEL_OUT      = config.MODEL_PKL_PATH
VECTORIZER_OUT = config.VECTORIZER_PKL_PATH

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
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
    print("=== Milestone 2 — Model Export (recommendation target) ===\n")

    if not PROCESSED_CSV.exists():
        sys.exit(f"[ERROR] processed.csv not found at {PROCESSED_CSV}")
    if not VOCAB_TXT.exists():
        sys.exit(f"[ERROR] vocab.txt not found at {VOCAB_TXT}")

    reviews = pd.read_csv(PROCESSED_CSV)
    reviews["review_text"] = reviews["review_text"].fillna("").astype(str)
    print(f"Loaded {len(reviews):,} reviews from processed.csv")

    vocab = load_vocab(VOCAB_TXT)
    print(f"Loaded {len(vocab):,} vocabulary entries from vocab.txt")

    # BoW over Task 1 vocabulary (same settings as notebook Task 2, CountVectorizer)
    vectorizer = CountVectorizer(
        vocabulary=vocab,
        lowercase=False,
        token_pattern=r"(?u)\S+",
    )
    X = vectorizer.fit_transform(reviews["review_text"].tolist())
    print(f"BoW matrix: {X.shape}, nnz={X.nnz:,}")

    # Target: rating >= threshold -> 1 (Recommended)
    y = (reviews["review_rating"].astype(float) >= config.RECOMMEND_RATING_THRESHOLD).astype(int).values
    pos_rate = y.mean()
    print(f"Class balance: Recommended {pos_rate:.1%} / Not Recommended {1 - pos_rate:.1%}")

    # Trust weights from is_a_buyer (handles bool / string variants from processed.csv)
    is_buyer = reviews["is_a_buyer"].map(
        {True: True, False: False, "True": True, "False": False, 1: True, 0: False}
    ).fillna(False).astype(bool).values
    sample_weight = np.where(is_buyer, 1.0, config.NON_BUYER_TRUST_WEIGHT)
    print(f"Trust weights: verified={int(is_buyer.sum()):,} (w=1.0), "
          f"non-buyer={int((~is_buyer).sum()):,} (w={config.NON_BUYER_TRUST_WEIGHT})")
    print(f"Effective weighted sample size: {sample_weight.sum():.1f} / {len(y):,}")

    # 5-fold stratified CV with sample_weight (manual loop for clarity)
    print("\nRunning 5-fold CV sanity check ...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s = [], []
    for fold_i, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        clf = LinearSVC(
            max_iter=5000,
            class_weight="balanced", random_state=RANDOM_STATE,
        )
        clf.fit(X[tr], y[tr], sample_weight=sample_weight[tr])
        pred = clf.predict(X[te])
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))
        print(f"  fold {fold_i}: acc={accs[-1]:.4f}  macroF1={f1s[-1]:.4f}")

    print(f"\n  Accuracy  : {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"  Macro-F1  : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

    # Train final model on the full dataset
    print("\nTraining final model on full dataset ...")
    clf = LinearSVC(
        max_iter=5000,
        class_weight="balanced", random_state=RANDOM_STATE,
    )
    clf.fit(X, y, sample_weight=sample_weight)
    print("Training complete.")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_OUT)
    joblib.dump(vectorizer, VECTORIZER_OUT)
    print(f"\nSaved model       -> {MODEL_OUT}")
    print(f"Saved vectorizer  -> {VECTORIZER_OUT}")
    print("\n[DONE] Run 'python backend/app.py' to start the Flask server.")


if __name__ == "__main__":
    main()
