"""
Milestone 2 — train + export the THREE base models used for fusion.

Mirrors the methodology in data/task2_3.py:
  - Target: binary recommendation = (review_rating >= 4)
  - Sample weights: verified buyers = 1.0; non-buyers = NON_BUYER_TRUST_WEIGHT
  - Three feature representations, each producing one classifier:
      A. Count Vectors (BoW)        -> LogisticRegression
      B. Unweighted FastText 300d   -> LogisticRegression
      C. TF-IDF-weighted FastText   -> LogisticRegression
  - All three classifiers expose predict_proba natively (no Platt calibration).
  - After CV, the measured macro-F1 per path is written to ml/fusion_weights.json
    so PredictService weights the fusion using the ACTUAL deployed models.

Run from the backend/ directory:
    python ml/train_and_export.py

Outputs (under ml/):
    model.pkl                    - Path A classifier (with predict_proba)
    tfidf_vectorizer.pkl         - Path A CountVectorizer
    model_unweighted.pkl         - Path B classifier
    model_weighted.pkl           - Path C classifier
    tfidf_weight_vectorizer.pkl  - Path C TfidfVectorizer
    fasttext_vocab.pkl           - dict[token -> np.ndarray(300)] shared by B & C
"""
import json
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import config

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

RANDOM_STATE = 42
EMBED_NAME = "fasttext-wiki-news-subwords-300"
EMBED_DIM = 300


def load_vocab(path: Path) -> dict:
    """Read vocab.txt into {word: index}."""
    vocab = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            word, idx = line.rsplit(":", 1)
            vocab[word] = int(idx)
    return vocab


def cv_evaluate(make_estimator, X, y, sample_weight, name: str) -> tuple[float, float]:
    """5-fold stratified CV; prints per-fold acc + macro-F1; returns means."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s = [], []
    print(f"\n  [{name}] 5-fold CV:")
    for fold_i, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        clf = make_estimator()
        try:
            clf.fit(X[tr], y[tr], sample_weight=sample_weight[tr])
        except TypeError:
            clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))
        print(f"    fold {fold_i}: acc={accs[-1]:.4f}  macroF1={f1s[-1]:.4f}")
    print(f"  [{name}] mean acc={np.mean(accs):.4f}  mean macroF1={np.mean(f1s):.4f}")
    return float(np.mean(accs)), float(np.mean(f1s))


def build_embed_lookup(vocab: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Build a (|V|, 300) FastText lookup matrix aligned to `vocab` indices.

    Fast path: if ml/fasttext_vocab.pkl already exists from a prior run, reuse it
    (~1s vs ~7min for the gensim download/load). Slow path: lazy-import gensim
    and load the 960MB pretrained model.
    """
    if config.FASTTEXT_VOCAB_PKL_PATH.exists():
        print(f"\nUsing cached FastText vocab: {config.FASTTEXT_VOCAB_PKL_PATH}")
        token_vec_dict: dict[str, np.ndarray] = joblib.load(config.FASTTEXT_VOCAB_PKL_PATH)
        embed_lookup = np.zeros((len(vocab), EMBED_DIM), dtype=np.float32)
        for word, idx in vocab.items():
            if word in token_vec_dict:
                embed_lookup[idx] = token_vec_dict[word]
        has_embed = np.linalg.norm(embed_lookup, axis=1) > 0
        print(f"  loaded {len(token_vec_dict):,} token vectors "
              f"(coverage {len(token_vec_dict)/len(vocab):.1%})")
        return embed_lookup, has_embed, token_vec_dict

    # Slow path — lazy-import so the script still runs if gensim is missing
    # but a cached vocab exists.
    import gensim.downloader as api
    print(f"\nLoading {EMBED_NAME} from gensim (cached at ~/gensim-data/) ...")
    t0 = time.time()
    embed_model = api.load(EMBED_NAME)
    print(f"  loaded in {time.time()-t0:.1f}s "
          f"(vocab={len(embed_model.key_to_index):,}, dim={embed_model.vector_size})")
    assert embed_model.vector_size == EMBED_DIM

    embed_lookup = np.zeros((len(vocab), EMBED_DIM), dtype=np.float32)
    token_vec_dict: dict[str, np.ndarray] = {}
    covered = 0
    for word, idx in vocab.items():
        if word in embed_model.key_to_index:
            v = np.asarray(embed_model[word], dtype=np.float32)
            embed_lookup[idx] = v
            token_vec_dict[word] = v
            covered += 1
    has_embed = np.linalg.norm(embed_lookup, axis=1) > 0
    print(f"  FastText coverage: {covered:,} / {len(vocab):,} "
          f"({covered/len(vocab):.1%}); persisted dict has {len(token_vec_dict):,} entries")
    return embed_lookup, has_embed, token_vec_dict


def main():
    print("=== Milestone 2 — Train + Export 3 base models for fusion ===\n")

    # --- common inputs --------------------------------------------------------
    if not config.PROCESSED_CSV_PATH.exists():
        sys.exit(f"[ERROR] processed.csv not found at {config.PROCESSED_CSV_PATH}")
    if not config.VOCAB_TXT_PATH.exists():
        sys.exit(f"[ERROR] vocab.txt not found at {config.VOCAB_TXT_PATH}")

    reviews = pd.read_csv(config.PROCESSED_CSV_PATH)
    reviews["review_text"] = reviews["review_text"].fillna("").astype(str)
    print(f"Loaded {len(reviews):,} reviews from processed.csv")

    vocab = load_vocab(config.VOCAB_TXT_PATH)
    print(f"Loaded {len(vocab):,} vocabulary entries from vocab.txt")

    y = (reviews["review_rating"].astype(float) >= config.RECOMMEND_RATING_THRESHOLD).astype(int).values
    pos_rate = y.mean()
    print(f"Class balance: Recommended {pos_rate:.1%} / Not Recommended {1 - pos_rate:.1%}")

    is_buyer = reviews["is_a_buyer"].map(
        {True: True, False: False, "True": True, "False": False, 1: True, 0: False}
    ).fillna(False).astype(bool).values
    sample_weight = np.where(is_buyer, 1.0, config.NON_BUYER_TRUST_WEIGHT).astype(np.float64)
    print(f"Trust weights: verified={int(is_buyer.sum()):,} (w=1.0), "
          f"non-buyer={int((~is_buyer).sum()):,} (w={config.NON_BUYER_TRUST_WEIGHT})")

    texts = reviews["review_text"].tolist()
    config.ML_DIR.mkdir(parents=True, exist_ok=True)

    measured_f1: dict[str, float] = {}

    # ============================================================
    # Path A — BoW + LogisticRegression
    # ============================================================
    print("\n--- Path A: Count Vectors + LogisticRegression ---")
    bow_vectorizer = CountVectorizer(
        vocabulary=vocab,
        lowercase=False,
        token_pattern=r"(?u)\S+",
    )
    X_bow = bow_vectorizer.fit_transform(texts)
    print(f"BoW matrix: {X_bow.shape}, nnz={X_bow.nnz:,}")

    def make_bow_clf():
        return LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    _, f1_bow = cv_evaluate(make_bow_clf, X_bow, y, sample_weight, "BoW")
    measured_f1["bow"] = f1_bow
    print("Training final BoW model on full data ...")
    bow_clf = make_bow_clf()
    bow_clf.fit(X_bow, y, sample_weight=sample_weight)
    joblib.dump(bow_clf, config.MODEL_PKL_PATH)
    joblib.dump(bow_vectorizer, config.VECTORIZER_PKL_PATH)
    print(f"  -> {config.MODEL_PKL_PATH}")
    print(f"  -> {config.VECTORIZER_PKL_PATH}")

    # ============================================================
    # FastText lookup (shared by Paths B & C)
    # ============================================================
    embed_lookup, has_embed, token_vec_dict = build_embed_lookup(vocab)
    joblib.dump(token_vec_dict, config.FASTTEXT_VOCAB_PKL_PATH)
    print(f"  -> {config.FASTTEXT_VOCAB_PKL_PATH}")

    # ============================================================
    # Path B — Unweighted FastText averaged embeddings + LogReg
    # ============================================================
    print("\n--- Path B: Unweighted FastText + LogisticRegression ---")
    # Reuse the BoW count matrix to compute mean embeddings.
    emb_token_counts = X_bow.dot(has_embed.astype(np.int32))
    sum_embed = X_bow.dot(embed_lookup)
    denom = np.where(emb_token_counts > 0, emb_token_counts, 1).reshape(-1, 1)
    X_unw = (sum_embed / denom).astype(np.float32)
    X_unw[emb_token_counts == 0] = 0.0
    n_zero_unw = int((emb_token_counts == 0).sum())
    print(f"Unweighted matrix: {X_unw.shape} "
          f"(reviews with no embedded tokens: {n_zero_unw:,})")

    def make_unw_clf():
        return LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    _, f1_unw = cv_evaluate(make_unw_clf, X_unw, y, sample_weight, "Unweighted FastText")
    measured_f1["unweighted"] = f1_unw
    print("Training final unweighted model on full data ...")
    unw_clf = make_unw_clf()
    unw_clf.fit(X_unw, y, sample_weight=sample_weight)
    joblib.dump(unw_clf, config.MODEL_UNWEIGHTED_PKL_PATH)
    print(f"  -> {config.MODEL_UNWEIGHTED_PKL_PATH}")

    # ============================================================
    # Path C — TF-IDF-weighted FastText embeddings + LogReg
    # ============================================================
    print("\n--- Path C: Weighted FastText (TF-IDF) + LogisticRegression ---")
    tfidf_vectorizer = TfidfVectorizer(
        vocabulary=vocab,
        lowercase=False,
        token_pattern=r"(?u)\S+",
        sublinear_tf=True,
        norm=None,
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    print(f"TF-IDF matrix: {tfidf_matrix.shape}, nnz={tfidf_matrix.nnz:,}")

    weighted_sum = tfidf_matrix.dot(embed_lookup)
    weights_for_embedded = tfidf_matrix.multiply(has_embed.astype(np.float32))
    weight_totals = np.asarray(weights_for_embedded.sum(axis=1)).ravel()
    denom = np.where(weight_totals > 0, weight_totals, 1).reshape(-1, 1)
    X_wgt = (weighted_sum / denom).astype(np.float32)
    X_wgt[weight_totals == 0] = 0.0
    n_zero_wgt = int((weight_totals == 0).sum())
    print(f"Weighted matrix: {X_wgt.shape} "
          f"(reviews with zero TF-IDF mass: {n_zero_wgt:,})")

    def make_wgt_clf():
        return LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    _, f1_wgt = cv_evaluate(make_wgt_clf, X_wgt, y, sample_weight, "Weighted FastText")
    measured_f1["weighted"] = f1_wgt
    print("Training final weighted model on full data ...")
    wgt_clf = make_wgt_clf()
    wgt_clf.fit(X_wgt, y, sample_weight=sample_weight)
    joblib.dump(wgt_clf, config.MODEL_WEIGHTED_PKL_PATH)
    joblib.dump(tfidf_vectorizer, config.TFIDF_WEIGHT_VECTORIZER_PKL_PATH)
    print(f"  -> {config.MODEL_WEIGHTED_PKL_PATH}")
    print(f"  -> {config.TFIDF_WEIGHT_VECTORIZER_PKL_PATH}")

    # ============================================================
    # Persist measured fusion weights -> read by config.FUSION_WEIGHTS
    # ============================================================
    weights_path = config.ML_DIR / "fusion_weights.json"
    weights_path.write_text(json.dumps(measured_f1, indent=2))
    print(f"\nFusion weights (measured macro-F1):")
    for k, v in measured_f1.items():
        print(f"  {k:<10s} = {v:.4f}")
    print(f"  -> {weights_path}")

    print("\n[DONE] All artifacts written to", config.ML_DIR)
    print("Start the Flask backend with: python app.py")


if __name__ == "__main__":
    main()
