"""
Centralised configuration for the backend.
All file paths and runtime settings are defined here.
Never hard-code paths elsewhere — import from this module.
"""
from pathlib import Path
import os;

# ── directory layout ─────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent          # backend/
DATA_DIR   = BASE_DIR / "data"
ML_DIR     = BASE_DIR / "ml"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Data assets (bundled inside backend/data/) ───────────────────────────────
RAW_CSV_PATH       = DATA_DIR / "cosmetics_beauty_products_reviews.csv"
PROCESSED_CSV_PATH = DATA_DIR / "processed.csv"
VOCAB_TXT_PATH     = DATA_DIR / "vocab.txt"
STOPWORDS_TXT_PATH = DATA_DIR / "stopwords_en.txt"
PRODUCT_IMAGES_JSON_PATH = DATA_DIR / "product_images.json"

# ── ML artifacts (produced by ml/train_and_export.py) ────────────────────────
# Path A: Count Vectors (BoW) + CalibratedClassifierCV(LinearSVC)
MODEL_PKL_PATH      = ML_DIR / "model.pkl"
VECTORIZER_PKL_PATH = ML_DIR / "tfidf_vectorizer.pkl"

# Path B: Unweighted FastText averaged embeddings + LogisticRegression
MODEL_UNWEIGHTED_PKL_PATH = ML_DIR / "model_unweighted.pkl"

# Path C: TF-IDF-weighted FastText embeddings + LogisticRegression
MODEL_WEIGHTED_PKL_PATH          = ML_DIR / "model_weighted.pkl"
TFIDF_WEIGHT_VECTORIZER_PKL_PATH = ML_DIR / "tfidf_weight_vectorizer.pkl"

# FastText per-token lookup table — shared by both embedding paths.
# dict[str, np.ndarray(300,)] restricted to the 8,054 tokens in vocab.txt.
FASTTEXT_VOCAB_PKL_PATH = ML_DIR / "fasttext_vocab.pkl"

# ── Fusion (weighted soft voting) ────────────────────────────────────────────
# Weights are 5-fold CV macro-F1 scores per representation. ml/train_and_export.py
# measures them on the deployed classifiers and writes them to fusion_weights.json
# so PredictService always weights with the ACTUAL deployed models. If the JSON is
# missing (fresh checkout, training hasn't run), fall back to the M1 numbers below.
_FUSION_WEIGHTS_JSON = ML_DIR / "fusion_weights.json"
_FUSION_WEIGHTS_DEFAULT = {
    "bow":        0.6996,  # M1: LogReg + Count Vectors
    "unweighted": 0.6491,  # M1: LogReg + unweighted FastText
    "weighted":   0.6428,  # M1: LogReg + TF-IDF-weighted FastText
}
if _FUSION_WEIGHTS_JSON.exists():
    import json
    try:
        FUSION_WEIGHTS = json.loads(_FUSION_WEIGHTS_JSON.read_text())
    except (json.JSONDecodeError, OSError):
        FUSION_WEIGHTS = _FUSION_WEIGHTS_DEFAULT.copy()
else:
    FUSION_WEIGHTS = _FUSION_WEIGHTS_DEFAULT.copy()
FUSION_THRESHOLD = 0.5

# ── SQLite database ──────────────────────────────────────────────────────────
DB_PATH     = DATA_DIR / "beaute.db"
DB_URI      = f"sqlite:///{DB_PATH}"

# ── Auth credentials (demo-grade, fixed in config) ───────────────────────────
AUTH_SECRET          = "beaute-demo-secret-2026"
ADMIN_USERNAME       = "admin"
ADMIN_PASSWORD       = "admin123"
CUSTOMER_USERNAME    = "customer"
CUSTOMER_PASSWORD    = "customer123"

# ── Flask settings ────────────────────────────────────────────────────────────
# FLASK_PORT  = 5000
# FLASK_DEBUG = True
FLASK_PORT = int(os.getenv("PORT", "5000"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"

# ── recommendation ────────────────────────────────────────────────────────────
SIMILAR_PRODUCTS_TOP_N = 6
COOCCURRING_PRODUCTS_TOP_N = 6

# ── ML trust weighting (matches notebook task2_3.ipynb §13) ──────────────────
# Reviews where is_a_buyer == False are down-weighted to this value during
# training and analytics; verified buyers contribute weight 1.0.
NON_BUYER_TRUST_WEIGHT = 0.3

# ── Recommendation target ────────────────────────────────────────────────────
# A review is "Recommended" if review_rating >= RECOMMEND_RATING_THRESHOLD
RECOMMEND_RATING_THRESHOLD = 4

# ── Price tier buckets (USD; matches notebook §15.5) ─────────────────────────
PRICE_TIERS = [
    ("Budget (<$500)",     0,     500),
    ("Mid ($500-$1500)",   500,   1500),
    ("Premium (>$1500)",   1500,  float("inf")),
]
