"""
Centralised configuration for the backend.
All file paths and runtime settings are defined here.
Never hard-code paths elsewhere — import from this module.
"""
from pathlib import Path

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

# ── ML artifacts (produced by ml/train_and_export.py) ────────────────────────
MODEL_PKL_PATH      = ML_DIR / "model.pkl"
VECTORIZER_PKL_PATH = ML_DIR / "tfidf_vectorizer.pkl"

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
FLASK_PORT  = 5000
FLASK_DEBUG = True

# ── recommendation ────────────────────────────────────────────────────────────
SIMILAR_PRODUCTS_TOP_N = 6
