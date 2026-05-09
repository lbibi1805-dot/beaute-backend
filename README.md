# Beauté Backend — ap4ds-a3-backend

Flask 3 REST API powering the Beauté cosmetics shopping platform.
Built for **RMIT COSC3801 / COSC3015 — Advanced Programming for Data Science, Assignment 3 Milestone 2**.

| Student | ID |
|---|---|
| Lam Dao Duc | s4019052 |
| Bach Luong Chi | s4029308 |
| Ngoc Duong Bao | s3425449 |

---

## Prerequisites

| Tool | Minimum version |
|---|---|
| Python | 3.10 |
| pip | 22+ |

---

## Project layout

```
backend/
├── app.py                   # Flask entry point — run this
├── config.py                # All paths and runtime settings
├── database.py              # SQLAlchemy instance + Review model
├── requirements.txt
├── data/                    # Bundled data assets + SQLite DB (runtime)
│   ├── cosmetics_beauty_products_reviews.csv
│   ├── processed.csv
│   ├── vocab.txt
│   └── stopwords_en.txt
├── ml/
│   ├── train_and_export.py  # One-time training script
│   ├── model.pkl            # Generated — not in git
│   └── tfidf_vectorizer.pkl # Generated — not in git
├── data_access/
│   ├── product_repository.py
│   └── review_repository.py
├── services/
│   ├── search_service.py
│   ├── predict_service.py
│   ├── review_service.py
│   └── recommendation_service.py
├── routes/
│   ├── products.py
│   └── reviews.py
├── enums/
│   └── label.py
└── docs/
    ├── context.md
    └── CodeRules.md
```

---

## Setup (first time)

### 1. Clone

```bash
git clone <repo-url>
cd backend
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the ML model

Run this **once** before starting the server. It reads `data/processed.csv` and `data/vocab.txt` and writes `ml/model.pkl` + `ml/tfidf_vectorizer.pkl`.

```bash
python ml/train_and_export.py
```

Expected output:
```
=== Milestone 2 — Model Export ===
Loaded 11,764 reviews from processed.csv
...
Saved model → ml/model.pkl
Saved vectorizer → ml/tfidf_vectorizer.pkl
```

### 5. Start the API server

```bash
python app.py
```

The server starts on **http://localhost:5000**.  
The SQLite database (`data/beaute.db`) is created automatically on first run.

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/products` | Search & filter products (`q`, `brand`, `category`, `min_price`, `max_price`) |
| `GET` | `/api/products/filters` | Distinct brands, categories, price range |
| `GET` | `/api/products/<id>` | Single product detail |
| `GET` | `/api/products/<id>/reviews` | All reviews for a product |
| `POST` | `/api/products/<id>/reviews` | Submit a review (triggers ML label prediction) |
| `GET` | `/api/products/<id>/similar` | Top-N similar products (cosine similarity) |
| `GET` | `/api/reviews/<review_id>` | Fetch a review by UUID |

### Example: search products

```bash
curl "http://localhost:5000/api/products?q=moisturizer&category=Moisturizer"
```

### Example: submit a review

```bash
curl -X POST http://localhost:5000/api/products/<product_id>/reviews \
  -H "Content-Type: application/json" \
  -d '{"title":"Great product","description":"Works really well","rating":5}'
```

Response includes `ai_label` ("Buy" or "Not Buy") and a `review_url` for direct access.

---

## Architecture

The backend follows a strict **layered architecture** (see `docs/CodeRules.md`):

```
Routes (Flask Blueprints)
  └─ Services  (business logic)
       ├─ Data Access  (repositories — SQLite via SQLAlchemy)
       └─ ML Layer     (PredictService — loads model.pkl once at startup)
```

- **No business logic in routes** — routes only parse HTTP requests and delegate.
- **Enums over magic strings** — `BuyLabel.BUY` / `BuyLabel.NOT_BUY` everywhere.
- **Single config module** — all file paths and settings in `config.py`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'flask'` | Activate your venv: `venv\Scripts\activate` |
| `FileNotFoundError: model.pkl` | Run `python ml/train_and_export.py` first |
| `FileNotFoundError: processed.csv` | Ensure `data/processed.csv` exists (it ships with the repo) |
| Port 5000 already in use | Kill the other process or change `FLASK_PORT` in `config.py` |
| Frontend gets CORS errors | Make sure Flask is running on port 5000; check `flask-cors` is installed |

---

## Re-running the development server

```bash
# (activate venv if not already active)
python app.py
```

No rebuild step is needed — Flask auto-reloads when `FLASK_DEBUG = True`.
