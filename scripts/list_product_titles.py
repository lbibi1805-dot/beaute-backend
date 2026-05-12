"""
Generate a unique sorted list of product_title values for manual image sourcing.
Run from beaute-backend/:
    python scripts/list_product_titles.py
Output: data/product_titles.json
"""
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent  # beaute-backend/
sys.path.insert(0, str(BACKEND_DIR))

import config

csv_path = config.RAW_CSV_PATH
out_path = config.DATA_DIR / "product_titles.json"

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is required: pip install pandas")

df = pd.read_csv(csv_path, low_memory=False)

col = "product_title" if "product_title" in df.columns else "product_name"
titles = sorted(df[col].dropna().unique().tolist())

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(titles, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Written {len(titles)} unique product titles to {out_path}")
