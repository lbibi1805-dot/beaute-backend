"""
Build data/product_images.json — a {product_id: image_url} map used by
ProductRepository._load_image_map().

Strategy
--------
Direct scraping of the source retailer is blocked, so we fetch real
cosmetics images from the Openverse public API (https://api.openverse.org)
which aggregates Creative-Commons images from Flickr and other sources.

For each product we:
  1. Detect a category keyword from the title (lipstick, shampoo, mascara…)
  2. Query Openverse once per category and cache the top results
  3. Deterministically assign each product an image from its category bucket
     based on hash(product_id) so the same product always gets the same image.

Run from beaute-backend/:
    python scripts/scrape_product_images.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import pandas as pd

import config

OPENVERSE_API = "https://api.openverse.org/v1/images/"
RESULTS_PER_CATEGORY = 8
REQUEST_TIMEOUT = 20
USER_AGENT = "beaute-demo/1.0 (educational use)"

# Ordered: longer/more-specific keywords come first so they match before
# shorter substrings (e.g. "lip liner" before "liner").
CATEGORY_KEYWORDS: list[tuple[str, str]] = [
    # specific cosmetics types
    ("lip liner",     "lip liner pencil cosmetic"),
    ("lip balm",      "lip balm cosmetic"),
    ("lip crayon",    "lip crayon cosmetic"),
    ("lip gloss",     "lip gloss cosmetic"),
    ("lipstick",      "lipstick cosmetic"),
    ("mascara",       "mascara cosmetic"),
    ("eyeliner",      "eyeliner cosmetic"),
    ("eye pencil",    "eye pencil cosmetic"),
    ("eyeshadow",     "eyeshadow palette"),
    ("eye shadow",    "eyeshadow palette"),
    ("kajal",         "kajal kohl eyeliner"),
    ("kohl",          "kohl kajal"),
    ("brow",          "eyebrow pencil cosmetic"),
    ("eyebrow",       "eyebrow pencil cosmetic"),
    ("false eyelash", "false eyelashes"),
    ("eyelash",       "eyelashes"),

    # face products
    ("foundation",    "foundation makeup bottle"),
    ("compact",       "compact powder makeup"),
    ("concealer",     "concealer makeup"),
    ("highlighter",   "highlighter makeup"),
    ("illuminator",   "illuminator makeup"),
    ("blusher",       "blush makeup"),
    ("blush",         "blush makeup"),
    ("corrector",     "color corrector makeup"),
    ("contour",       "contour stick makeup"),
    ("primer",        "makeup primer"),
    ("bb cream",      "bb cream"),
    ("cushion",       "cushion compact"),
    ("powder",        "loose powder makeup"),

    # skincare
    ("face wash",     "face wash skincare"),
    ("cleanser",      "facial cleanser skincare"),
    ("sunscreen",     "sunscreen lotion"),
    ("sun lotion",    "sunscreen lotion"),
    ("moisturiser",   "moisturizer cream"),
    ("moisturizer",   "moisturizer cream"),
    ("cream",         "face cream cosmetic"),
    ("serum",         "serum skincare bottle"),
    ("oil",           "facial oil bottle"),
    ("face mask",     "face mask sheet skincare"),
    ("mask",          "face mask skincare"),
    ("lotion",        "body lotion bottle"),
    ("mist",          "facial mist spray"),
    ("regime",        "skincare routine bottles"),
    ("scrub",         "face scrub skincare"),

    # hair
    ("shampoo",       "shampoo bottle"),
    ("conditioner",   "hair conditioner bottle"),
    ("hair color",    "hair dye color"),
    ("hair colour",   "hair dye color"),
    ("hair oil",      "hair oil bottle"),

    # nails
    ("nail enamel",   "nail polish bottle"),
    ("nail polish",   "nail polish bottle"),
    ("nail color",    "nail polish bottle"),
    ("nail lacquer",  "nail polish bottle"),

    # fragrance / misc
    ("perfume",       "perfume bottle"),
    ("fragrance",     "perfume bottle"),
    ("spray",         "spray bottle cosmetic"),

    # tools / accessories
    ("brush",         "makeup brush"),
    ("roller",        "face roller jade"),
    ("ice globe",     "facial ice globes"),
    ("massager",      "face massager"),
    ("comb",          "hair comb"),
    ("sharpener",     "cosmetic pencil sharpener"),
    ("mask with face shield", "face shield mask"),
    ("face mask",     "face mask"),
    ("face shield",   "face shield"),

    # palettes & multi-products
    ("palette",       "makeup palette"),

    # bundles
    ("combo",         "cosmetics gift set"),
    ("kit",           "cosmetics kit"),
    ("gift box",      "cosmetics gift set"),
    ("box",           "cosmetics gift set"),
    ("pouch",         "makeup pouch bag"),
    ("bag",           "makeup pouch bag"),
    ("collection",    "cosmetics collection"),
    ("duo",           "cosmetics duo"),
    ("pack",          "cosmetics pack"),
]

DEFAULT_CATEGORY = ("cosmetics", "cosmetics makeup")


def detect_category(title: str) -> tuple[str, str]:
    t = title.lower()
    for keyword, query in CATEGORY_KEYWORDS:
        if keyword in t:
            return keyword, query
    return DEFAULT_CATEGORY


def fetch_openverse(query: str, per_page: int = RESULTS_PER_CATEGORY,
                    retries: int = 2) -> list[str]:
    params = urllib.parse.urlencode({
        "q":          query,
        "per_page":   per_page,
        "license":    "cc0,by,by-sa",
        "extension":  "jpg,jpeg,png",
    })
    url = f"{OPENVERSE_API}?{params}"
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return [r["url"] for r in data.get("results", []) if r.get("url")]
        except Exception as exc:
            last_exc = exc
            time.sleep(0.7 * (attempt + 1))
    if last_exc:
        raise last_exc
    return []


def pick_for_product(product_id: str, urls: list[str]) -> str:
    if not urls:
        return ""
    idx = int(hashlib.md5(product_id.encode()).hexdigest(), 16) % len(urls)
    return urls[idx]


def main() -> None:
    df = pd.read_csv(config.RAW_CSV_PATH, low_memory=False)
    col_title = "product_title" if "product_title" in df.columns else "product_name"

    products_df = df[["product_id", col_title]].drop_duplicates(subset=["product_id"])
    products_df = products_df.dropna(subset=["product_id", col_title])

    print(f"Categorising {len(products_df)} unique products…")
    by_category: dict[str, str] = {}             # keyword -> query
    product_to_category: dict[str, str] = {}     # product_id -> keyword
    for _, row in products_df.iterrows():
        keyword, query = detect_category(str(row[col_title]))
        by_category[keyword] = query
        product_to_category[str(row["product_id"])] = keyword

    # Always fetch a generic "cosmetics" bucket up front so we can fall back
    # to it when a more specific query returns nothing.
    print("  Pre-fetching fallback 'cosmetics' bucket…")
    fallback_urls: list[str] = []
    for fallback_query in ("cosmetics makeup", "makeup", "beauty product"):
        try:
            fallback_urls = fetch_openverse(fallback_query, per_page=20)
            if fallback_urls:
                break
        except Exception as exc:
            print(f"      WARN: fallback {fallback_query!r}: {exc}")
        time.sleep(0.5)
    print(f"      fallback bucket: {len(fallback_urls)} urls")

    category_images: dict[str, list[str]] = {}
    for i, (keyword, query) in enumerate(sorted(by_category.items()), 1):
        print(f"  [{i:>2}/{len(by_category)}] {keyword!r:<24} -> Openverse query {query!r}")
        try:
            urls = fetch_openverse(query)
        except Exception as exc:
            print(f"      WARN: {exc}")
            urls = []
        if not urls:
            print(f"      WARN: no results for {keyword!r} — using fallback bucket")
            urls = fallback_urls
        category_images[keyword] = urls
        time.sleep(0.3)  # gentle rate limit

    items: list[dict] = []
    missing = 0
    for product_id, keyword in product_to_category.items():
        urls = category_images.get(keyword, []) or fallback_urls
        url = pick_for_product(product_id, urls)
        if not url:
            missing += 1
            continue
        items.append({"product_id": product_id, "image_url": url, "category": keyword})

    out_path = config.PRODUCT_IMAGES_JSON_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(items)} entries to {out_path} "
          f"(missing: {missing})")


if __name__ == "__main__":
    main()
