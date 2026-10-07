"""
build_index.py  -  RUN THIS ONCE (in Google Colab), not inside the Streamlit app.

It reads the products from your Supabase database, adds the knowledge/*.md files,
cuts everything into chunks, turns every chunk into an embedding, and saves it permanently:
    index/faiss.index   -> the FAISS vector index
    index/chunks.json   -> the text + details of every chunk (same order as the index)

Before running, set these two environment variables in a SEPARATE Colab cell (do NOT type them in this file):
    SUPABASE_URL        e.g. https://abcd1234.supabase.co
    SUPABASE_ANON_KEY   your PUBLISHABLE / anon key (never the secret key)
Then run this file with:   !python build_index.py

Run it again ONLY when you add products, rename products, or change ingredients/descriptions
or the knowledge files.  Prices, links, sellers and community insights are read LIVE by the
app and never need a rebuild.
"""
import glob
import json
import os
import re
from collections import Counter
from pathlib import Path

import faiss

import core

# Folder of this script. If the code is pasted into a Colab cell (no file), use the current folder instead.
BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"   # free, small, runs on CPU
KNOWLEDGE_GLOB = str(BASE / "knowledge" / "*.md")
INDEX_DIR = BASE / "index"


def parse_list(value):
    """Supabase text[] columns arrive as real lists. This also handles {"a","b"} text just in case."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return []
    quoted = re.findall(r'"([^"]+)"', s)
    if quoted:
        return [q.strip() for q in quoted]
    return [p.strip() for p in s.strip("{}").split(",") if p.strip()]


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalise_origin(value):
    return "Pakistan" if str(value).strip().lower() in ("pakistan", "pakistani", "local", "pk") else "Imported"


def dedupe_products(rows, source_rows=(), insight_rows=()):
    """The same product can exist twice in the table (e.g. imported twice). Keep ONE per brand+name:
    the copy that already has seller links / community insights, otherwise the oldest one.
    The ids of the dropped copies are remembered so their links and insights still show up."""
    used = Counter()
    for r in list(source_rows) + list(insight_rows):
        used[str(r.get("product_id"))] += 1
    groups = {}
    for r in rows:
        key = (str(r.get("brand") or "").strip().lower(), str(r.get("name") or "").strip().lower())
        groups.setdefault(key, []).append(r)
    kept = []
    for items in groups.values():
        items.sort(key=lambda r: (-used[str(r["id"])], str(r.get("created_at") or "9999"), str(r["id"])))
        keeper = dict(items[0])
        keeper["_aliases"] = [str(r["id"]) for r in items[1:]]
        kept.append(keeper)
    return kept


def build_chunks(rows):
    chunks = []

    # ---------- 1) PRODUCTS: one chunk per product ----------
    for r in sorted(rows, key=lambda r: (str(r.get("brand", "")), str(r.get("name", "")))):
        pid = str(r["id"])
        meta = {
            "product_id": pid,                      # the Supabase id - links this index to the live tables
            "ref": pid[:8],                         # short id the LLM sees (easier for it to copy)
            "name": str(r.get("name") or "").strip(),
            "brand": str(r.get("brand") or "").strip(),
            "category": str(r.get("category") or "").strip(),
            "product_type": str(r.get("product_type") or "").strip(),
            "origin": normalise_origin(r.get("origin")),
            "price": to_float(r.get("price")),
            "currency": str(r.get("currency") or "PKR").strip(),
            "description": str(r.get("description") or "").strip(),
            "skin_types": parse_list(r.get("skin_types")),
            "concerns": parse_list(r.get("concerns")),
            "goals": parse_list(r.get("goals")),
            "key_ingredients": parse_list(r.get("key_ingredients")),
            "sensitivity_level": str(r.get("sensitivity_level") or "").strip(),
            "prescription_required": bool(r.get("prescription_required")),
            "last_verified": str(r.get("last_verified") or "").strip(),
            "is_active": r.get("is_active") is not False,
            "alias_ids": r.get("_aliases", []),      # ids of duplicate rows that were merged into this one
            # filled live from product_sources / retailers by the app
            "seller": "", "purchase_url": "", "verification_status": "",
        }
        text = (
            f"{meta['name']} by {meta['brand']}. "
            f"Category: {meta['category']} ({meta['product_type']}). "
            f"Origin: {meta['origin']}. "
            f"Key ingredients: {', '.join(meta['key_ingredients'])}. "
            f"Suitable skin types: {', '.join(meta['skin_types'])}. "
            f"Helps with: {', '.join(meta['concerns'])}. "
            f"Supports goals: {', '.join(meta['goals'])}. "
            f"Sensitivity level: {meta['sensitivity_level']}. "
            f"{meta['description']}"
        )
        chunks.append({"type": "product", "text": text, "meta": meta})

    # ---------- 2) KNOWLEDGE: one chunk per '## heading' section ----------
    for path in sorted(glob.glob(KNOWLEDGE_GLOB)):
        raw = open(path, encoding="utf-8").read()
        for section in re.split(r"\n(?=## )", raw):
            section = section.strip()
            if section.startswith("## ") and len(section) > 40:
                chunks.append({"type": "knowledge", "text": section,
                               "meta": {"source": os.path.basename(path)}})
    return chunks


def main():
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = (os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_KEY") or "").strip()
    if not url or not key:
        raise SystemExit("Set SUPABASE_URL and SUPABASE_ANON_KEY first, in a separate Colab cell:\n"
                         "  import os\n"
                         '  os.environ["SUPABASE_URL"] = "https://YOUR-PROJECT.supabase.co"\n'
                         '  os.environ["SUPABASE_ANON_KEY"] = "YOUR_PUBLISHABLE_KEY"')

    from sentence_transformers import SentenceTransformer     # imported here so the file can be tested without it

    rows = core.supabase_get(url, key, "products")
    if not rows:
        raise SystemExit("Supabase returned 0 products. Most likely the 'products' table has no public "
                         "READ policy (run supabase_policies.sql in the Supabase SQL Editor), "
                         "or the URL/key is wrong.")

    def optional(table):
        try:
            return core.supabase_get(url, key, table)
        except Exception:
            return []

    before = len(rows)
    rows = dedupe_products(rows, optional("product_sources"), optional("reddit_insights"))
    if len(rows) != before:
        print(f"Found {before} rows but only {len(rows)} different products: "
              f"{before - len(rows)} duplicate rows were merged.")

    chunks = build_chunks(rows)
    prods = [c["meta"] for c in chunks if c["type"] == "product"]
    n_know = len(chunks) - len(prods)
    print(f"{len(prods)} product chunks + {n_know} knowledge chunks")
    print("Categories:", dict(Counter(m["category"] for m in prods)))
    print("Origins:   ", dict(Counter(m["origin"] for m in prods)))
    unmapped = sorted({m["category"] for m in prods if core.bucket_of(m["category"]) is None})
    if unmapped:
        print("NOTE: these categories are not used for routines (fine for toners etc.):", unmapped)

    # ---------- 3) EMBED EVERYTHING (this is the only time it happens) ----------
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode([c["text"] for c in chunks], normalize_embeddings=True,
                       show_progress_bar=True).astype("float32")

    # ---------- 4) FAISS: inner product on normalised vectors = cosine similarity ----------
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    # ---------- 5) SAVE PERMANENTLY ----------
    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump({"model": MODEL_NAME, "chunks": chunks}, f, ensure_ascii=False)
    print("Saved index/faiss.index and index/chunks.json")

    # ---------- 6) QUICK SELF-TEST ----------
    q = model.encode(["oily acne-prone skin needs a lightweight sunscreen"],
                     normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, 3)
    print("\nSelf-test (top 3 for 'oily acne-prone skin needs a lightweight sunscreen'):")
    for s, i in zip(scores[0], ids[0]):
        print(f"  {s:.3f}  {chunks[i]['text'][:90]}...")


if __name__ == "__main__":
    main()
