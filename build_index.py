"""
build_index.py  -  RUN THIS ONCE (in Google Colab), not inside the Streamlit app.

It reads ALL products from Supabase and the knowledge/*.md files, cuts them into chunks,
turns every chunk into an embedding, and saves everything permanently:
    index/faiss.index   -> the FAISS vector index
    index/chunks.json   -> the text + details of every chunk (same order as the index)

Run it again ONLY when you ADD new products or change a product's descriptive text
(ingredients, concerns, skin types, goals, description) or edit the knowledge files.
You do NOT need to re-run it for price, seller, purchase_url, last_verified or is_active changes:
the app reads those live from Supabase.

Needs two environment variables: SUPABASE_URL and SUPABASE_ANON_KEY.
"""
import glob
import json
import os
import re

import faiss
from sentence_transformers import SentenceTransformer

import core

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"   # free, small, runs on CPU
KNOWLEDGE_GLOB = "knowledge/*.md"
INDEX_DIR = "index"


def build_chunks(rows):
    chunks = []

    # ---------- 1) PRODUCTS: one chunk per product (active AND inactive, so you can switch
    #             is_active on/off in Supabase later without rebuilding) ----------
    for r in rows:
        meta = core.normalise_row(r)
        chunks.append({"type": "product", "text": core.product_chunk_text(meta), "meta": meta})

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
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_ANON_KEY")
    if not url or not key:
        raise SystemExit("Set SUPABASE_URL and SUPABASE_ANON_KEY first (see the Colab steps).")

    rows = core.fetch_products(url, key)
    chunks = build_chunks(rows)
    n_prod = sum(c["type"] == "product" for c in chunks)
    print(f"{n_prod} product chunks + {len(chunks) - n_prod} knowledge chunks")

    # ---------- 3) EMBED EVERYTHING (this is the only time it happens) ----------
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode([c["text"] for c in chunks], normalize_embeddings=True,
                       show_progress_bar=True).astype("float32")

    # ---------- 4) FAISS: inner product on normalised vectors = cosine similarity ----------
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    # ---------- 5) SAVE PERMANENTLY ----------
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, f"{INDEX_DIR}/faiss.index")
    with open(f"{INDEX_DIR}/chunks.json", "w", encoding="utf-8") as f:
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
