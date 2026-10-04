"""
core.py  -  all the logic (no Streamlit code here).

HOW DATA FLOWS
  * FAISS index (built ONCE by build_index.py, saved on disk)  -> used only to RANK products.
  * Supabase (read live, cached ~60 s by app.py)               -> source of truth for everything
    shown or filtered: name, price, seller, purchase_url, category, origin, ingredients,
    is_active, last_verified.
  So: change a price/link/ingredient/is_active in Supabase -> the app reflects it within a minute,
  and nothing is ever re-embedded per query. Only the user's short profile text is embedded at runtime.
"""
import json
import re

import faiss
import requests
from rapidfuzz import fuzz

GROQ_MODEL = "llama-3.3-70b-versatile"      # fallback: "llama-3.1-8b-instant"

# product category (from Supabase) -> routine "bucket"
BUCKETS = {
    "cleanser": ["Cleanser"],
    "moisturizer": ["Moisturizer"],
    "sunscreen": ["Sunscreen"],
    "treatment": ["Serum", "Essence", "Focused treatment", "Pharmacy"],
}
PER_BUCKET = {"cleanser": 4, "moisturizer": 4, "sunscreen": 4, "treatment": 6}

# "One specific product" mode
SINGLE_CHOICES = {
    "Cleanser": "cleanser",
    "Moisturizer": "moisturizer",
    "Sunscreen": "sunscreen",
    "Treatment (serum / essence / pharmacy)": "treatment",
}
SINGLE_LIMIT = 8          # candidates sent to the LLM in single-product mode (it picks up to 3)

# Dropdown labels that count as "severe"
ACNE_SEVERE = "Severe / Persistent Concern"
REDNESS_SEVERE = "Severe"
IRRITATION_SEVERE = "Severe"

DERM_MESSAGE = "You should consult a dermatologist."
SPECIALIST_MESSAGE = "Please check with your specialist before using products."
PRESCRIPTION_BLOCKS_ROUTINE = True   # True: show only the specialist message. False: show it AND the routine.

# Medicines: still in the index (so we can recognise them), but NEVER recommended.
BLOCKED_RE = re.compile(
    r"tretinoin|clindamycin|isotretinoin|hydroquinone|clobetasol|betamethasone|mometasone",
    re.I)
# Strong actives: removed for very sensitive / severe cases, and counted for warnings.
STRONG_RE = re.compile(
    r"\b(retinol|retinal\w*|adapalene|benzoyl peroxide|glycolic|salicylic|aha|bha|lactic|mandelic)\b",
    re.I)
# Extra actives avoided while pregnant (on top of STRONG_RE and BLOCKED_RE).
PREGNANCY_AVOID_RE = re.compile(
    r"retinol|retinal|retinoid|adapalene|tretinoin|salicylic|benzoyl|glycolic|lactic|mandelic|"
    r"\bbha\b|\baha\b|hydroquinone|kojic|arbutin|ascorbic|vitamin c|azelaic", re.I)

ESCALATION_WORDS = ["swelling", "swollen", "pus", "allergic", "allergy", "open wound", "bleeding",
                    "rapidly", "getting worse", "spreading", "eye", "eyelid", "blister", "fever"]
ALLERGY_SEVERE_RE = re.compile(
    r"\b(swell\w*|swollen|anaphyla\w*|hives|breath\w*|throat|blister\w*|bleed\w*|pus|eyelid\w*|fever)\b", re.I)

PREGNANCY_RE = re.compile(
    r"\b(pregnan\w*|breast-?feeding|lactating|trimester|nursing (mother|a baby)|"
    r"(i am|i'm|im) expecting|expecting a baby|expectant)\b", re.I)
PRESCRIPTION_RE = re.compile(r"prescri|\bderm(atologist)?s?\b|\bmy doctor\b", re.I)

VERDICTS = ["Keep", "Review", "Consider Replacing", "Unknown"]


# ------------------------------------------------------------------ Supabase (read-only, REST)
def fetch_products(url, key, timeout=15):
    """Reads the whole `products` table from Supabase (anon key + public read policy)."""
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    rows, offset, page = [], 0, 1000
    while True:
        r = requests.get(f"{url.rstrip('/')}/rest/v1/products", headers=headers, timeout=timeout,
                         params={"select": "*", "order": "id.asc", "limit": page, "offset": offset})
        r.raise_for_status()
        batch = r.json()
        rows.extend(batch)
        if len(batch) < page:
            break
        offset += page
    return rows


def _list(v):
    if isinstance(v, list):
        return [str(x) for x in v]
    return []


def normalise_row(r):
    """One Supabase row -> the 'meta' dict used everywhere. product_id is stable (P + database id)."""
    price = r.get("price")
    try:
        price = float(price) if price not in (None, "") else None
    except (TypeError, ValueError):
        price = None
    return {
        "product_id": f"P{int(r['id']):03d}",
        "name": str(r.get("name") or "").strip(),
        "brand": str(r.get("brand") or "").strip(),
        "category": str(r.get("category") or "").strip(),
        "product_type": str(r.get("product_type") or "").strip(),
        "origin": str(r.get("origin") or "").strip(),
        "price": price,
        "currency": str(r.get("currency") or "PKR").strip(),
        "description": str(r.get("description") or "").strip(),
        "skin_types": _list(r.get("skin_types")),
        "concerns": _list(r.get("concerns")),
        "goals": _list(r.get("goals")),
        "key_ingredients": _list(r.get("key_ingredients")),
        "sensitivity_level": str(r.get("sensitivity_level") or "").strip(),
        "last_verified": str(r.get("last_verified") or "").strip(),
        "seller": str(r.get("seller") or "").strip(),
        "purchase_url": str(r.get("purchase_url") or "").strip(),
        "is_active": bool(r.get("is_active", True)),
    }


def live_products(rows):
    """{product_id: meta} for ACTIVE products only (is_active = false in Supabase hides a product)."""
    out = {}
    for r in rows:
        m = normalise_row(r)
        if m["is_active"]:
            out[m["product_id"]] = m
    return out


def snapshot_products(chunks):
    """Fallback if Supabase is unreachable: the product data saved inside the index at build time."""
    return {c["meta"]["product_id"]: c["meta"] for c in chunks
            if c["type"] == "product" and c["meta"].get("is_active", True)}


def product_chunk_text(meta):
    """The text that gets embedded (used by build_index.py)."""
    return (
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


# ------------------------------------------------------------------ loading the saved index
def load_index(index_path="index/faiss.index", chunks_path="index/chunks.json"):
    index = faiss.read_index(index_path)
    with open(chunks_path, encoding="utf-8") as f:
        data = json.load(f)
    return index, data["chunks"], data["model"]


# ------------------------------------------------------------------ helpers
def ingredients_text(meta):
    return " ".join(meta.get("key_ingredients", [])) + " " + meta.get("description", "")


def is_blocked(meta):
    return bool(BLOCKED_RE.search(ingredients_text(meta)))


def is_strong(meta):
    return bool(STRONG_RE.search(ingredients_text(meta)))


def bucket_of(category):
    for b, cats in BUCKETS.items():
        if category in cats:
            return b
    return None


def price_text(meta):
    return f"Rs. {meta['price']:,.0f}" if meta.get("price") else "Price not available"


# ------------------------------------------------------------------ safety (plain Python, no LLM)
def all_severe(profile):
    """Acne, redness AND irritation all at the top 'severe' level -> dermatologist message only."""
    return (profile["acne"] == ACNE_SEVERE and profile["redness"] == REDNESS_SEVERE
            and profile["irritation"] == IRRITATION_SEVERE)


def mentions_pregnancy(text):
    return bool(PREGNANCY_RE.search(text or ""))


def mentions_prescription(text):
    return bool(PRESCRIPTION_RE.search(text or ""))


def safety_flags(profile):
    flags, must_escalate = [], False
    if profile["acne"] == ACNE_SEVERE:
        flags.append("Severe or persistent acne should be evaluated by a dermatologist. "
                     "We will keep suggestions gentle and avoid strong actives.")
        must_escalate = True
    if profile["irritation"] in ("Frequently", IRRITATION_SEVERE) or profile["redness"] == REDNESS_SEVERE:
        flags.append("Your skin looks very sensitive or reactive. We will keep the routine minimal "
                     "and avoid strong actives. Consider professional advice.")
    note = (profile.get("extra_note") or "").lower()
    hit = [w for w in ESCALATION_WORDS if w in note]
    if hit:
        flags.append("What you described (" + ", ".join(hit) + ") may need professional care. "
                     "Please see a doctor or dermatologist rather than relying on an app.")
        must_escalate = True
    return flags, must_escalate


def allergy_flags(profile):
    flags = []
    if profile.get("allergy"):
        hit = sorted({m.group(0).lower() for m in ALLERGY_SEVERE_RE.finditer(profile["allergy"])})
        if hit:
            flags.append("Your allergy description mentions (" + ", ".join(hit) + "). Reactions like this "
                         "should be assessed by a doctor or allergist, not an app.")
    return flags


def gentle_only(profile):
    return (profile["acne"] == ACNE_SEVERE
            or profile["irritation"] in ("Frequently", IRRITATION_SEVERE)
            or profile["redness"] == REDNESS_SEVERE
            or (profile["skin_type"] == "Sensitive / Unsure" and profile["irritation"] != "None"))


ALLERGY_STOP = {
    "allergic", "allergy", "allergies", "allergen", "from", "skin", "reaction", "reactions", "rash",
    "when", "with", "have", "having", "gets", "products", "product", "containing", "contain", "contains",
    "type", "types", "mild", "severe", "cause", "causes", "causing", "itchy", "itching", "redness",
    "after", "using", "very", "sometimes", "also", "especially", "like", "such", "some", "react",
    "reacts", "acid", "extract", "that", "this", "them", "they", "been", "were", "face", "burning",
    "swelling", "swollen", "bumps", "red", "dryness", "usually", "always", "mostly", "cream", "creams",
}


def allergy_keywords(text):
    """Turns 'allergic to fragrance and salicylic acid, gets itchy' into ['fragrance', 'salicylic']."""
    words = re.findall(r"[a-z]+", (text or "").lower())
    return sorted({w for w in words if len(w) >= 4 and w not in ALLERGY_STOP})


def allergen_hit(meta, allergens):
    text = ingredients_text(meta).lower() + " " + meta.get("name", "").lower()
    for kw in allergens:
        stem = kw[:-1] if kw.endswith("s") and len(kw) > 4 else kw
        if stem in text:
            return kw
    return None


def make_ctx(profile):
    return {"gentle": gentle_only(profile),
            "pregnant": bool(profile.get("pregnant")),
            "allergens": allergy_keywords(profile.get("allergy", ""))}


def exclusion_reason(m, ctx):
    """None if the product is fine; otherwise (verdict, reason). Used for NEW and OWNED products."""
    if is_blocked(m):
        return ("Review", "This looks like a medicine. We don't judge medicines; follow your doctor's advice.")
    hit = allergen_hit(m, ctx["allergens"])
    if hit:
        return ("Consider Replacing",
                f"May contain something you are allergic to ('{hit}'). Check the full ingredient list.")
    if ctx["pregnant"] and (is_strong(m) or PREGNANCY_AVOID_RE.search(ingredients_text(m))):
        return ("Consider Replacing", "Contains an active ingredient. Please check with your doctor while pregnant.")
    if ctx["gentle"] and is_strong(m):
        return ("Consider Replacing", "Contains a strong active, which is too harsh for the skin you described.")
    return None


# ------------------------------------------------------------------ retrieval
def profile_query(p):
    want = f" Looking for: {p['single_bucket']}." if p.get("mode") == "single" else ""
    return (f"{p['skin_type']} skin. Acne: {p['acne']}. Redness: {p['redness']}. "
            f"Sensitivity: {p['irritation']}. Goals: {', '.join(p['goals']) or 'general care'}. "
            f"Concerns: {', '.join(p['concerns']) or 'none listed'}.{want}")


def retrieve(profile, index, chunks, embedder, products):
    """Embeds ONLY the query, ranks with the saved FAISS index, then filters using LIVE Supabase data."""
    q = embedder.encode([profile_query(profile)], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, index.ntotal)        # tiny index -> rank everything

    ctx = make_ctx(profile)
    single = profile.get("mode") == "single"
    wanted = [profile["single_bucket"]] if single else list(BUCKETS)
    grouped = {b: [] for b in wanted}
    knowledge = []
    for i in ids[0]:
        if i < 0:
            continue
        c = chunks[i]
        if c["type"] == "knowledge":
            if len(knowledge) < 5:
                knowledge.append(c)
            continue
        m = products.get(c["meta"]["product_id"])
        if m is None:                                   # deleted / deactivated in Supabase
            continue
        b = bucket_of(m["category"])
        if b is None or b not in grouped:
            continue
        if b == "treatment" and ctx["pregnant"]:        # pregnancy: no actives / treatments at all
            continue
        cap = SINGLE_LIMIT if single else PER_BUCKET[b]
        if len(grouped[b]) >= cap:
            continue
        is_pk = m["origin"] == "Pakistan"
        if (is_pk and not profile["allow_pakistani"]) or (not is_pk and not profile["allow_imported"]):
            continue
        if exclusion_reason(m, ctx):
            continue
        if m.get("price") and profile["budget"] > 0 and m["price"] > profile["budget"]:
            continue
        grouped[b].append(m)
    return grouped, knowledge


GENERIC_WORDS = {"cleanser", "cleansing", "cream", "moisturizer", "moisturiser", "moisturizing",
                 "sunscreen", "sunblock", "serum", "face", "wash", "gel", "lotion", "toner",
                 "spf", "daily", "skin", "essence", "treatment"}
SKIP_WORDS = {"the", "facial", "and", "by", "for"}


def _tokens(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1 and w not in SKIP_WORDS]


def match_existing(text, products):
    """Match the user's own products to our database (live Supabase data).
    Every distinctive word (brand / product name) must be found, and most words must match."""
    cand_tokens = {pid: _tokens(f"{m['name']} {m['brand']}") for pid, m in products.items()}
    matched, unmatched = [], []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        name = re.split(r"\s+[-–|]\s+", line)[0]
        user = _tokens(name)
        distinct = [w for w in user if w not in GENERIC_WORDS]
        best, best_key = None, None
        if distinct:
            for pid, ctoks in cand_tokens.items():
                found = {w: max((fuzz.ratio(w, c) for c in ctoks), default=0) >= 85 for w in user}
                coverage = sum(found.values()) / len(user)
                if coverage >= 0.67 and all(found[w] for w in distinct):
                    key = (coverage, -len(ctoks))          # best coverage, then closest length
                    if best_key is None or key > best_key:
                        best, best_key = pid, key
        if best:
            if products[best] not in matched:
                matched.append(products[best])
        else:
            unmatched.append(line)
    return matched, unmatched


# ------------------------------------------------------------------ prompts + LLM
def product_line(m):
    return (f"[{m['product_id']}] {m['name']} ({m['brand']}) | {m['origin']} | {price_text(m)} | "
            f"Ingredients: {', '.join(m['key_ingredients']) or 'n/a'} | "
            f"Skin types: {', '.join(m['skin_types']) or 'n/a'} | "
            f"Concerns: {', '.join(m['concerns']) or 'n/a'} | "
            f"Strong active: {'yes' if is_strong(m) else 'no'}")


def _profile_for_prompt(profile):
    hidden = {"extra_note", "allergy", "pregnant", "has_allergy", "mode", "single_bucket"}
    return {k: v for k, v in profile.items() if k not in hidden}


def _special_rules(ctx):
    rules = []
    if ctx["pregnant"]:
        rules.append("The user is pregnant or breastfeeding: use ONLY basic products, no active ingredients, "
                     "and never include a Treatment step.")
    if ctx["allergens"]:
        rules.append("The user is allergic to: " + ", ".join(ctx["allergens"]) +
                     ". Never choose a product that contains or is likely to contain these.")
    return rules


def build_messages(profile, grouped, knowledge, matched, unmatched, ctx, extra=""):
    options = ""
    for b, items in grouped.items():
        options += f"\n## {b.upper()} OPTIONS\n"
        options += "\n".join(product_line(m) for m in items) or "(none available)"
        options += "\n"
    owned = "\n".join(product_line(m) for m in matched) or "(none matched)"
    kb = "\n\n".join(c["text"][:600] for c in knowledge)

    base = [
        "Recommend ONLY products listed under OPTIONS or EXISTING PRODUCTS, using their product_id. "
        "Never invent products, prices, sellers or links.",
        "Do not diagnose or prescribe. Do not claim to cure anything.",
        "If an existing product is suitable, keep it (use_existing) instead of buying a new one.",
        "Total price of NEW purchases must stay within the budget.",
        "Use at most ONE strong-active product in the whole routine (zero if the skin is sensitive). "
        "Sunscreen only in the morning.",
        "Base 'why' only on the product details and knowledge given. Keep each 'why' under 25 words.",
    ] + _special_rules(ctx)
    rules = "\n".join(f"{i}. {r}" for i, r in enumerate(base, 1))
    steps = ("AM steps: Cleanser, Moisturizer, Sunscreen. PM steps: Cleanser, Moisturizer. "
             if ctx["pregnant"] else
             "AM steps: Cleanser, Treatment (optional), Moisturizer, Sunscreen. "
             "PM steps: Cleanser, Treatment (optional), Moisturizer. ")
    system = (
        "You are a careful skincare assistant for users in Pakistan.\nRULES:\n" + rules + "\n"
        "Reply with JSON ONLY, in exactly this shape:\n"
        '{"am":[{"step":"Cleanser","product_id":"P001 or null","use_existing":"P002 or null","why":"..."}],'
        '"pm":[...],'
        '"audit":[{"product_id":"P002","verdict":"Keep|Review|Consider Replacing|Unknown","reason":"..."}],'
        '"notes":"one or two friendly sentences"}\n' + steps +
        "'audit' lists ONLY the EXISTING PRODUCTS shown."
    )
    user = (f"USER PROFILE:\n{json.dumps(_profile_for_prompt(profile))}\n\n"
            f"EXISTING PRODUCTS (already owned, cost Rs. 0):\n{owned}\n\n"
            f"OPTIONS TO BUY:\n{options}\n\nSKINCARE KNOWLEDGE:\n{kb}\n{extra}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_single_messages(profile, bucket, items, knowledge, matched, ctx):
    options = "\n".join(product_line(m) for m in items)
    owned = "\n".join(product_line(m) for m in matched) or "(none matched)"
    kb = "\n\n".join(c["text"][:600] for c in knowledge)
    base = [
        f"The user wants ONE {bucket.upper()} product. Choose up to 3 options ONLY from OPTIONS, by product_id, "
        "best match first. Never invent products, prices, sellers or links.",
        "Do not diagnose or prescribe. Do not claim to cure anything.",
        "Base 'why' only on the product details and knowledge given. Keep each 'why' under 25 words.",
    ] + _special_rules(ctx)
    rules = "\n".join(f"{i}. {r}" for i, r in enumerate(base, 1))
    system = (
        "You are a careful skincare assistant for users in Pakistan.\nRULES:\n" + rules + "\n"
        "Reply with JSON ONLY, in exactly this shape:\n"
        '{"options":[{"product_id":"P001","why":"..."}],'
        '"audit":[{"product_id":"P002","verdict":"Keep|Review|Consider Replacing|Unknown","reason":"..."}],'
        '"notes":"one or two friendly sentences"}\n'
        "'audit' lists ONLY the EXISTING PRODUCTS shown (use [] if none)."
    )
    user = (f"USER PROFILE:\n{json.dumps(_profile_for_prompt(profile))}\n\n"
            f"EXISTING PRODUCTS (already owned):\n{owned}\n\n"
            f"OPTIONS ({bucket}):\n{options}\n\nSKINCARE KNOWLEDGE:\n{kb}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_llm(client, messages):
    resp = client.chat.completions.create(
        model=GROQ_MODEL, temperature=0.2,
        response_format={"type": "json_object"}, messages=messages)
    return json.loads(resp.choices[0].message.content)


# ------------------------------------------------------------------ checking the LLM answer
def clean_result(result, lookup, owned_ids, ctx):
    """Drops hallucinated IDs and unsafe owned products, fixes verdicts. Returns a safe structure."""
    def clean_steps(steps, allow_sunscreen):
        out = []
        for s in steps or []:
            step = str(s.get("step", "")).strip()
            if step.lower() == "sunscreen" and not allow_sunscreen:
                continue
            if step.lower() == "treatment" and ctx["pregnant"]:
                continue
            pid, ex = s.get("product_id"), s.get("use_existing")
            m = lookup.get(pid) if pid else None
            e = lookup.get(ex) if ex and ex in owned_ids else None
            if e and exclusion_reason(e, ctx):
                e = None
            if m or e:
                out.append({"step": step, "product": m, "existing": e, "why": s.get("why", "")})
        return out
    return {
        "am": clean_steps(result.get("am"), True),
        "pm": clean_steps(result.get("pm"), False),
        "audit_raw": result.get("audit") or [],
        "notes": result.get("notes", ""),
    }


def clean_single(result, lookup, fallback_items):
    out, seen = [], set()
    for o in result.get("options") or []:
        m = lookup.get(o.get("product_id"))
        if m and m["product_id"] not in seen:
            seen.add(m["product_id"])
            out.append({"product": m, "why": o.get("why", "")})
    if not out:                                            # LLM gave nothing usable -> best-ranked items
        out = [{"product": m, "why": "Closest match to your skin profile and filters."}
               for m in fallback_items[:3]]
    return out[:3], result.get("audit") or [], result.get("notes", "")


def new_purchases(routine, owned_ids):
    seen = {}
    for part in ("am", "pm"):
        for s in routine[part]:
            m = s["product"]
            if m and m["product_id"] not in owned_ids:
                seen[m["product_id"]] = m
    return list(seen.values())


def total_cost(routine, owned_ids):
    return sum(m["price"] or 0 for m in new_purchases(routine, owned_ids))


def strong_count(routine):
    ids = {}
    for part in ("am", "pm"):
        for s in routine[part]:
            for m in (s["product"], s["existing"]):
                if m and is_strong(m):
                    ids[m["product_id"]] = m
    return len(ids)


def build_audit(audit_raw, matched, unmatched, ctx):
    by_id = {a.get("product_id"): a for a in audit_raw}
    rows = []
    for m in matched:
        a = by_id.get(m["product_id"], {})
        v = a.get("verdict") if a.get("verdict") in VERDICTS else "Unknown"
        reason = a.get("reason") or "Not enough information to judge."
        ex = exclusion_reason(m, ctx)                       # safety rules always override the LLM
        if ex:
            v, reason = ex
        rows.append({"product": f"{m['name']} ({m['brand']})", "verdict": v, "reason": reason})
    for line in unmatched:                  # products not in our database are always Unknown
        rows.append({"product": line, "verdict": "Unknown",
                     "reason": "This product is not in our database, so we cannot judge it."})
    return rows


# ------------------------------------------------------------------ pipelines
def generate_routine(profile, index, chunks, embedder, client, products):
    """FULL ROUTINE. Returns (routine, audit, meta_info). routine is None if nothing could be built."""
    ctx = make_ctx(profile)
    grouped, knowledge = retrieve(profile, index, chunks, embedder, products)
    matched, unmatched = match_existing(profile.get("current_routine", ""), products)
    usable = [m for m in matched if not exclusion_reason(m, ctx)]
    owned_ids = {m["product_id"] for m in usable}
    lookup = {m["product_id"]: m for items in grouped.values() for m in items}
    lookup.update({m["product_id"]: m for m in usable})

    if not any(grouped.values()) and not usable:
        return None, [], {"error": "No matching products found. Try a higher budget or include more product types."}

    messages = build_messages(profile, grouped, knowledge, usable, unmatched, ctx)
    routine = clean_result(call_llm(client, messages), lookup, owned_ids, ctx)

    # budget check done in Python (LLMs are bad at arithmetic) - one retry if over
    cost = total_cost(routine, owned_ids)
    if profile["budget"] > 0 and cost > profile["budget"]:
        extra = (f"\nYour last routine cost Rs. {cost:,.0f} in new purchases, which is over the budget of "
                 f"Rs. {profile['budget']:,.0f}. Choose cheaper options or keep more existing products.")
        routine = clean_result(call_llm(client, build_messages(
            profile, grouped, knowledge, usable, unmatched, ctx, extra)), lookup, owned_ids, ctx)
        cost = total_cost(routine, owned_ids)

    audit = build_audit(routine["audit_raw"], matched, unmatched, ctx)
    info = {
        "cost": cost,
        "over_budget": profile["budget"] > 0 and cost > profile["budget"],
        "strong_count": strong_count(routine),
        "new": new_purchases(routine, owned_ids),
        "knowledge": knowledge,
    }
    return routine, audit, info


def generate_single(profile, index, chunks, embedder, client, products):
    """ONE PRODUCT. Returns (options, audit, info). options is None if nothing matched."""
    ctx = make_ctx(profile)
    bucket = profile["single_bucket"]
    if bucket == "treatment" and ctx["pregnant"]:
        return None, [], {"error": "Because you mentioned pregnancy, we don't suggest treatment or active "
                                   "products. Please check with your doctor."}
    grouped, knowledge = retrieve(profile, index, chunks, embedder, products)
    items = grouped.get(bucket, [])
    matched, unmatched = match_existing(profile.get("current_routine", ""), products)
    usable = [m for m in matched if not exclusion_reason(m, ctx)]
    if not items:
        return None, build_audit([], matched, unmatched, ctx), {
            "error": "No matching products found. Try a higher budget or include more product types."}

    lookup = {m["product_id"]: m for m in items}
    result = call_llm(client, build_single_messages(profile, bucket, items, knowledge, usable, ctx))
    options, audit_raw, notes = clean_single(result, lookup, items)
    audit = build_audit(audit_raw, matched, unmatched, ctx)
    return options, audit, {"notes": notes, "knowledge": knowledge}
