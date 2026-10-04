import os

import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

import core

st.set_page_config(page_title="Skinfluence PK", page_icon="🧴", layout="centered")

LIVE_TTL_SECONDS = 60   # how often Supabase changes are picked up (price, links, is_active, ...)

# ---------- earthy theme (palette: F4EEE7 / D2BB9F / B18756 / 8F540E / 643B0A / 482A07) ----------
st.markdown("""
<style>
:root{--lightest:#F4EEE7;--lighter:#D2BB9F;--light:#B18756;--primary:#8F540E;--dark:#643B0A;--darker:#482A07;}
.stApp{background:var(--lightest);}
h1,h2,h3,h4,h5{color:var(--darker)!important;letter-spacing:-0.01em;}
p,label,li,span,div[data-testid="stMarkdownContainer"]{color:var(--darker);}
.stCaption,div[data-testid="stCaptionContainer"]{color:var(--dark)!important;}
/* rounded + soft */
div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:20px;border:1px solid var(--lighter);
  background:rgba(255,255,255,.45);}
[data-baseweb="input"],[data-baseweb="textarea"],[data-baseweb="select"]>div{border-radius:14px!important;
  border-color:var(--lighter)!important;}
div[data-testid="stAlert"]{border-radius:16px;background:rgba(210,187,159,.35)!important;
  border:1px solid var(--lighter);}
div[data-testid="stAlert"] *{color:var(--darker)!important;}
.stButton>button,.stLinkButton>a{border-radius:999px!important;border:1px solid var(--primary);
  padding:.55rem 1.6rem;font-weight:600;transition:all .15s ease;}
.stButton>button[kind="primary"],.stButton>button[data-testid="stBaseButton-primary"]{
  background:var(--primary);color:var(--lightest)!important;}
.stButton>button[kind="primary"]:hover,.stButton>button[data-testid="stBaseButton-primary"]:hover{
  background:var(--dark);border-color:var(--dark);}
.stLinkButton>a{background:var(--lighter);color:var(--darker)!important;}
.stLinkButton>a:hover{background:var(--light);border-color:var(--light);}
hr{border-color:var(--lighter);}
.callout{background:var(--dark);color:var(--lightest)!important;padding:1.3rem 1.5rem;border-radius:20px;
  font-size:1.15rem;font-weight:600;margin:1rem 0;}
.callout *{color:var(--lightest)!important;}
.pill{display:inline-block;background:var(--lighter);color:var(--darker);padding:.1rem .7rem;
  border-radius:999px;font-size:.8rem;font-weight:600;}
</style>
""", unsafe_allow_html=True)

INTRO = """**Personalised skincare recommendations built around what you can actually buy in Pakistan.**

Skincare shouldn't require hours of research.
Tell Skinfluence about your skin, your concerns, and what you're looking for, and we'll help you find relevant products from a curated selection of Pakistani pharmacy products, local skincare brands, and trusted international options available in Pakistan.

Our recommendations are based on product ingredients, skin concerns, and your preferences—not just marketing claims. We also incorporate insights from real skincare discussions to help you discover products that people in Pakistan actually talk about and use.

Find products that make sense for your skin, without the guesswork."""

DISCLAIMER = (
    "Disclaimer: Skinfluence PK provides general skincare information and product recommendations "
    "for informational purposes only. It does not provide medical advice, diagnose skin conditions, "
    "prescribe treatments, or replace a qualified dermatologist or healthcare professional. "
    "Recommendations are based on the information you provide and available product data. "
    "If you have severe, persistent, or worsening skin concerns, consult a qualified healthcare professional."
)


def secret(name):
    try:
        value = st.secrets[name]
    except Exception:
        value = os.environ.get(name, "")
    return value or ""


# ---------- loaded ONCE per server, not per query ----------
@st.cache_resource
def load_resources():
    index, chunks, model_name = core.load_index()
    embedder = SentenceTransformer(model_name)   # same model that built the index
    return index, chunks, embedder


@st.cache_resource
def get_client():
    key = secret("GROQ_API_KEY")
    return Groq(api_key=key) if key else None


# ---------- live product data from Supabase (re-read at most once per LIVE_TTL_SECONDS) ----------
@st.cache_data(ttl=LIVE_TTL_SECONDS, show_spinner=False)
def fetch_live(url, key):
    return core.fetch_products(url, key)


def get_products(chunks):
    url, key = secret("SUPABASE_URL"), secret("SUPABASE_ANON_KEY")
    if url and key:
        try:
            return core.live_products(fetch_live(url, key)), True
        except Exception:
            pass
    return core.snapshot_products(chunks), False     # fallback: data saved inside the index


try:
    index, chunks, embedder = load_resources()
except Exception as e:
    st.error(f"Search index not found or unreadable. Run build_index.py and upload the 'index' folder. ({e})")
    st.stop()
client = get_client()
products, is_live = get_products(chunks)

# admin hint: open the app with ?admin=1 to see it
if st.query_params.get("admin") == "1":
    indexed = {c["meta"]["product_id"] for c in chunks if c["type"] == "product"}
    missing = sorted(set(products) - indexed)
    st.sidebar.write(f"Active products in Supabase: {len(products)}")
    st.sidebar.write("Supabase connection: " + ("live" if is_live else "NOT connected (using saved snapshot)"))
    if missing:
        st.sidebar.warning(f"{len(missing)} product(s) are not in the search index yet "
                           f"({', '.join(missing[:8])}). Re-run build_index.py to include them.")

# ---------- header ----------
st.title("🧴 Skinfluence PK")
st.markdown(INTRO)
st.info(DISCLAIMER)

if client is None:
    st.error("Groq API key not found. Add GROQ_API_KEY in Streamlit secrets.")
    st.stop()
if not is_live:
    st.warning("Could not reach the live product database, so prices and links may be slightly out of date.")

# ---------- questionnaire (no st.form, so sections can show/hide instantly) ----------
st.subheader("1. What are you looking for?")
mode_label = st.radio("I would like...", ["A full routine", "One specific product"], horizontal=True)
single_choice = None
if mode_label == "One specific product":
    single_choice = st.selectbox("Which type of product?", list(core.SINGLE_CHOICES))

st.subheader("2. Your skin")
age = st.number_input("Age", 10, 90, 22)
skin_type = st.selectbox("Skin type", ["Dry", "Oily", "Combination", "Normal", "Sensitive / Unsure"])
c1, c2, c3 = st.columns(3)
acne = c1.selectbox("Acne", ["None", "Occasional", "Frequent", core.ACNE_SEVERE])
redness = c2.selectbox("Redness", ["None", "Mild", "Frequent", core.REDNESS_SEVERE])
irritation = c3.selectbox("Irritation", ["None", "Sometimes", "Frequently", core.IRRITATION_SEVERE])
goals = st.multiselect("Skin goals", [
    "Reduce breakouts", "Improve hydration", "Improve skin texture",
    "Reduce appearance of dark spots/acne marks", "Control excess oil", "Support skin barrier",
    "Brighten dull-looking skin", "Maintain healthy-looking skin", "Simplify skincare routine",
    "Early anti-aging / prevention"])
concerns = st.multiselect("Extra concerns (optional)", [
    "Blackheads", "Whiteheads / closed comedones", "Post-acne marks", "Uneven texture", "Dryness",
    "Flakiness", "Excess oil", "Dehydrated skin", "Dark circles", "Uneven skin tone",
    "Roughness", "Enlarged-looking pores"])

has_allergy = st.radio("Do you have any known skincare allergies?", ["No", "Yes"], horizontal=True)
allergy_text = ""
if has_allergy == "Yes":
    allergy_text = st.text_area(
        "Briefly describe your allergy (what triggers it and what happens)", height=90, max_chars=400,
        placeholder="e.g. I'm allergic to fragrance and salicylic acid. My skin gets itchy and red.")

extra_note = st.text_input(
    "Anything else we should know? (optional)",
    help="For example pregnancy, or a prescription from your dermatologist.")

st.subheader("3. Your current routine")
no_routine = st.checkbox("I don't currently have a skincare routine")
current = ""
if not no_routine:
    current = st.text_area("Products you use now (one per line, e.g. 'CeraVe Foaming Cleanser - cleanser')",
                           height=110)

st.subheader("4. Preferences")
budget = st.number_input(
    "Skincare budget for NEW purchases (Rs.)" if mode_label == "A full routine" else "Budget for this product (Rs.)",
    0, 200000, 4000, step=500)
st.write("Where should we look?")
w1, w2 = st.columns(2)
allow_pk = w1.checkbox("Pakistani products", value=True)
allow_imp = w2.checkbox("Imported products", value=False)

go = st.button("Build my routine" if mode_label == "A full routine" else "Show me product options",
               type="primary")

# ---------- run ----------
if go:
    single = mode_label == "One specific product"
    profile = dict(
        age=age, skin_type=skin_type, acne=acne, redness=redness, irritation=irritation,
        goals=goals, concerns=concerns, extra_note=extra_note,
        has_allergy=has_allergy == "Yes", allergy=allergy_text.strip() if has_allergy == "Yes" else "",
        pregnant=core.mentions_pregnancy(extra_note),
        current_routine="" if no_routine else current,
        budget=budget, allow_pakistani=allow_pk, allow_imported=allow_imp,
        mode="single" if single else "routine",
        single_bucket=core.SINGLE_CHOICES[single_choice] if single else None)

    st.divider()

    # --- gate 1: all three severe -> ONLY the dermatologist message
    if core.all_severe(profile):
        st.markdown(f'<div class="callout">{core.DERM_MESSAGE}</div>', unsafe_allow_html=True)
        st.stop()

    # --- gate 2: doctor's / dermatologist's prescription mentioned -> specialist message
    if core.mentions_prescription(extra_note):
        st.markdown(f'<div class="callout">{core.SPECIALIST_MESSAGE}</div>', unsafe_allow_html=True)
        if core.PRESCRIPTION_BLOCKS_ROUTINE:
            st.stop()

    # --- input checks
    if not (allow_pk or allow_imp):
        st.error("Please tick at least one option under 'Where should we look?'.")
        st.stop()
    if has_allergy == "Yes" and not allergy_text.strip():
        st.error("You said you have an allergy. Please describe it briefly so we can avoid it.")
        st.stop()

    # --- warnings
    flags, must_escalate = core.safety_flags(profile)
    for f in flags + core.allergy_flags(profile):
        st.warning(f)
    if profile["pregnant"]:
        st.warning("You mentioned pregnancy, so we are not recommending any active ingredients or "
                   "treatments. We will only suggest basics. Please check any skincare with your doctor.")
    if profile["has_allergy"]:
        st.info("We avoid products whose listed ingredients match what you described. Our database lists "
                "key ingredients only, so always check the full ingredient list on the label.")

    def show_card(m, title, why, owned=False):
        with st.container(border=True):
            tag = "✅ You already own this" if owned else "🛒 New purchase"
            st.markdown(f"**{title}: {m['name']}**  \n{m['brand']} · {tag}")
            seller = m.get("seller") or "not listed"
            st.caption(f"Category: {m['category']} · Origin: {m['origin']} · "
                       f"Price: {core.price_text(m)} · Seller: {seller} · Last checked: {m['last_verified']}")
            st.write(f"**Why this product?** {why}")
            url = m.get("purchase_url", "")
            if not owned and url.startswith(("http://", "https://")):
                st.link_button("Buy Product", url)
            elif not owned:
                st.caption("Direct purchase link not available yet.")

    def show_audit(audit):
        if audit:
            st.subheader("Your existing routine")
            icon = {"Keep": "✅", "Review": "🟡", "Consider Replacing": "🔁", "Unknown": "❔"}
            for a in audit:
                st.write(f"{icon[a['verdict']]} **{a['product']}**: {a['verdict']}. {a['reason']}")

    with st.spinner("Searching products and building your recommendations..."):
        try:
            if single:
                result, audit, info = core.generate_single(profile, index, chunks, embedder, client, products)
            else:
                result, audit, info = core.generate_routine(profile, index, chunks, embedder, client, products)
        except Exception as e:
            st.error(f"The AI service is busy or returned an error. Please try again in a minute. ({e})")
            st.stop()

    if result is None:
        st.error(info["error"])
        show_audit(audit)
        st.stop()

    if single:
        st.header(f"Options for you: {single_choice}")
        for n, o in enumerate(result, 1):
            show_card(o["product"], f"Option {n}", o["why"])
        show_audit(audit)
        if info.get("notes"):
            st.info(info["notes"])
    else:
        routine = result
        st.header("Your personalised routine")
        st.subheader("☀️ Morning")
        for s in routine["am"]:
            m = s["product"] or s["existing"]
            show_card(m, s["step"], s["why"], owned=s["product"] is None)
        st.subheader("🌙 Evening")
        for s in routine["pm"]:
            m = s["product"] or s["existing"]
            show_card(m, s["step"], s["why"], owned=s["product"] is None)

        st.subheader("Budget")
        st.write(f"New purchases: **Rs. {info['cost']:,.0f}** of your Rs. {budget:,.0f} budget.")
        if info["over_budget"]:
            st.warning("This routine is slightly over your budget. You can start with the cleanser, "
                       "moisturizer and sunscreen first and add the treatment later.")
        if info["strong_count"] > 1:
            st.warning("This routine contains more than one strong active. Introduce them one at a time, "
                       "and stop if your skin becomes irritated.")
        show_audit(audit)
        if routine["notes"]:
            st.info(routine["notes"])

    st.caption("New products: patch test first and add only one new active at a time.")
    st.caption(DISCLAIMER)
