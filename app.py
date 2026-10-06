import os

import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

import core

st.set_page_config(page_title="Skinfluence PK", page_icon="🧴", layout="centered")

DISCLAIMER = (
    "Disclaimer: Skinfluence PK provides general skincare information and product recommendations "
    "for informational purposes only. It does not provide medical advice, diagnose skin conditions, "
    "prescribe treatments, or replace a qualified dermatologist or healthcare professional. "
    "Recommendations are based on the information you provide and available product data. "
    "If you have severe, persistent, or worsening skin concerns, consult a qualified healthcare professional."
)

# ---------------------------------------------------------------- earthy warm theme
# Lightest #F4EEE7 | Lighter #D2BB9F | Light #B18756 | Primary #8F540E | Dark #643B0A | Darker #482A07
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&family=Playfair+Display:wght@600;700&display=swap');
:root { --lightest:#F4EEE7; --lighter:#D2BB9F; --light:#B18756; --primary:#8F540E; --dark:#643B0A; --darker:#482A07; }
html, body, .stApp, [data-testid="stAppViewContainer"] { background-color: var(--lightest); color: var(--darker);
    font-family: 'Nunito', 'Segoe UI', sans-serif; }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 780px; padding-top: 2.2rem; padding-bottom: 4rem; }
h1, h2, h3 { font-family: 'Playfair Display', Georgia, serif !important; color: var(--dark) !important; letter-spacing: .2px; }
h3 { border-bottom: 2px solid var(--lighter); padding-bottom: .35rem; margin-top: 1.6rem; }
label, .stMarkdown p, .stCaption, [data-testid="stWidgetLabel"] p { color: var(--darker); }

/* buttons: soft, rounded */
.stButton > button, [data-testid^="stBaseLinkButton"], .stLinkButton a {
    border-radius: 999px !important; border: 1px solid var(--primary) !important;
    background-color: var(--primary) !important; color: var(--lightest) !important;
    font-weight: 700; padding: .55rem 1.6rem; transition: background-color .15s ease; }
.stButton > button:hover, [data-testid^="stBaseLinkButton"]:hover, .stLinkButton a:hover {
    background-color: var(--dark) !important; border-color: var(--dark) !important; color: #fff !important; }
.stButton > button p, .stLinkButton a p { color: inherit !important; }

/* messages */
[data-testid="stAlert"] { border-radius: 18px; }
[data-testid="stAlert"] > div { background-color: rgba(210,187,159,.38) !important; color: var(--darker) !important; border-radius: 18px; }

/* hero + disclaimer cards */
.hero { background: linear-gradient(135deg, #FBF8F4 0%, var(--lighter) 160%); border: 1px solid var(--lighter);
    border-radius: 28px; padding: 2rem 2.2rem; margin-bottom: 1.1rem; box-shadow: 0 6px 24px rgba(72,42,7,.07); }
.hero h1 { margin: 0 0 .2rem 0; font-size: 2.3rem; }
.hero .tagline { font-family: 'Playfair Display', Georgia, serif; font-size: 1.25rem; font-weight: 600;
    color: var(--primary); margin: .2rem 0 1rem 0; line-height: 1.4; }
.hero p { font-size: 1.02rem; line-height: 1.65; margin: 0 0 .85rem 0; color: var(--darker); }
.hero p.closing { font-weight: 700; color: var(--dark); margin-bottom: 0; }
.disclaimer { background: rgba(210,187,159,.32); border-left: 5px solid var(--light); border-radius: 18px;
    padding: .9rem 1.2rem; font-size: .9rem; line-height: 1.55; color: var(--darker); margin-bottom: 1.2rem; }
.pill { display:inline-block; background: var(--lighter); color: var(--darker); border-radius: 999px;
    padding: .1rem .75rem; font-size: .8rem; font-weight: 700; margin-right: .4rem; }
.bigmsg { background: #FBF8F4; border: 2px solid var(--primary); border-radius: 24px; padding: 1.6rem 1.8rem;
    font-size: 1.35rem; font-weight: 700; color: var(--dark); text-align: center; margin: 1rem 0; }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- loaded ONCE per server, not per query
@st.cache_resource
def load_resources():
    index, chunks, model_name = core.load_index()
    embedder = SentenceTransformer(model_name)   # same model that built the index
    return index, chunks, embedder


@st.cache_resource
def get_client():
    key = _secret("GROQ_API_KEY")
    return Groq(api_key=key) if key else None


def _secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, "")


# Supabase: re-read at most once a minute, so any change you make there shows up in the app within ~1 minute.
@st.cache_data(ttl=60, show_spinner=False)
def fetch_live():
    url = _secret("SUPABASE_URL")
    key = _secret("SUPABASE_KEY") or _secret("SUPABASE_ANON_KEY")
    if not url or not key:
        raise RuntimeError("Supabase is not configured")
    return core.fetch_live(url, key)   # raises on failure, and failures are NOT cached


def get_live():
    try:
        return fetch_live(), True
    except Exception:
        return {}, False


index, chunks, embedder = load_resources()
client = get_client()
live, live_ok = get_live()
live_chunks = core.apply_live(chunks, live)

# ---------------------------------------------------------------- header + intro (shown BEFORE the disclaimer)
st.markdown(
    """
<div class="hero">
  <h1>🧴 Skinfluence PK</h1>
  <div class="tagline">Personalised skincare recommendations built around what you can actually buy in Pakistan.</div>
  <p>Skincare shouldn't require hours of research.<br>
  Tell Skinfluence about your skin, your concerns, and what you're looking for, and we'll help you find
  relevant products from a curated selection of Pakistani pharmacy products, local skincare brands, and
  trusted international options available in Pakistan.</p>
  <p>Our recommendations are based on product ingredients, skin concerns, and your preferences—not just
  marketing claims. We also incorporate insights from real skincare discussions to help you discover products
  that people in Pakistan actually talk about and use.</p>
  <p class="closing">Find products that make sense for your skin, without the guesswork.</p>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)

if client is None:
    st.error("Groq API key not found. Add GROQ_API_KEY in Streamlit secrets.")
    st.stop()

# ---------------------------------------------------------------- questionnaire
# (no st.form here: plain widgets let sections appear / disappear instantly)
st.subheader("1. Your skin")
age = st.number_input("Age", 10, 90, 22)
skin_type = st.selectbox("Skin type", ["Dry", "Oily", "Combination", "Normal", "Sensitive / Unsure"])
c1, c2, c3 = st.columns(3)
acne = c1.selectbox("Acne", ["None", "Occasional", "Frequent", "Severe / Persistent Concern"])
redness = c2.selectbox("Redness", ["None", "Mild", "Frequent", "Significant"])
irritation = c3.selectbox("Irritation", ["None", "Sometimes", "Frequently", "Very Sensitive"])
goals = st.multiselect("Skin goals", [
    "Reduce breakouts", "Improve hydration", "Improve skin texture",
    "Reduce appearance of dark spots/acne marks", "Control excess oil", "Support skin barrier",
    "Brighten dull-looking skin", "Maintain healthy-looking skin", "Simplify skincare routine",
    "Early anti-aging / prevention"])
concerns = st.multiselect("Extra concerns (optional)", [
    "Blackheads", "Whiteheads / closed comedones", "Post-acne marks", "Uneven texture", "Dryness",
    "Flakiness", "Excess oil", "Dehydrated skin", "Dark circles", "Uneven skin tone",
    "Roughness", "Enlarged-looking pores"])

has_allergy = st.radio("Do you have any known skin allergies?", ["No", "Yes"], horizontal=True)
allergy_text = ""
if has_allergy == "Yes":
    allergy_text = st.text_area(
        "Please describe your allergy in a short paragraph",
        max_chars=400, height=100,
        placeholder="Example: I get red itchy skin from fragrance and from salicylic acid.")

extra_note = st.text_area("Anything else we should know? (optional)", max_chars=400, height=80)

st.subheader("2. Your current routine")
no_routine = st.checkbox("I don't currently have a skincare routine")
current = ""
if not no_routine:
    current = st.text_area(
        "Products you use now (one per line, e.g. 'CeraVe Foaming Cleanser - cleanser')", height=110)

st.subheader("3. Preferences")
mode_label = st.radio("What are you looking for?",
                      ["A full routine (morning & evening)", "Just one product"])
single = mode_label == "Just one product"
one_choice = None
if single:
    one_choice = st.selectbox("Which product do you need?", list(core.ONE_PRODUCT_CHOICES))

budget = st.number_input(
    "Budget for this product (Rs.)" if single else "Skincare budget for NEW purchases (Rs.)",
    0, 200000, 2000 if single else 4000, step=500)

st.markdown("**Where should we look?**")
w1, w2 = st.columns(2)
allow_pk = w1.checkbox("Pakistani products", value=True)
allow_imp = w2.checkbox("Imported products", value=False)

go = st.button("Find my products" if single else "Build my routine", type="primary")


# ---------------------------------------------------------------- run
if go:
    problems = []
    if not (allow_pk or allow_imp):
        problems.append("Please tick at least one option under 'Where should we look?'.")
    if has_allergy == "Yes" and not allergy_text.strip():
        problems.append("Please describe your allergy, or choose 'No'.")
    for p in problems:
        st.error(p)

    if not problems:
        profile = dict(
            age=age, skin_type=skin_type, acne=acne, redness=redness, irritation=irritation,
            goals=goals, concerns=concerns, extra_note=extra_note,
            allergy_text=allergy_text.strip() if has_allergy == "Yes" else "",
            current_routine="" if no_routine else current,
            budget=budget, allow_pakistan=allow_pk, allow_imported=allow_imp,
            mode="single" if single else "routine",
            only_bucket=core.ONE_PRODUCT_CHOICES.get(one_choice) if single else None,
            pregnant=core.mentions_pregnancy(extra_note))

        msg = core.stop_message(profile)
        if msg:                                           # message only, nothing else is shown
            st.session_state["result"] = {"kind": "message", "text": msg}
        else:
            flags, _ = core.safety_flags(profile)
            with st.spinner("Searching products and building your recommendations..."):
                try:
                    res = core.generate(profile, index, live_chunks, embedder, client)
                except Exception as e:
                    st.error(f"The AI service is busy or returned an error. Please try again in a minute. ({e})")
                    st.stop()
            res["flags"] = flags
            res["budget"] = budget
            st.session_state["result"] = res


# ---------------------------------------------------------------- show results
def product_card(step_title, m, why, owned=False):
    with st.container(border=True):
        tag = "✅ You already own this" if owned else "🛒 New purchase"
        st.markdown(f"**{step_title}{m['name']}**  \n{m['brand']} · {tag}")
        seller = m.get("seller") or "not listed"
        status = m.get("verification_status") or "Unverified"
        checked = f" · Last checked: {m['last_verified']}" if m.get("last_verified") else ""
        st.caption(f"Category: {m['category']} · Origin: {m['origin']} · Price: {core.price_text(m)} · "
                   f"Seller: {seller} · Status: {status}{checked}")
        st.write(f"**Why this product?** {why}")
        for ins in (m.get("insights") or []):
            src = f" ([r/{ins['subreddit']}]({ins['url']}))" if ins.get("url") and ins.get("subreddit") else (
                f" ([source]({ins['url']}))" if ins.get("url") else "")
            st.markdown(f"💬 **Community insight** *(personal experience, not medical evidence)*: "
                        f"{ins['summary']}{src}")
        if not owned and m.get("purchase_url"):
            st.link_button("Buy Product", m["purchase_url"])
            others = [o for o in (m.get("other_sources") or []) if o.get("url")]
            if others:
                st.caption("Also available at: " + " · ".join(
                    f"[{o['seller'] or 'Seller'}]({o['url']})" + (f" (Rs. {o['price']:,.0f})" if o.get("price") else "")
                    for o in others))
        elif not owned:
            st.caption("Direct purchase link not available yet.")


def show_audit(audit):
    if audit:
        st.subheader("Your existing routine")
        icon = {"Keep": "✅", "Review": "🟡", "Consider Replacing": "🔁", "Unknown": "❔"}
        for a in audit:
            st.write(f"{icon[a['verdict']]} **{a['product']}**: {a['verdict']}. {a['reason']}")


res = st.session_state.get("result")
if res:
    st.divider()
    if res["kind"] == "message":
        st.markdown(f'<div class="bigmsg">{res["text"]}</div>', unsafe_allow_html=True)
        st.stop()

    for f in res.get("flags", []):
        st.warning(f)
    for n in res.get("notices", []):
        st.info(n)

    if res["kind"] == "error":
        st.error(res["message"])

    elif res["kind"] == "routine":
        routine, info = res["routine"], res["info"]
        st.header("Your personalised routine")
        st.subheader("☀️ Morning")
        for s in routine["am"]:
            owned = s["product"] is None
            product_card(f"{s['step']}: ", s["product"] or s["existing"], s["why"], owned)
        st.subheader("🌙 Evening")
        for s in routine["pm"]:
            owned = s["product"] is None
            product_card(f"{s['step']}: ", s["product"] or s["existing"], s["why"], owned)

        st.subheader("Budget")
        st.write(f"New purchases: **Rs. {info['cost']:,.0f}** of your Rs. {res['budget']:,.0f} budget.")
        if info["over_budget"]:
            st.warning("This routine is slightly over your budget. You can start with the cleanser, "
                       "moisturizer and sunscreen first and add the treatment later.")
        if info["strong_count"] > 1:
            st.warning("This routine contains more than one strong active. Introduce them one at a time, "
                       "and stop if your skin becomes irritated.")
        show_audit(res["audit"])
        if routine["notes"]:
            st.info(routine["notes"])

    elif res["kind"] == "single":
        st.header("Your product options")
        for i, p in enumerate(res["picks"], 1):
            product_card(f"Option {i}: ", p["product"], p["why"])
        show_audit(res["audit"])
        if res["notes"]:
            st.info(res["notes"])

    if res["kind"] in ("routine", "single"):
        st.caption("New products: patch test first and add only one new active at a time.")
        st.caption("Prices, sellers and links are read live from our product database."
                   if live_ok else
                   "Live price and link data is unavailable right now, so saved data is shown. "
                   "Please check the price on the seller's page.")
        st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)
