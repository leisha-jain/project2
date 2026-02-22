import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from textblob import TextBlob
import time
import os

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TruthLens India — Fake News Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --ink: #0d0d0d;
    --paper: #f5f0e8;
    --cream: #ede8dc;
    --accent: #c0392b;
    --accent2: #2c7a4b;
    --warn: #d68910;
    --muted: #7a7060;
    --border: #d4cfc4;
    --saffron: #e67e22;
    --saffron-bg: rgba(230,126,34,0.07);
}

* { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--paper);
    color: var(--ink);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 1rem 4rem; max-width: 760px; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 3rem 0 2rem;
    border-bottom: 2px solid var(--ink);
    margin-bottom: 2.5rem;
    position: relative;
}
.hero::after {
    content: '🇮🇳';
    position: absolute;
    top: 1rem;
    right: 0;
    font-size: 1.8rem;
    opacity: 0.5;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.75rem;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.8rem, 8vw, 4.5rem);
    font-weight: 900;
    line-height: 1;
    letter-spacing: -0.02em;
    margin: 0 0 0.5rem;
}
.hero h1 .truth { color: var(--ink); }
.hero h1 .lens { color: var(--saffron); }
.hero h1 .india { color: var(--accent); font-size: 0.6em; vertical-align: super; }
.hero-sub {
    font-size: 0.9rem;
    color: var(--muted);
    font-weight: 300;
    line-height: 1.6;
    max-width: 480px;
    margin: 0 auto;
}
.model-pill {
    display: inline-block;
    margin-top: 1rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.3rem 1rem;
    border: 1.5px solid var(--saffron);
    color: var(--saffron);
    background: var(--saffron-bg);
}

/* ── Input Section ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.63rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.4rem;
}
textarea {
    font-family: 'DM Sans', sans-serif !important;
    background: #faf7f2 !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 0 !important;
    color: var(--ink) !important;
    font-size: 0.95rem !important;
    line-height: 1.75 !important;
}
textarea:focus { border-color: var(--ink) !important; box-shadow: none !important; }
input[type="text"] {
    font-family: 'DM Mono', monospace !important;
    background: #faf7f2 !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 0 !important;
    color: var(--ink) !important;
    font-size: 0.83rem !important;
}

/* ── Button ── */
.stButton > button {
    background: var(--ink) !important;
    color: var(--paper) !important;
    border: none !important;
    border-radius: 0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.73rem !important;
    letter-spacing: 0.22em !important;
    text-transform: uppercase !important;
    padding: 0.8rem 2rem !important;
    width: 100% !important;
    transition: background 0.15s !important;
}
.stButton > button:hover { background: var(--saffron) !important; }

/* ── Result Card ── */
.result-card {
    border: 2px solid var(--ink);
    margin-top: 2rem;
    background: #faf7f2;
    overflow: hidden;
}

.result-top {
    padding: 2rem 2rem 1.5rem;
    position: relative;
}
.result-top::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 6px;
}
.result-top.fake::before { background: var(--accent); }
.result-top.real::before { background: var(--accent2); }
.result-top.uncertain::before {
    background: linear-gradient(90deg, var(--accent2) 50%, var(--accent) 50%);
}

.verdict-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.25rem;
}
.verdict-text {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 900;
    line-height: 1;
    margin: 0 0 0.4rem;
}
.verdict-text.fake { color: var(--accent); }
.verdict-text.real { color: var(--accent2); }
.verdict-text.uncertain { color: var(--warn); }

.verdict-desc {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 0.05em;
    line-height: 1.5;
}

/* Confidence bar */
.conf-section { margin-top: 1.4rem; }
.conf-header {
    display: flex;
    justify-content: space-between;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    margin-bottom: 0.35rem;
    letter-spacing: 0.06em;
}
.conf-track {
    height: 10px;
    background: var(--cream);
    border: 1px solid var(--border);
    position: relative;
    overflow: hidden;
}
.conf-fill { height: 100%; transition: width 0.8s ease; }
.conf-fill.fake { background: var(--accent); }
.conf-fill.real { background: var(--accent2); }
.conf-fill.uncertain { background: var(--warn); }

/* Dual probability bar */
.dual-bar-wrap { margin-top: 1rem; }
.dual-bar-label {
    display: flex;
    justify-content: space-between;
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: var(--muted);
    margin-bottom: 0.25rem;
}
.dual-bar-track {
    height: 6px;
    background: var(--cream);
    border: 1px solid var(--border);
    display: flex;
    overflow: hidden;
}
.dual-real { background: var(--accent2); height: 100%; }
.dual-fake { background: var(--accent); height: 100%; }

/* ── Metrics Grid ── */
.metrics-section {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    border-top: 1px solid var(--border);
}
.metric-cell {
    padding: 1.1rem;
    text-align: center;
    border-right: 1px solid var(--border);
}
.metric-cell:last-child { border-right: none; }
.metric-val {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 700;
    line-height: 1;
    color: var(--ink);
}
.metric-val.warn { color: var(--accent); }
.metric-key {
    font-family: 'DM Mono', monospace;
    font-size: 0.57rem;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 0.3rem;
}

/* ── Flags Section ── */
.flags-section {
    border-top: 1px solid var(--border);
    padding: 1.2rem 2rem;
}
.flags-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.6rem;
}
.tag {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    padding: 0.18rem 0.5rem;
    margin: 0.18rem;
    border: 1px solid;
}
.tag.high { border-color: var(--accent); color: var(--accent); background: rgba(192,57,43,0.06); }
.tag.medium { border-color: var(--warn); color: var(--warn); background: rgba(214,137,16,0.06); }
.tag.low { border-color: var(--muted); color: var(--muted); }

/* ── Model Info Footer ── */
.model-footer {
    border-top: 1px solid var(--border);
    padding: 0.8rem 2rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: var(--saffron-bg);
}
.model-footer-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--saffron);
}
.model-footer-name {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--ink);
    font-weight: 500;
}

/* Misc */
hr.divider { border: none; border-top: 1px solid var(--border); margin: 2.5rem 0; }
.footer {
    text-align: center;
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 2rem 0 0;
    border-top: 1px solid var(--border);
    margin-top: 3rem;
    line-height: 2;
}
.stSelectbox > div > div {
    border-radius: 0 !important;
    background: #faf7f2 !important;
    border-color: var(--border) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Model Path ───────────────────────────────────────────────────────────────
# Option 1: local folder (place bfnk-bert-model/ next to app.py after unzipping)
# Option 2: HuggingFace Hub ID (after pushing from Colab)
BFNK_LOCAL = "./bfnk-bert-model"
BFNK_HF_ID = "YOUR_HF_USERNAME/bfnk-bert-fake-news"   # update after pushing to HF

MODEL_PATH = BFNK_LOCAL if os.path.exists(BFNK_LOCAL) else BFNK_HF_ID

# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

# ─── Inference ────────────────────────────────────────────────────────────────
def predict(text, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).numpy()[0]

    # id2label from training: {0: 'REAL', 1: 'FAKE'}
    id2label = model.config.id2label
    fake_idx = next((i for i, l in id2label.items() if "FAKE" in l.upper()), 1)
    real_idx = 1 - fake_idx

    fake_score = float(probs[fake_idx])
    real_score = float(probs[real_idx])

    if 0.42 <= fake_score <= 0.58:
        verdict = "UNCERTAIN"
    elif fake_score > 0.5:
        verdict = "FAKE"
    else:
        verdict = "REAL"

    return verdict, fake_score, real_score

# ─── Text Signals ─────────────────────────────────────────────────────────────
CLICKBAIT = [
    "shocking", "unbelievable", "you won't believe", "breaking", "exposed",
    "secret", "bombshell", "exclusive", "banned", "leaked", "scandal",
    "conspiracy", "deep state", "share before deleted", "censored",
    "they don't want you to know", "plandemic", "viral", "fake claim",
    "misleading", "old video", "morphed", "edited photo"
]
EMOTION = [
    "outrageous", "disgusting", "horrifying", "terrifying", "devastating",
    "absolutely", "completely", "worst ever", "unprecedented",
    "everyone knows", "never", "always"
]

def text_signals(text):
    tl = text.lower()
    blob = TextBlob(text)
    words = text.split()
    caps_pct = round(
        sum(1 for w in words if w.isupper() and len(w) > 2) / max(len(words), 1) * 100, 1
    )
    excl = text.count('!') + text.count('?!')
    pol = blob.sentiment.polarity
    sub = blob.sentiment.subjectivity
    sentiment = "Positive" if pol > 0.2 else ("Negative" if pol < -0.2 else "Neutral")
    subjectivity = "High" if sub > 0.6 else ("Med" if sub > 0.3 else "Low")
    return {
        "caps_pct": caps_pct,
        "excl": excl,
        "sentiment": sentiment,
        "subjectivity": subjectivity,
        "clickbait": [w for w in CLICKBAIT if w in tl],
        "emotion": [w for w in EMOTION if w in tl],
        "word_count": len(words),
    }

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">BharatFakeNewsKosh · BERT · Indian News</div>
    <h1>
        <span class="truth">Truth</span><span class="lens">Lens</span>
        <span class="india">India</span>
    </h1>
    <div class="hero-sub">
        Fake news detection trained exclusively on Indian news articles
        from fact-checking organizations across India
    </div>
    <div class="model-pill">★ Custom BERT · BharatFakeNewsKosh 2022 · 10,010 English Articles</div>
</div>
""", unsafe_allow_html=True)

# ─── Load Model ───────────────────────────────────────────────────────────────
model_loaded = False
if not os.path.exists(BFNK_LOCAL) and BFNK_HF_ID.startswith("YOUR_HF"):
    st.error("""
    **BFNK model not found.**

    You need to train it first:
    1. Open `BFNK_Training.ipynb` in Google Colab (T4 GPU)
    2. Run all cells (~25 mins)
    3. Download `bfnk-bert-model.zip`
    4. Unzip it → place `bfnk-bert-model/` folder next to `app.py`
    5. Restart the app

    Or push to HuggingFace Hub and update `BFNK_HF_ID` in `app.py`.
    """)
    st.stop()
else:
    with st.spinner("Loading BharatFakeNewsKosh BERT model..."):
        try:
            tokenizer, model = load_model()
            model_loaded = True
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()

# ─── Input ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">News Article or Headline</div>', unsafe_allow_html=True)

user_text = st.text_area(
    "",
    height=190,
    placeholder="Paste the English news headline or article text here...\n\nExamples:\n• Old video of Amit Shah being confronted shared as recent event\n• Government confirms new policy to boost rural employment",
    label_visibility="collapsed"
)

source_url = st.text_input(
    "",
    placeholder="Source URL (optional) — e.g. https://altnews.in/...",
    label_visibility="collapsed"
)

go = st.button("▶ Analyze with BharatFakeNewsKosh Model")

# ─── Prediction ───────────────────────────────────────────────────────────────
if go:
    if not user_text.strip():
        st.warning("Please enter some text to analyze.")
    elif len(user_text.strip()) < 15:
        st.warning("Please enter at least 15 characters for a reliable result.")
    else:
        with st.spinner("Analyzing..."):
            time.sleep(0.2)
            verdict, fake_score, real_score = predict(user_text, tokenizer, model)
            sig = text_signals(user_text)

        cls = verdict.lower()
        fake_pct = round(fake_score * 100, 1)
        real_pct = round(real_score * 100, 1)
        conf_pct = round(max(fake_score, real_score) * 100, 1)

        icons = {"FAKE": "⚠", "REAL": "✓", "UNCERTAIN": "~"}
        descs = {
            "FAKE": f"The model predicts this is likely FAKE news with {fake_pct}% probability.",
            "REAL": f"The model predicts this is likely REAL news with {real_pct}% probability.",
            "UNCERTAIN": f"The model is uncertain — fake probability {fake_pct}% vs real {real_pct}%. Verify manually.",
        }

        # Build flags HTML
        flags_html = (
            "".join(f'<span class="tag high">{w}</span>' for w in sig["clickbait"]) +
            "".join(f'<span class="tag medium">{w}</span>' for w in sig["emotion"]) +
            (f'<span class="tag low">⚠ {sig["excl"]} exclamation(s)</span>' if sig["excl"] > 1 else "") +
            f'<span class="tag low">{sig["word_count"]} words</span>'
        )
        if not sig["clickbait"] and not sig["emotion"] and sig["excl"] <= 1:
            flags_html += '<span class="tag low">No major language red flags</span>'

        st.markdown(f"""
        <div class="result-card">

            <!-- Verdict -->
            <div class="result-top {cls}">
                <div class="verdict-eyebrow">BFNK Model Verdict</div>
                <div class="verdict-text {cls}">{icons[verdict]} {verdict} NEWS</div>
                <div class="verdict-desc">{descs[verdict]}</div>

                <!-- Confidence bar -->
                <div class="conf-section">
                    <div class="conf-header">
                        <span>Model Confidence</span>
                        <span>{conf_pct}%</span>
                    </div>
                    <div class="conf-track">
                        <div class="conf-fill {cls}" style="width:{conf_pct}%"></div>
                    </div>
                </div>

                <!-- Dual probability -->
                <div class="dual-bar-wrap">
                    <div class="dual-bar-label">
                        <span>Real {real_pct}%</span>
                        <span>Fake {fake_pct}%</span>
                    </div>
                    <div class="dual-bar-track">
                        <div class="dual-real" style="width:{real_pct}%"></div>
                        <div class="dual-fake" style="width:{fake_pct}%"></div>
                    </div>
                </div>
            </div>

            <!-- Metrics -->
            <div class="metrics-section">
                <div class="metric-cell">
                    <div class="metric-val {'warn' if sig['caps_pct'] > 10 else ''}">{sig['caps_pct']}%</div>
                    <div class="metric-key">ALL CAPS</div>
                </div>
                <div class="metric-cell">
                    <div class="metric-val">{sig['sentiment']}</div>
                    <div class="metric-key">Sentiment</div>
                </div>
                <div class="metric-cell">
                    <div class="metric-val {'warn' if sig['subjectivity'] == 'High' else ''}">{sig['subjectivity']}</div>
                    <div class="metric-key">Subjectivity</div>
                </div>
            </div>

            <!-- Flagged words -->
            <div class="flags-section">
                <div class="flags-title">Flagged Language Patterns</div>
                {flags_html}
            </div>

            <!-- Model info -->
            <div class="model-footer">
                <div class="model-footer-label">Model</div>
                <div class="model-footer-name">
                    BERT fine-tuned on BharatFakeNewsKosh · 10,010 English Indian news articles · 2022
                    {f' &nbsp;·&nbsp; 🔗 {source_url[:40]}' if source_url.strip() else ''}
                </div>
            </div>

        </div>
        """, unsafe_allow_html=True)

        # Extra tips
        if verdict == "UNCERTAIN":
            st.info("💡 The model is not confident. Cross-check with **Alt News**, **Boom Live**, or **FactChecker.in**.")
        if sig["word_count"] < 20:
            st.info("💡 Short text — paste the full article headline + body for a more accurate result.")
        if verdict == "FAKE" and conf_pct > 85:
            st.error("🚨 High-confidence fake news detected. Do not share this content without verification.")

# ─── Divider ─────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ─── Info sections ────────────────────────────────────────────────────────────
with st.expander("About the Model"):
    st.markdown("""
    **Training data:** BharatFakeNewsKosh — India's first public benchmark fake news dataset

    | Detail | Value |
    |---|---|
    | English articles | 10,010 |
    | Real news (Label = True) | 6,046 (60.4%) |
    | Fake news (Label = False) | 3,964 (39.6%) |
    | Text input | Statement + News Body combined |
    | Base model | `bert-base-uncased` |
    | Max token length | 256 |
    | Training epochs | 3 |
    | Sources | Alt News, Boom Live, FactChecker.in, and 14 other IFCN-certified fact-checkers |
    | Topics | Politics, society, health, COVID, elections, viral claims |

    **Label meaning:**
    - `REAL (0)` — The original claim was fact-checked and found to be TRUE
    - `FAKE (1)` — The original claim was fact-checked and found to be FALSE/MISLEADING

    **Verdict thresholds:**
    - Fake probability > 58% → **FAKE**
    - Fake probability < 42% → **REAL**
    - Between 42–58% → **UNCERTAIN** (model not confident enough)
    """)

with st.expander("Try Example Headlines"):
    examples = {
        "🔴 Likely Fake": "OLD VIDEO: Amit Shah being confronted by reporter is being shared again as a recent 2024 election incident — VIRAL",
        "🔵 Likely Fake (Health)": "Government secretly added microchips to COVID vaccines to control the population — doctors told to stay silent",
        "🟢 Likely Real": "The Reserve Bank of India held interest rates steady at its latest monetary policy committee meeting citing easing inflation.",
        "🟡 Uncertain / Borderline": "A video allegedly showing election booth tampering in Uttar Pradesh is being widely circulated on WhatsApp.",
    }
    for label, text in examples.items():
        st.markdown(f"**{label}**")
        st.code(text, language=None)

with st.expander("How to Deploy (Streamlit Cloud)"):
    st.markdown("""
    **Option A — Local model (recommended for private use):**
    1. Unzip `bfnk-bert-model.zip` from your Colab training
    2. Place `bfnk-bert-model/` folder next to `app.py`
    3. Push everything to GitHub (the model folder too)
    4. Deploy on [share.streamlit.io](https://share.streamlit.io)

    > ⚠️ Model folder is ~430MB — GitHub has a 100MB per-file limit.
    > Use [Git LFS](https://git-lfs.com/) or Option B instead.

    **Option B — HuggingFace Hub (recommended for Streamlit Cloud):**
    1. Run Cell 13 in `BFNK_Training.ipynb` (push to HF Hub)
    2. In `app.py`, update: `BFNK_HF_ID = "your-username/bfnk-bert-fake-news"`
    3. Push `app.py` + `requirements.txt` to GitHub (no model files needed)
    4. Deploy on Streamlit Cloud — it downloads the model from HF automatically
    """)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    TruthLens India &nbsp;·&nbsp; Powered by BharatFakeNewsKosh &nbsp;·&nbsp; BERT Fine-Tuned<br>
    For educational use only &nbsp;·&nbsp; Always verify with trusted Indian fact-checkers<br>
    Alt News · Boom Live · FactChecker.in · The Quint WebQoof
</div>
""", unsafe_allow_html=True)
