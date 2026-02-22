import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from textblob import TextBlob
import time

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TruthLens — Ensemble Fake News Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── CSS ────────────────────────────────────────────────────────────────────
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
}
* { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--paper);
    color: var(--ink);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 1rem 4rem; max-width: 820px; }

.hero {
    text-align: center;
    padding: 3rem 0 2rem;
    border-bottom: 2px solid var(--ink);
    margin-bottom: 2rem;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.6rem;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.6rem, 8vw, 4.2rem);
    font-weight: 900;
    line-height: 1;
    letter-spacing: -0.02em;
    margin: 0 0 0.4rem;
}
.hero h1 span { color: var(--accent); }
.hero-sub { font-size: 0.92rem; color: var(--muted); font-weight: 300; }
.badge-row { margin-top: 1rem; display: flex; gap: 0.5rem; justify-content: center; flex-wrap: wrap; }
.badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border: 1px solid;
}
.b1 { border-color: var(--ink); color: var(--ink); }
.b2 { border-color: #1a5276; color: #1a5276; }
.b3 { border-color: var(--accent2); color: var(--accent2); }

.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
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
    line-height: 1.7 !important;
}
textarea:focus { border-color: var(--ink) !important; box-shadow: none !important; }
input[type="text"] {
    font-family: 'DM Mono', monospace !important;
    background: #faf7f2 !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 0 !important;
    color: var(--ink) !important;
    font-size: 0.85rem !important;
}
.stButton > button {
    background: var(--ink) !important;
    color: var(--paper) !important;
    border: none !important;
    border-radius: 0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    padding: 0.75rem 2.5rem !important;
    width: 100% !important;
}
.stButton > button:hover { background: var(--accent) !important; }

.ensemble-board {
    border: 2px solid var(--ink);
    margin-top: 2rem;
    background: #faf7f2;
    overflow: hidden;
}
.ensemble-top {
    padding: 1.8rem 2rem 1.4rem;
    border-bottom: 1px solid var(--border);
    position: relative;
}
.ensemble-top::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 5px;
}
.ensemble-top.fake::before { background: var(--accent); }
.ensemble-top.real::before { background: var(--accent2); }
.ensemble-top.uncertain::before { background: var(--warn); }

.verdict-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.2rem;
}
.verdict-main {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 900;
    line-height: 1;
    margin: 0;
}
.verdict-main.fake { color: var(--accent); }
.verdict-main.real { color: var(--accent2); }
.verdict-main.uncertain { color: var(--warn); }
.verdict-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    margin-top: 0.5rem;
}
.conf-wrap { margin-top: 1.2rem; }
.conf-label {
    display: flex;
    justify-content: space-between;
    font-family: 'DM Mono', monospace;
    font-size: 0.66rem;
    color: var(--muted);
    margin-bottom: 0.3rem;
}
.conf-track { height: 8px; background: var(--cream); border: 1px solid var(--border); }
.conf-fill { height: 100%; }
.conf-fill.fake { background: var(--accent); }
.conf-fill.real { background: var(--accent2); }
.conf-fill.uncertain { background: var(--warn); }

.models-section { padding: 1.5rem 2rem; border-bottom: 1px solid var(--border); }
.models-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
}
.model-row {
    display: flex;
    align-items: center;
    margin-bottom: 1rem;
    gap: 0.75rem;
}
.model-name {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    min-width: 165px;
    color: var(--ink);
    line-height: 1.4;
}
.model-era { font-size: 0.58rem; color: var(--muted); }
.model-bar-wrap { flex: 1; }
.model-sublabel {
    display: flex;
    justify-content: space-between;
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: var(--muted);
    margin-bottom: 0.2rem;
}
.model-bar-track { height: 5px; background: var(--cream); border: 1px solid var(--border); }
.model-bar-fill { height: 100%; }
.model-bar-fill.fake { background: var(--accent); }
.model-bar-fill.real { background: var(--accent2); }
.model-bar-fill.uncertain { background: var(--warn); }
.model-chip {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    padding: 0.15rem 0.45rem;
    border: 1px solid;
    white-space: nowrap;
}
.model-chip.fake { border-color: var(--accent); color: var(--accent); }
.model-chip.real { border-color: var(--accent2); color: var(--accent2); }
.model-chip.uncertain { border-color: var(--warn); color: var(--warn); }

.signals-section { padding: 1.5rem 2rem; border-bottom: 1px solid var(--border); }
.signals-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
}
.signals-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; }
.signal-box {
    border: 1px solid var(--border);
    padding: 0.75rem;
    background: var(--cream);
    text-align: center;
}
.signal-val {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem;
    font-weight: 700;
    line-height: 1;
    color: var(--ink);
}
.signal-val.warn { color: var(--accent); }
.signal-key {
    font-family: 'DM Mono', monospace;
    font-size: 0.57rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 0.3rem;
}

.flags-section { padding: 1.2rem 2rem; }
.flags-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.65rem;
}
.tag {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    padding: 0.18rem 0.5rem;
    margin: 0.2rem;
    border: 1px solid;
}
.tag.high { border-color: var(--accent); color: var(--accent); background: rgba(192,57,43,0.06); }
.tag.medium { border-color: var(--warn); color: var(--warn); background: rgba(214,137,16,0.06); }
.tag.low { border-color: var(--muted); color: var(--muted); }

.agreement-banner {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    padding: 0.7rem 2rem;
    border-top: 1px solid var(--border);
}
.agreement-banner.agree { background: rgba(44,122,75,0.08); color: var(--accent2); }
.agreement-banner.partial { background: rgba(214,137,16,0.08); color: var(--warn); }
.agreement-banner.disagree { background: rgba(192,57,43,0.08); color: var(--accent); }

hr.divider { border: none; border-top: 1px solid var(--border); margin: 2rem 0; }
.footer {
    text-align: center;
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 2rem 0 0;
    border-top: 1px solid var(--border);
    margin-top: 3rem;
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

# ─── Model Registry ──────────────────────────────────────────────────────────
MODELS = {
    "general": {
        "id": "hamzab/roberta-fake-news-classification",
        "name": "RoBERTa General",
        "era": "2016–2018 News",
        "weight": 0.40,
    },
    "distilroberta": {
        "id": "vikram71198/distilroberta-base-finetuned-fake-news-detection",
        "name": "DistilRoBERTa Broad",
        "era": "2017–2021 News",
        "weight": 0.35,
    },
    "covid": {
        "id": "spencer-gable-cook/COVID-19_Misinformation_Detector",
        "name": "BERT COVID-19",
        "era": "2020–2022 Pandemic",
        "weight": 0.25,
    },
}

@st.cache_resource
def load_all_models():
    loaded = {}
    for key, cfg in MODELS.items():
        try:
            tok = AutoTokenizer.from_pretrained(cfg["id"])
            mdl = AutoModelForSequenceClassification.from_pretrained(cfg["id"])
            mdl.eval()
            loaded[key] = {"tokenizer": tok, "model": mdl, "cfg": cfg, "ok": True}
        except Exception as e:
            loaded[key] = {"cfg": cfg, "ok": False, "error": str(e)}
    return loaded

def get_fake_score(probs, id2label):
    """Return the probability that text is FAKE/MISINFORMATION."""
    for idx, lbl in id2label.items():
        lu = lbl.upper()
        if any(k in lu for k in ["FAKE", "FALSE", "MISINFORMATION", "MISLEADING"]):
            return float(probs[idx])
    # fallback: label_1 or index 1 is typically fake
    return float(probs[1]) if len(probs) > 1 else float(probs[0])

def infer(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
    fake_score = get_fake_score(probs, model.config.id2label)
    real_score = 1.0 - fake_score
    verdict = "FAKE" if fake_score > 0.5 else "REAL"
    confidence = fake_score if verdict == "FAKE" else real_score
    return verdict, confidence, fake_score, real_score

def ensemble_predict(text, models_dict):
    results = {}
    weighted_fake = 0.0
    total_w = 0.0
    for key, m in models_dict.items():
        if not m["ok"]:
            results[key] = {"verdict": "ERROR", "confidence": 0, "fake_score": 0.5, "real_score": 0.5}
            continue
        verdict, confidence, fake_score, real_score = infer(text, m["tokenizer"], m["model"])
        results[key] = {"verdict": verdict, "confidence": confidence, "fake_score": fake_score, "real_score": real_score}
        w = m["cfg"]["weight"]
        weighted_fake += fake_score * w
        total_w += w

    final_fake = weighted_fake / total_w if total_w > 0 else 0.5
    if 0.42 <= final_fake <= 0.58:
        verdict = "UNCERTAIN"
    elif final_fake > 0.5:
        verdict = "FAKE"
    else:
        verdict = "REAL"
    certainty = round(abs(final_fake - 0.5) * 200, 1)
    return verdict, certainty, final_fake, results

# ─── Signals ─────────────────────────────────────────────────────────────────
CLICKBAIT = ["shocking","unbelievable","you won't believe","breaking","exposed","secret",
             "bombshell","exclusive","miracle","banned","leaked","scandal","conspiracy",
             "truth revealed","hoax","mainstream media","wake up","share before deleted",
             "censored","deep state","plandemic","they don't want you to know"]
EMOTION = ["outrageous","disgusting","horrifying","terrifying","devastating","absolutely",
           "completely","totally","worst ever","best ever","unprecedented","everyone knows",
           "nobody talks about","always","never"]

def text_signals(text):
    tl = text.lower()
    blob = TextBlob(text)
    words = text.split()
    caps_pct = round(sum(1 for w in words if w.isupper() and len(w) > 2) / max(len(words), 1) * 100, 1)
    excl = text.count('!') + text.count('?!')
    pol = blob.sentiment.polarity
    sub = blob.sentiment.subjectivity
    sentiment = "Positive" if pol > 0.2 else ("Negative" if pol < -0.2 else "Neutral")
    subjectivity = "High" if sub > 0.6 else ("Med" if sub > 0.3 else "Low")
    return {
        "caps_pct": caps_pct, "excl": excl,
        "sentiment": sentiment, "subjectivity": subjectivity,
        "clickbait": [w for w in CLICKBAIT if w in tl],
        "emotion": [w for w in EMOTION if w in tl],
        "word_count": len(words),
    }

# ─── UI ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">3-Model Ensemble · AI Fact Intelligence</div>
    <h1>Truth<span>Lens</span></h1>
    <div class="hero-sub">Three specialized AI models working together to detect misinformation</div>
    <div class="badge-row">
        <span class="badge b1">RoBERTa General · 2016–2018</span>
        <span class="badge b2">DistilRoBERTa Broad · 2017–2021</span>
        <span class="badge b3">BERT COVID-19 · 2020–2022</span>
    </div>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading 3 AI models — ~60s on first run..."):
    models_dict = load_all_models()

loaded_n = sum(1 for m in models_dict.values() if m["ok"])
if loaded_n == 0:
    st.error("All models failed to load. Check internet connection.")
    st.stop()
elif loaded_n < 3:
    st.warning(f"⚠ Only {loaded_n}/3 models loaded. Results may be less accurate.")

st.markdown('<div class="section-label">Article or Headline to Analyze</div>', unsafe_allow_html=True)
user_text = st.text_area("", height=180, placeholder="Paste news text here. Longer text = better accuracy...", label_visibility="collapsed")
source_url = st.text_input("", placeholder="Source URL (optional)", label_visibility="collapsed")
go = st.button("▶ Run Ensemble Analysis")

if go:
    if not user_text.strip() or len(user_text.strip()) < 20:
        st.warning("Please enter at least 20 characters.")
    else:
        with st.spinner(f"Running {loaded_n} models..."):
            time.sleep(0.3)
            verdict, certainty, final_fake, per_model = ensemble_predict(user_text, models_dict)
            sig = text_signals(user_text)

        cls = verdict.lower()
        fake_pct = round(final_fake * 100, 1)
        real_pct = 100 - fake_pct
        icons = {"FAKE": "⚠", "REAL": "✓", "UNCERTAIN": "~"}
        subs = {
            "FAKE": f"Ensemble weighted fake probability: {fake_pct}%",
            "REAL": f"Ensemble weighted real probability: {real_pct}%",
            "UNCERTAIN": f"Models are split — fake probability: {fake_pct}% — verify manually",
        }

        valid_v = [r["verdict"] for r in per_model.values() if r["verdict"] != "ERROR"]
        all_agree = len(set(valid_v)) == 1
        majority = valid_v.count(verdict) >= 2 if len(valid_v) >= 2 else True
        if all_agree:
            agr_cls, agr_msg = "agree", f"✓ All {len(valid_v)} models agree: {verdict}"
        elif majority:
            agr_cls, agr_msg = "partial", f"~ {valid_v.count(verdict)}/{len(valid_v)} models agree on {verdict} — minority dissent"
        else:
            agr_cls, agr_msg = "disagree", "⚠ Models strongly disagree — treat this result with extra caution"

        # Build model rows HTML
        model_rows_html = ""
        for key, r in per_model.items():
            if r["verdict"] == "ERROR":
                continue
            cfg = models_dict[key]["cfg"]
            fp = round(r["fake_score"] * 100)
            rp = 100 - fp
            model_rows_html += f"""
            <div class="model-row">
                <div class="model-name">
                    {cfg["name"]}<br>
                    <span class="model-era">{cfg["era"]} · w={cfg["weight"]}</span>
                </div>
                <div class="model-bar-wrap">
                    <div class="model-sublabel">
                        <span>Real {rp}%</span><span>Fake {fp}%</span>
                    </div>
                    <div class="model-bar-track">
                        <div class="model-bar-fill {r['verdict'].lower()}" style="width:{fp}%"></div>
                    </div>
                </div>
                <span class="model-chip {r['verdict'].lower()}">{r['verdict']}</span>
            </div>"""

        # Flagged words
        flags_html = (
            "".join(f'<span class="tag high">{w}</span>' for w in sig["clickbait"]) +
            "".join(f'<span class="tag medium">{w}</span>' for w in sig["emotion"]) +
            (f'<span class="tag low">⚠ {sig["excl"]} exclamations</span>' if sig["excl"] > 1 else "") +
            f'<span class="tag low">{sig["word_count"]} words</span>' +
            (f'<span class="tag low">🔗 {source_url[:35]}...</span>' if source_url.strip() else "")
        )

        st.markdown(f"""
        <div class="ensemble-board">
            <div class="ensemble-top {cls}">
                <div class="verdict-label">Ensemble Verdict</div>
                <div class="verdict-main {cls}">{icons.get(verdict,"?")} {verdict} NEWS</div>
                <div class="verdict-sub">{subs.get(verdict,"")}</div>
                <div class="conf-wrap">
                    <div class="conf-label"><span>Model Certainty</span><span>{certainty}%</span></div>
                    <div class="conf-track">
                        <div class="conf-fill {cls}" style="width:{certainty}%"></div>
                    </div>
                </div>
            </div>

            <div class="models-section">
                <div class="models-title">Individual Model Breakdown</div>
                {model_rows_html}
            </div>

            <div class="signals-section">
                <div class="signals-title">Linguistic Signals</div>
                <div class="signals-grid">
                    <div class="signal-box">
                        <div class="signal-val {'warn' if sig['caps_pct'] > 10 else ''}">{sig['caps_pct']}%</div>
                        <div class="signal-key">ALL CAPS</div>
                    </div>
                    <div class="signal-box">
                        <div class="signal-val {'warn' if sig['excl'] > 2 else ''}">{sig['excl']}</div>
                        <div class="signal-key">Exclamations</div>
                    </div>
                    <div class="signal-box">
                        <div class="signal-val">{sig['sentiment']}</div>
                        <div class="signal-key">Sentiment</div>
                    </div>
                    <div class="signal-box">
                        <div class="signal-val {'warn' if sig['subjectivity'] == 'High' else ''}">{sig['subjectivity']}</div>
                        <div class="signal-key">Subjectivity</div>
                    </div>
                </div>
            </div>

            <div class="flags-section">
                <div class="flags-title">Flagged Language</div>
                {flags_html}
            </div>

            <div class="agreement-banner {agr_cls}">{agr_msg}</div>
        </div>
        """, unsafe_allow_html=True)

        if verdict == "UNCERTAIN":
            st.info("💡 The models are split. This could be satire, opinion, or a specialized topic. Always verify with primary sources.")
        if sig["word_count"] < 30:
            st.info("💡 Short text — longer articles produce more reliable ensemble results.")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

with st.expander("How the 3-Model Ensemble Works"):
    st.markdown("""
    **Why 3 models?** Each model was trained on different data from different eras. Combining them covers a wider range of misinformation patterns.

    | Model | Training Data | Specialization | Weight |
    |---|---|---|---|
    | RoBERTa General | WELFake 72K articles (2016–2018) | Political & general fake news | 40% |
    | DistilRoBERTa Broad | 32K multi-source (2017–2021) | Broad domain coverage | 35% |
    | BERT COVID-19 | Anti-Vax + CoAID tweets (2020–2022) | Health & pandemic misinfo | 25% |

    **Ensemble formula:** `Final Score = (0.40 × M1) + (0.35 × M2) + (0.25 × M3)`

    If the weighted fake probability falls between 42–58%, the verdict is **UNCERTAIN** — meaning the models genuinely disagree and you should verify manually.
    """)

with st.expander("Try Example Headlines"):
    examples = {
        "🔴 Fake (Political)": "SHOCKING: Government SECRETLY installing mind control chips in COVID vaccines — SHARE BEFORE DELETED!",
        "🔵 Fake (Health)": "Doctors BANNED from revealing that drinking bleach cures all diseases. Big Pharma conspiracy EXPOSED!",
        "🟢 Real": "Federal Reserve raises interest rates by 25 basis points amid concerns about persistent core inflation.",
        "🟡 Borderline": "New study suggests coffee may have unexpected health benefits when consumed in moderation, researchers say.",
    }
    for label, text in examples.items():
        st.markdown(f"**{label}**")
        st.code(text, language=None)

st.markdown("""
<div class="footer">
    TruthLens Ensemble · 3 AI Models · Streamlit · Free on Streamlit Cloud<br>
    Educational use only · Always verify with trusted fact-checkers
</div>
""", unsafe_allow_html=True)
