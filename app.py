import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

st.title("Fake News Detector")
st.write("This app detects whether a news article is real or fake using a BERT model trained on BharatFakeNewsKosh dataset.")

# Model is in the bfnk-bert-model folder in the same repo
MODEL_PATH = "./bfnk-bert-model"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Input
st.subheader("Enter News Text")
news_text = st.text_area("Paste the news article or headline here:", height=200)

if st.button("Check"):
    if news_text.strip() == "":
        st.warning("Please enter some text first.")
    else:
        # Predict
        inputs = tokenizer(news_text, return_tensors="pt", truncation=True, max_length=256, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).numpy()[0]

        real_score = round(float(probs[0]) * 100, 2)
        fake_score = round(float(probs[1]) * 100, 2)

        st.subheader("Result")

        if fake_score > 58:
            st.error(f"FAKE NEWS — {fake_score}% confidence")
        elif real_score > 58:
            st.success(f"REAL NEWS — {real_score}% confidence")
        else:
            st.warning(f"UNCERTAIN — Real: {real_score}% | Fake: {fake_score}%")

        st.write("---")
        st.write(f"Real probability: {real_score}%")
        st.write(f"Fake probability: {fake_score}%")
