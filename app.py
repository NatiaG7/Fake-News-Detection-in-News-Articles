"""Streamlit app for fake news detection."""

import streamlit as st

from src.predict import predict_text

st.set_page_config(page_title="Fake News Detector", page_icon="🕵️", layout="centered")

st.title("Fake News Detector")
st.write(
    "Paste a news article below. The model uses TF-IDF features and "
    "Logistic Regression to estimate whether the text looks **real** or **fake**."
)

user_input = st.text_area("Article text", height=250, placeholder="Paste article text here...")

if st.button("Analyze article", type="primary"):
    if not user_input.strip():
        st.warning("Please paste some text first.")
    else:
        result = predict_text(user_input)
        if result["is_real"]:
            st.success("Prediction: This looks like **REAL** news.")
        else:
            st.error("Prediction: This looks like **FAKE** news.")

with st.expander("About this model"):
    st.markdown(
        """
        - **Task:** Binary text classification (fake vs real news)
        - **Features:** TF-IDF (5,000 terms, English stop words)
        - **Model:** Logistic Regression
        - **Typical test accuracy:** ~97.7% on held-out split (ISOT-style dataset)
        """
    )
