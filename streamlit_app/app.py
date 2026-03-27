import streamlit as st
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import DataLoader
from src.data.preprocessor import TextPreprocessor
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator
import joblib

st.set_page_config(page_title="Fake News Detector", layout="wide")

st.markdown("""
# 🔍 Fake News Detection System
Detect misinformation in news articles using Machine Learning

**Project Goal:** Classify news articles as Real or Fake with high accuracy (F1 ≥ 0.85)
"""
)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Home", "Predict", "Model Performance", "About"])

if page == "Home":
    st.header("Welcome to the Fake News Detection Project")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Project Overview")
        st.write("""
        This application uses machine learning to identify fake news articles.
        
        **Features:**
        - Real-time prediction on user input
        - Multiple model comparison
        - Performance metrics dashboard
        - Text preprocessing pipeline
        """
)
    
    with col2:
        st.subheader("🎯 Success Metrics")
        st.write("""
        Target Performance:
        - **F1-Score:** ≥ 0.85
        - **Accuracy:** High and balanced
        - **Precision & Recall:** Balanced across classes
        """
)

elif page == "Predict":
    st.header("🔮 Predict News Article")
    
    st.write("Enter news content below to check if it's real or fake:")
    
    input_type = st.radio("Select input type:", ["Title + Text", "Title Only", "Text Only"])
    
    if input_type == "Title + Text":
        title = st.text_input("Article Title:")
        text = st.text_area("Article Text:")
        user_input = title + " " + text
    elif input_type == "Title Only":
        title = st.text_input("Article Title:")
        user_input = title
    else:
        text = st.text_area("Article Text:")
        user_input = text
    
    if st.button("Predict", key="predict_btn"):
        if not user_input.strip():
            st.warning("Please enter some text to analyze")
        else:
            st.info("Processing your input...")
            
            # This is a placeholder - actual prediction logic will be added
            st.success("✅ Prediction Complete")
            st.write("Prediction logic will be implemented once models are trained")

elif page == "Model Performance":
    st.header("📈 Model Performance Dashboard")
    
    st.write("This section will display:")
    st.markdown("""
    - Model comparison across all trained models
    - Accuracy, Precision, Recall, F1-Score metrics
    - Confusion matrices
    - Feature importance analysis
    - Cross-validation results
    """)

elif page == "About":
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ### Project Proposal: Fake News Detection in News Articles
    **Student:** Natia Gogitidze  
    **Institution:** Fullstack Academy (AI/ML Cohort 2510-FTB-CT-AIM-PT)  
    **Instructor:** Dr. George Perdrizet  
    **TA:** Andrew Thomas
    
    #### Problem Statement
    Online news platforms contain large amounts of misinformation. This project aims to detect 
    fake news by analyzing textual content using NLP and machine learning techniques.
    
    #### Approach
    1. **Data Collection:** Kaggle Fake and Real News Dataset (~44,000 articles)
    2. **Preprocessing:** Text cleaning, tokenization, stopword removal
    3. **Feature Extraction:** TF-IDF vectorization
    4. **Model Training:** Logistic Regression, Naive Bayes, Random Forest
    5. **Evaluation:** Comprehensive metrics and cross-validation
    6. **UI:** Streamlit web application for real-time predictions
    
    #### Success Criteria
    - F1-Score ≥ 0.85
    - Balanced precision and recall
    - Good performance on unseen test data
    """
)