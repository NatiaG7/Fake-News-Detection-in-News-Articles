"""
Model Trainer Module
Trains multiple classification models for fake news detection
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

class ModelTrainer:
    """Train and manage multiple models"""
    
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        self.models = {}
        self.scores = {}
    
    def fit_vectorizer(self, X_train):
        """Fit TF-IDF vectorizer"""
        self.vectorizer.fit(X_train)
        return self
    
    def transform(self, X):
        """Transform text using fitted vectorizer"""
        return self.vectorizer.transform(X)
    
    def train_logistic_regression(self, X_train_vec, y_train):
        """Train Logistic Regression model"""
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_vec, y_train)
        self.models['logistic_regression'] = model
        return model
    
    def train_naive_bayes(self, X_train_vec, y_train):
        """Train Naive Bayes model"""
        model = MultinomialNB()
        model.fit(X_train_vec, y_train)
        self.models['naive_bayes'] = model
        return model
    
    def train_random_forest(self, X_train_vec, y_train):
        """Train Random Forest model"""
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_vec, y_train)
        self.models['random_forest'] = model
        return model
    
    def train_all(self, X_train, y_train):
        """Train all three models"""
        X_train_vec = self.fit_vectorizer(X_train).transform(X_train)
        
        self.train_logistic_regression(X_train_vec, y_train)
        self.train_naive_bayes(X_train_vec, y_train)
        self.train_random_forest(X_train_vec, y_train)
        
        return self.models
    
    def evaluate(self, X_test, y_test):
        """Evaluate all trained models"""
        X_test_vec = self.transform(X_test)
        scores = {}
        
        for name, model in self.models.items():
            score = model.score(X_test_vec, y_test)
            scores[name] = score
        
        self.scores = scores
        return scores
    
    def save_models(self, models_dir="models"):
        """Save trained models"""
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(self.models, os.path.join(models_dir, "models.pkl"))
        joblib.dump(self.vectorizer, os.path.join(models_dir, "vectorizer.pkl"))
    
    def load_models(self, models_dir="models"):
        """Load pre-trained models"""
        self.models = joblib.load(os.path.join(models_dir, "models.pkl"))
        self.vectorizer = joblib.load(os.path.join(models_dir, "vectorizer.pkl"))