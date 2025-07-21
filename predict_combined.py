import os
import sys
import pickle
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Set paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load stop words
stop_words = set(stopwords.words('english'))

# Text preprocessing functions
def clean_text(text):
    '''
    Basic text cleaning function
    
    Args:
        text (str): Input text
        
    Returns:
        str: Cleaned text
    '''
    if not isinstance(text, str):
        return ''
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_text(text):
    '''
    Tokenize text into words
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of tokens
    '''
    return word_tokenize(text)

def remove_stopwords(tokens):
    '''
    Remove stopwords from token list
    
    Args:
        tokens (list): List of tokens
        
    Returns:
        list: List of tokens without stopwords
    '''
    return [token for token in tokens if token not in stop_words]

def lemmatize_tokens(tokens):
    '''
    Lemmatize tokens
    
    Args:
        tokens (list): List of tokens
        
    Returns:
        list: List of lemmatized tokens
    '''
    return [lemmatizer.lemmatize(token) for token in tokens]

def preprocess_text(text, remove_stop=True, lemmatize=True):
    '''
    Complete text preprocessing pipeline
    
    Args:
        text (str): Input text
        remove_stop (bool): Whether to remove stopwords
        lemmatize (bool): Whether to lemmatize tokens
        
    Returns:
        str: Preprocessed text
    '''
    # Clean text
    cleaned_text = clean_text(text)
    
    # Tokenize
    tokens = tokenize_text(cleaned_text)
    
    # Remove stopwords if requested
    if remove_stop:
        tokens = remove_stopwords(tokens)
    
    # Lemmatize if requested
    if lemmatize:
        tokens = lemmatize_tokens(tokens)
    
    # Join tokens back into text
    return ' '.join(tokens)

# Function to load the best model
def load_best_model():
    '''
    Load the best model and vectorizer
    
    Returns:
        tuple: (model, vectorizer)
    '''
    # Read best model name
    best_model_path = os.path.join(MODELS_DIR, 'best_model_combined.txt')
    
    if not os.path.exists(best_model_path):
        print(f"Error: Best model file not found at {best_model_path}")
        return None, None
    
    with open(best_model_path, 'r') as f:
        best_model_name = f.read().strip()
    
    # Load the best model
    model_filename = f"{best_model_name.lower().replace(' ', '_')}_model_combined.pkl"
    model_path = os.path.join(MODELS_DIR, model_filename)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None, None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load the vectorizer
    vectorizer_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer_combined.pkl')
    
    if not os.path.exists(vectorizer_path):
        print(f"Error: Vectorizer file not found at {vectorizer_path}")
        return None, None
    
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    return model, vectorizer

# Function to predict if a news article is fake or real
def predict_fake_news(text, model, vectorizer):
    '''
    Predict if a news article is fake or real
    
    Args:
        text (str): News article text
        model: Trained model
        vectorizer: TF-IDF vectorizer
        
    Returns:
        tuple: (prediction, confidence)
    '''
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Vectorize the text
    text_tfidf = vectorizer.transform([processed_text])
    
    # Make prediction
    prediction = model.predict(text_tfidf)[0]
    
    # Get confidence score
    confidence = None
    try:
        # For models that support predict_proba
        proba = model.predict_proba(text_tfidf)[0]
        confidence = proba[1] if prediction == 1 else proba[0]
    except (AttributeError, NotImplementedError):
        try:
            # For models that support decision_function
            decision = model.decision_function(text_tfidf)[0]
            confidence = abs(decision) / 2  # Normalize to [0, 1] range
        except (AttributeError, NotImplementedError):
            # If neither is available
            confidence = 0.5
    
    # 1 means fake news, 0 means real news
    
    return prediction, confidence

# Main function
def main():
    '''
    Main function to demonstrate prediction
    '''
    print("Loading model and vectorizer...")
    model, vectorizer = load_best_model()
    
    if model is None or vectorizer is None:
        print("Failed to load model or vectorizer. Exiting.")
        return
    
    # Example news articles
    examples = [
        "Scientists discover new species of deep-sea fish in the Pacific Ocean. The discovery was made during a research expedition led by marine biologists from the University of California.",
        "BREAKING: Celebrity secretly admits to being an alien from Mars! Sources close to the star reveal shocking details about their extraterrestrial origins and plans to contact their home planet.",
        "Government announces new tax policy to take effect next fiscal year. Economic experts predict it will lead to a 2% increase in GDP over the next five years."
    ]
    
    print("\nPredicting fake news for example articles...")
    for i, text in enumerate(examples):
        prediction, confidence = predict_fake_news(text, model, vectorizer)
        
        print(f"\nExample {i+1}:")
        print(f"Text: {text[:100]}...")
        print(f"Prediction: {'FAKE' if prediction == 1 else 'REAL'} news")
        print(f"Confidence: {confidence:.4f}")

# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()