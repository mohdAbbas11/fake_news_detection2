import os
import sys
import pandas as pd
import numpy as np
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Create directories if they don't exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

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

# Function to load and prepare data from fake.csv and true.csv
def load_data():
    '''
    Load and prepare data from fake.csv and true.csv for model training
    
    Returns:
        pd.DataFrame: Prepared dataset with 'text' and 'label' columns
    '''
    # Define file paths
    fake_csv_path = os.path.join(DATA_DIR, 'fake.csv')
    true_csv_path = os.path.join(DATA_DIR, 'true.csv')
    
    # Check if both files exist
    if not os.path.exists(fake_csv_path) or not os.path.exists(true_csv_path):
        print(f"Error: Required files not found.")
        print(f"fake.csv exists: {os.path.exists(fake_csv_path)}")
        print(f"true.csv exists: {os.path.exists(true_csv_path)}")
        return None
    
    # Load datasets
    print(f"Loading fake news dataset from {fake_csv_path}...")
    fake_df = pd.read_csv(fake_csv_path)
    print(f"Loaded {len(fake_df)} fake news articles")
    
    print(f"Loading true news dataset from {true_csv_path}...")
    true_df = pd.read_csv(true_csv_path)
    print(f"Loaded {len(true_df)} true news articles")
    
    # Add labels to each dataset
    fake_df['is_fake'] = 1  # 1 for fake news
    true_df['is_fake'] = 0  # 0 for true news
    
    # Check if both datasets have the same columns
    fake_cols = set(fake_df.columns)
    true_cols = set(true_df.columns)
    common_cols = fake_cols.intersection(true_cols)
    
    print(f"Common columns in both datasets: {', '.join(common_cols)}")
    
    # Ensure both datasets have 'text' column
    if 'text' not in common_cols:
        print("Error: 'text' column not found in both datasets.")
        return None
    
    # Select only common columns
    columns_to_keep = list(common_cols)
    fake_df = fake_df[columns_to_keep]
    true_df = true_df[columns_to_keep]
    
    # Combine datasets
    combined_df = pd.concat([fake_df, true_df], ignore_index=True)
    
    # Remove rows with missing values
    combined_df = combined_df.dropna(subset=['text'])
    
    # Shuffle the dataset
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Combined dataset created with {len(combined_df)} samples")
    print(f"Class distribution: {combined_df['is_fake'].value_counts().to_dict()}")
    
    return combined_df

# Function to train and evaluate models
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    '''
    Train and evaluate multiple models
    
    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training labels
        y_test: Testing labels
        
    Returns:
        tuple: (trained_models, model_metrics, best_model_name)
    '''
    # Define models to train
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Support Vector Machine': LinearSVC(random_state=42)
    }
    
    # Train and evaluate each model
    trained_models = {}
    model_metrics = []
    
    for name, model in tqdm(models.items(), desc="Training models"):
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Calculate ROC AUC if the model supports predict_proba
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)
        except (AttributeError, NotImplementedError):
            # If predict_proba is not available, use decision function if available
            try:
                y_score = model.decision_function(X_test)
                roc_auc = roc_auc_score(y_test, y_score)
            except (AttributeError, NotImplementedError):
                # If neither is available, use predictions as scores
                roc_auc = roc_auc_score(y_test, y_pred)
        
        # Store metrics
        model_metrics.append({
            'model_name': name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        })
        
        # Store trained model
        trained_models[name] = model
        
        # Print results
        print(f"\n{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Real', 'Fake'], 
                    yticklabels=['Real', 'Fake'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {name}')
        plt.tight_layout()
        
        # Save confusion matrix plot
        cm_plot_path = os.path.join(PROCESSED_DATA_DIR, f"{name.lower().replace(' ', '_')}_confusion_matrix_combined.png")
        plt.savefig(cm_plot_path)
        plt.close()
    
    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame(model_metrics)
    
    # Find best model based on F1 score
    best_model_idx = metrics_df['f1'].idxmax()
    best_model_name = metrics_df.loc[best_model_idx, 'model_name']
    
    print(f"\nBest model based on F1 score: {best_model_name}")
    
    return trained_models, metrics_df, best_model_name

# Function to extract feature importance from Logistic Regression model
def extract_feature_importance(vectorizer, model):
    '''
    Extract feature importance from a Logistic Regression model
    
    Args:
        vectorizer: TF-IDF vectorizer
        model: Trained Logistic Regression model
        
    Returns:
        pd.DataFrame: DataFrame with feature importance
    '''
    # Get feature names and coefficients
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    
    # Create DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': coefficients,
        'abs_importance': np.abs(coefficients)
    })
    
    # Sort by absolute importance
    feature_importance = feature_importance.sort_values('abs_importance', ascending=False)
    
    return feature_importance

# Main function
def main():
    '''
    Main function to train and save models
    '''
    print("Starting model training on combined fake and true news datasets...")
    
    # Load data
    df = load_data()
    
    if df is None:
        print("Failed to load datasets. Exiting.")
        return
    
    # Preprocess text
    print("Preprocessing text...")
    df['processed_text'] = df['text'].apply(lambda x: preprocess_text(x))
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['is_fake'], test_size=0.2, random_state=42, stratify=df['is_fake']
    )
    
    # Create TF-IDF vectorizer
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train and evaluate models
    trained_models, metrics_df, best_model_name = train_and_evaluate_models(
        X_train_tfidf, X_test_tfidf, y_train, y_test
    )
    
    # Save model comparison
    metrics_path = os.path.join(PROCESSED_DATA_DIR, 'model_comparison_combined.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Model comparison saved to {metrics_path}")
    
    # Save best model name
    best_model_path = os.path.join(MODELS_DIR, 'best_model_combined.txt')
    with open(best_model_path, 'w') as f:
        f.write(best_model_name)
    print(f"Best model name saved to {best_model_path}")
    
    # Save feature importance for Logistic Regression
    if 'Logistic Regression' in trained_models:
        feature_importance = extract_feature_importance(
            vectorizer, trained_models['Logistic Regression']
        )
        feature_importance_path = os.path.join(PROCESSED_DATA_DIR, 'feature_importance_combined.csv')
        feature_importance.to_csv(feature_importance_path, index=False)
        print(f"Feature importance saved to {feature_importance_path}")
        
        # Plot top features
        plt.figure(figsize=(12, 8))
        
        # Top positive features (indicative of fake news)
        plt.subplot(1, 2, 1)
        top_positive = feature_importance[feature_importance['importance'] > 0].head(15)
        sns.barplot(x='importance', y='feature', data=top_positive, palette='Reds_r')
        plt.title('Top Features Indicative of Fake News')
        plt.xlabel('Coefficient')
        plt.tight_layout()
        
        # Top negative features (indicative of real news)
        plt.subplot(1, 2, 2)
        top_negative = feature_importance[feature_importance['importance'] < 0].head(15)
        sns.barplot(x='importance', y='feature', data=top_negative, palette='Blues_r')
        plt.title('Top Features Indicative of Real News')
        plt.xlabel('Coefficient')
        plt.tight_layout()
        
        # Save plot
        feature_plot_path = os.path.join(PROCESSED_DATA_DIR, 'feature_importance_combined.png')
        plt.savefig(feature_plot_path, bbox_inches='tight')
        plt.close()
        print(f"Feature importance plot saved to {feature_plot_path}")
    
    # Save all models
    for name, model in trained_models.items():
        model_filename = f"{name.lower().replace(' ', '_')}_model_combined.pkl"
        model_path = os.path.join(MODELS_DIR, model_filename)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"Model '{name}' saved to {model_path}")
    
    # Save vectorizer separately
    vectorizer_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer_combined.pkl')
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"TF-IDF vectorizer saved to {vectorizer_path}")
    
    print("Model training and evaluation complete!")

# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()