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
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = CURRENT_DIR

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

# Function to create a simple dataset if no real data is available
def create_sample_dataset():
    '''
    Create a sample dataset for demonstration purposes
    
    Returns:
        pd.DataFrame: Sample dataset with 'text' and 'label' columns
    '''
    print("Creating sample dataset for demonstration...")
    
    # Sample fake news articles
    fake_news = [
        "BREAKING: Prime Minister resigns after secret scandal revealed by anonymous source.",
        "Scientists discover miracle cure for all diseases, pharmaceutical companies trying to hide it.",
        "SHOCKING: Famous actor found to be alien in disguise, government covering up the truth.",
        "Election RIGGED! Thousands of fake votes found in secret location.",
        "COVID-19 vaccine contains microchips to track citizens, insider reveals.",
        "Celebrity announces they've found the fountain of youth, looks 30 years younger overnight.",
        "Government hiding evidence of UFO crash, whistleblower leaks documents.",
        "New study finds common food causes cancer, media silent about it.",
        "Bank glitch makes everyone millionaires for 24 hours, cover-up in progress.",
        "Secret society controlling world governments exposed in leaked emails."
    ]
    
    # Sample real news articles
    real_news = [
        "Parliament passes new education bill with bipartisan support after lengthy debate.",
        "Research team publishes findings on potential new treatment for diabetes in medical journal.",
        "Film star wins award for performance in historical drama, thanks cast and crew.",
        "Election results confirmed after official count, new representatives to take office next month.",
        "Health ministry releases guidelines for COVID-19 prevention based on latest research.",
        "Actor discusses preparation for challenging role in upcoming biographical film.",
        "Space agency successfully launches satellite to monitor climate change patterns.",
        "Nutritionists recommend balanced diet including newly studied superfood according to research.",
        "Central bank announces interest rate decision after quarterly economic review.",
        "International summit concludes with agreement on trade policies between participating nations."
    ]
    
    # Create DataFrame
    fake_df = pd.DataFrame({'text': fake_news, 'label': 1})  # 1 for fake
    real_df = pd.DataFrame({'text': real_news, 'label': 0})  # 0 for real
    
    # Combine and shuffle
    df = pd.concat([fake_df, real_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

# Function to load and prepare data
def load_data():
    '''
    Load and prepare data for model training
    
    Returns:
        pd.DataFrame: Prepared dataset with 'text' and 'label' columns
    '''
    # Check if we have real datasets
    indian_fake_news_path = os.path.join(RAW_DATA_DIR, "indian_fake_news.csv")
    fake_news_path = os.path.join(RAW_DATA_DIR, "Fake.csv")
    true_news_path = os.path.join(RAW_DATA_DIR, "True.csv")
    
    # If we have the Kaggle dataset
    if os.path.exists(fake_news_path) and os.path.exists(true_news_path):
        print("Loading Kaggle's Fake and Real News Dataset...")
        
        # Load datasets
        fake_df = pd.read_csv(fake_news_path)
        real_df = pd.read_csv(true_news_path)
        
        # Add labels
        fake_df['label'] = 1  # 1 for fake
        real_df['label'] = 0  # 0 for real
        
        # Combine text and title
        fake_df['text'] = fake_df['title'] + " " + fake_df['text']
        real_df['text'] = real_df['title'] + " " + real_df['text']
        
        # Select relevant columns
        fake_df = fake_df[['text', 'label']]
        real_df = real_df[['text', 'label']]
        
        # Combine datasets
        df = pd.concat([fake_df, real_df], ignore_index=True)
        
        # If we also have the Indian dataset, add it
        if os.path.exists(indian_fake_news_path):
            print("Adding Indian Fake News Dataset...")
            indian_df = pd.read_csv(indian_fake_news_path)
            
            # Map labels (assuming 'fake' is labeled as 1 and 'real' as 0)
            label_map = {'fake': 1, 'real': 0}
            indian_df['label'] = indian_df['label'].map(label_map)
            
            # Select relevant columns (adjust column names as needed)
            if 'content' in indian_df.columns:
                indian_df['text'] = indian_df['content']
            elif 'text' not in indian_df.columns:
                # If neither 'content' nor 'text' exists, try to find the text column
                text_columns = [col for col in indian_df.columns if col.lower() in ['article', 'news', 'body']]
                if text_columns:
                    indian_df['text'] = indian_df[text_columns[0]]
                else:
                    print("Could not identify text column in Indian dataset, skipping...")
                    indian_df = None
            
            if indian_df is not None:
                indian_df = indian_df[['text', 'label']]
                df = pd.concat([df, indian_df], ignore_index=True)
    
    # If we only have the Indian dataset
    elif os.path.exists(indian_fake_news_path):
        print("Loading Indian Fake News Dataset...")
        df = pd.read_csv(indian_fake_news_path)
        
        # Map labels (assuming 'fake' is labeled as 1 and 'real' as 0)
        label_map = {'fake': 1, 'real': 0}
        df['label'] = df['label'].map(label_map)
        
        # Select relevant columns (adjust column names as needed)
        if 'content' in df.columns:
            df['text'] = df['content']
        elif 'text' not in df.columns:
            # If neither 'content' nor 'text' exists, try to find the text column
            text_columns = [col for col in df.columns if col.lower() in ['article', 'news', 'body']]
            if text_columns:
                df['text'] = df[text_columns[0]]
            else:
                print("Could not identify text column in Indian dataset, using sample data instead...")
                df = create_sample_dataset()
        
        df = df[['text', 'label']]
    
    # If no real datasets are available, create a sample dataset
    else:
        df = create_sample_dataset()
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Remove rows with empty text
    df = df[df['text'].notna() & (df['text'] != '')]
    
    # Limit dataset size for demonstration (if it's too large)
    if len(df) > 10000:
        df = df.sample(10000, random_state=42)
    
    print(f"Dataset loaded with {len(df)} samples")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")
    
    return df

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
        cm_plot_path = os.path.join(PROCESSED_DATA_DIR, f"{name.lower().replace(' ', '_')}_confusion_matrix.png")
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
    print("Starting model training...")
    
    # Load data
    df = load_data()
    
    # Preprocess text
    print("Preprocessing text...")
    df['processed_text'] = df['text'].apply(lambda x: preprocess_text(x))
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['label'], test_size=0.2, random_state=42
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
    metrics_path = os.path.join(PROCESSED_DATA_DIR, 'model_comparison.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Model comparison saved to {metrics_path}")
    
    # Save best model name
    best_model_path = os.path.join(MODELS_DIR, 'best_model.txt')
    with open(best_model_path, 'w') as f:
        f.write(best_model_name)
    print(f"Best model name saved to {best_model_path}")
    
    # Save feature importance for Logistic Regression
    if 'Logistic Regression' in trained_models:
        feature_importance = extract_feature_importance(
            vectorizer, trained_models['Logistic Regression']
        )
        feature_importance_path = os.path.join(PROCESSED_DATA_DIR, 'feature_importance.csv')
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
        feature_plot_path = os.path.join(PROCESSED_DATA_DIR, 'feature_importance.png')
        plt.savefig(feature_plot_path, bbox_inches='tight')
        plt.close()
        print(f"Feature importance plot saved to {feature_plot_path}")
    
    # Save all models
    for name, model in trained_models.items():
        model_filename = f"{name.lower().replace(' ', '_')}_model.pkl"
        model_path = os.path.join(MODELS_DIR, model_filename)
        
        with open(model_path, 'wb') as f:
            pickle.dump((model, vectorizer), f)
        
        print(f"Model '{name}' saved to {model_path}")
    
    print("Model training and evaluation complete!")

# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()