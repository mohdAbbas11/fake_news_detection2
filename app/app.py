import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Try to import TensorFlow, but handle the case when it's not available
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.warning("TensorFlow is not installed. LSTM and BERT models will not be available.")


# Set page configuration
st.set_page_config(
    page_title="Fake News Detector - India",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Define paths
MODELS_DIR = os.path.join('..', 'models')
DATA_DIR = os.path.join('..', 'data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

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

# Load models
@st.cache_resource
def load_models():
    models = {}
    vectorizers = {}
    
    # Try to load the best model name (original and combined)
    best_model_path = os.path.join(MODELS_DIR, 'best_model.txt')
    best_model_combined_path = os.path.join(MODELS_DIR, 'best_model_combined.txt')
    best_model_name = None
    best_model_combined_name = None
    
    if os.path.exists(best_model_path):
        with open(best_model_path, 'r') as f:
            best_model_name = f.read().strip()
            
    if os.path.exists(best_model_combined_path):
        with open(best_model_combined_path, 'r') as f:
            best_model_combined_name = f.read().strip()
            # Append "Combined" to distinguish from original models
            best_model_combined_name = f"{best_model_combined_name} Combined"
    
    # Load traditional ML models
    model_files = [
        ('naive_bayes_model.pkl', 'Naive Bayes'),
        ('logistic_regression_model.pkl', 'Logistic Regression'),
        ('random_forest_model.pkl', 'Random Forest'),
        ('svm_model.pkl', 'Support Vector Machine')
    ]
    
    # Add combined models
    combined_model_files = [
        ('naive_bayes_model_combined.pkl', 'Naive Bayes Combined'),
        ('logistic_regression_model_combined.pkl', 'Logistic Regression Combined'),
        ('random_forest_model_combined.pkl', 'Random Forest Combined'),
        ('support_vector_machine_model_combined.pkl', 'Support Vector Machine Combined')
    ]
    
    # Load all models (original and combined)
    all_model_files = model_files + combined_model_files
    
    for file_name, model_name in all_model_files:
        model_path = os.path.join(MODELS_DIR, file_name)
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    # For combined models, the pickle file contains just the model
                    if "_combined" in file_name:
                        model = pickle.load(f)
                        # Load the combined vectorizer separately
                        vectorizer_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer_combined.pkl')
                        if os.path.exists(vectorizer_path):
                            with open(vectorizer_path, 'rb') as vf:
                                vectorizer = pickle.load(vf)
                        else:
                            st.warning(f"Combined vectorizer not found at {vectorizer_path}")
                            continue
                    else:
                        # Original models have model and vectorizer in the same pickle file
                        model, vectorizer = pickle.load(f)
                    
                models[model_name] = model
                vectorizers[model_name] = vectorizer
            except Exception as e:
                st.error(f"Error loading {model_name}: {e}")
    
    # Only load LSTM and BERT models if TensorFlow is available
    if TENSORFLOW_AVAILABLE:
        # Load LSTM model if available
        lstm_model_path = os.path.join(MODELS_DIR, 'lstm_model.h5')
        tokenizer_path = os.path.join(MODELS_DIR, 'tokenizer.pkl')
        
        if os.path.exists(lstm_model_path) and os.path.exists(tokenizer_path):
            try:
                lstm_model = load_model(lstm_model_path)
                with open(tokenizer_path, 'rb') as f:
                    tokenizer = pickle.load(f)
                models['LSTM'] = lstm_model
                vectorizers['LSTM'] = tokenizer
            except Exception as e:
                st.error(f"Error loading LSTM model: {e}")
        
        # Load BERT model if available
        bert_model_path = os.path.join(MODELS_DIR, 'bert_model.h5')
        
        if os.path.exists(bert_model_path):
            try:
                from transformers import TFBertModel, BertTokenizer
                bert_model = load_model(bert_model_path, custom_objects={'TFBertModel': TFBertModel})
                bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                models['BERT'] = bert_model
                vectorizers['BERT'] = bert_tokenizer
            except Exception as e:
                st.error(f"Error loading BERT model: {e}")
    
    return models, vectorizers, best_model_name, best_model_combined_name

# Make predictions
def predict_fake_news(text, model_name, models, vectorizers):
    '''
    Predict whether a news article is fake or real
    
    Args:
        text (str): Input text
        model_name (str): Name of the model to use
        models (dict): Dictionary of loaded models
        vectorizers (dict): Dictionary of loaded vectorizers
        
    Returns:
        tuple: (prediction, probability, processing_time)
    '''
    start_time = time.time()
    
    # Preprocess text
    preprocessed_text = preprocess_text(text)
    
    # Get model and vectorizer
    model = models.get(model_name)
    vectorizer = vectorizers.get(model_name)
    
    if model is None or vectorizer is None:
        return None, None, None
    
    # Make prediction based on model type
    if model_name == 'LSTM':
        if not TENSORFLOW_AVAILABLE:
            st.error("TensorFlow is not installed. Cannot use LSTM model.")
            return None, None, None
            
        # Tokenize and pad sequence
        sequence = vectorizer.texts_to_sequences([preprocessed_text])
        padded_sequence = pad_sequences(sequence, maxlen=200)
        
        # Predict
        prediction_prob = model.predict(padded_sequence)[0][0]
        prediction = 1 if prediction_prob >= 0.5 else 0
    
    elif model_name == 'BERT':
        if not TENSORFLOW_AVAILABLE:
            st.error("TensorFlow is not installed. Cannot use BERT model.")
            return None, None, None
            
        # Tokenize for BERT
        encodings = vectorizer(
            preprocessed_text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='tf'
        )
        
        # Predict
        prediction_prob = model.predict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        })[0][0]
        prediction = 1 if prediction_prob >= 0.5 else 0
    
    else:  # Traditional ML models
        # Transform text
        features = vectorizer.transform([preprocessed_text])
        
        # Predict
        prediction = model.predict(features)[0]
        
        # Get probability if available
        try:
            prediction_prob = model.predict_proba(features)[0][1]
        except (AttributeError, NotImplementedError):
            # If predict_proba is not available, use decision function if available
            try:
                prediction_prob = model.decision_function(features)[0]
                # Normalize to [0, 1] range
                prediction_prob = (prediction_prob - model.decision_function(vectorizer.transform(['']))[0]) / 2 + 0.5
            except (AttributeError, NotImplementedError):
                # If neither is available, use prediction as probability
                prediction_prob = float(prediction)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    return prediction, prediction_prob, processing_time

# Load feature importance if available
@st.cache_data
def load_feature_importance():
    feature_importance_path = os.path.join(PROCESSED_DATA_DIR, 'feature_importance.csv')
    if os.path.exists(feature_importance_path):
        return pd.read_csv(feature_importance_path)
    return None

# Load model comparison if available
@st.cache_data
def load_model_comparison(combined=False):
    if combined:
        model_comparison_path = os.path.join(PROCESSED_DATA_DIR, 'model_comparison_combined.csv')
    else:
        model_comparison_path = os.path.join(PROCESSED_DATA_DIR, 'model_comparison.csv')
    
    # Check if the file exists
    if os.path.exists(model_comparison_path):
        try:
            return pd.read_csv(model_comparison_path)
        except Exception as e:
            st.error(f"Error loading model comparison data: {e}")
            return None
    else:
        # Log the path that was checked
        st.warning(f"Model comparison file not found at: {model_comparison_path}")
        # Try to get the absolute path for debugging
        abs_path = os.path.abspath(model_comparison_path)
        st.info(f"Absolute path checked: {abs_path}")
        return None

# Main function
def main():
    # Load models
    models, vectorizers, best_model_name, best_model_combined_name = load_models()
    
    # Set default model (prioritize combined model if available)
    default_model = best_model_combined_name if best_model_combined_name else best_model_name
    if not default_model and models:
        default_model = next(iter(models.keys()))
    
    # Sidebar
    st.sidebar.title("Fake News Detector")
    st.sidebar.image("https://img.icons8.com/color/96/000000/detective.png", width=100)
    
    # Show which dataset the models are trained on
    if any("Combined" in model_name for model_name in models.keys()):
        st.sidebar.success("✅ Combined dataset models available")
    
    # Navigation
    page = st.sidebar.radio("Navigation", ["Home", "Detect Fake News", "Model Performance", "About"])
    
    if page == "Home":
        st.title("Fake News Detection System for India")
        st.markdown("""
        <style>
        .big-font {
            font-size:20px !important;
        }
        </style>
        <div class="big-font">
        Welcome to the Fake News Detection System! This application uses Natural Language Processing (NLP) 
        and Machine Learning techniques to analyze news articles and classify them as either real or fake.
        </div>
        """, unsafe_allow_html=True)
        
        st.image("https://www.shutterstock.com/image-vector/fake-news-rubber-stamp-red-600nw-1028723563.jpg", width=600)
        
        st.markdown("### Key Features")
        st.markdown("""
        - **Multiple ML Models**: Choose from various machine learning models for fake news detection
        - **Real-time Analysis**: Get instant predictions on news articles
        - **Explanation**: Understand what makes news likely to be fake
        - **Performance Metrics**: View detailed model performance statistics
        """)
        
        st.markdown("### How to Use")
        st.markdown("""
        1. Navigate to the **Detect Fake News** page
        2. Enter or paste a news article
        3. Select a model for analysis
        4. Click 'Analyze' to get results
        """)
        
        st.markdown("### Why This Matters")
        st.markdown("""
        <div class="big-font">
        Fake news can have serious consequences, from influencing public opinion to affecting democratic processes. 
        This tool aims to help users critically evaluate news content and identify potentially misleading information.
        </div>
        """, unsafe_allow_html=True)
    
    elif page == "Detect Fake News":
        st.title("Detect Fake News")
        
        # Input area for news article
        news_text = st.text_area("Enter news article text", height=200)
        
        # Model selection
        available_models = list(models.keys())
        if available_models:
            selected_model = st.selectbox(
                "Select Model", 
                available_models,
                index=available_models.index(default_model) if default_model in available_models else 0
            )
        else:
            st.error("No models available. Please train models first.")
            return
        
        # Analysis button
        if st.button("Analyze"):
            if not news_text.strip():
                st.warning("Please enter a news article to analyze.")
            else:
                with st.spinner("Analyzing..."):
                    # Make prediction
                    prediction, probability, processing_time = predict_fake_news(
                        news_text, selected_model, models, vectorizers
                    )
                    
                    if prediction is not None:
                        # Display results
                        st.markdown("### Analysis Results")
                        
                        # Create columns for results
                        col1, col2, col3 = st.columns(3)
                        
                        # Display prediction
                        with col1:
                            if prediction == 1:
                                st.error("Prediction: FAKE NEWS")
                            else:
                                st.success("Prediction: REAL NEWS")
                        
                        # Display probability
                        with col2:
                            st.info(f"Confidence: {probability:.2%}")
                        
                        # Display processing time
                        with col3:
                            st.info(f"Processing Time: {processing_time:.4f} seconds")
                        
                        # Display gauge chart for probability
                        fig, ax = plt.subplots(figsize=(10, 2))
                        
                        # Create a horizontal gauge
                        ax.barh([0], [1], color='lightgray', height=0.5)
                        ax.barh([0], [probability], color='red' if prediction == 1 else 'green', height=0.5)
                        
                        # Add labels
                        ax.text(0, 0, "Real", ha='center', va='center', fontsize=12, fontweight='bold')
                        ax.text(1, 0, "Fake", ha='center', va='center', fontsize=12, fontweight='bold')
                        ax.text(probability, 0, f"{probability:.2%}", ha='center', va='center', 
                                fontsize=12, fontweight='bold', color='white')
                        
                        # Remove axes
                        ax.set_xlim(0, 1)
                        ax.set_ylim(-0.5, 0.5)
                        ax.axis('off')
                        
                        st.pyplot(fig)
                        
                        # Display text analysis
                        st.markdown("### Text Analysis")
                        
                        # Preprocess text for display
                        preprocessed_text = preprocess_text(news_text)
                        
                        # Display word count and character count
                        col1, col2 = st.columns(2)
                        with col1:
                            st.info(f"Word Count: {len(preprocessed_text.split())}")
                        with col2:
                            st.info(f"Character Count: {len(preprocessed_text)}")
                        
                        # Display top words
                        words = preprocessed_text.split()
                        word_freq = pd.Series(words).value_counts().head(10)
                        
                        st.markdown("#### Top 10 Words")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        word_freq.plot(kind='bar', ax=ax)
                        plt.title('Top 10 Words')
                        plt.xlabel('Word')
                        plt.ylabel('Frequency')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Display feature importance if available for Logistic Regression
                        if selected_model == "Logistic Regression":
                            feature_importance = load_feature_importance()
                            if feature_importance is not None:
                                st.markdown("#### Feature Importance")
                                st.markdown("Words that influence the prediction:")
                                
                                # Find words in the article that are in the feature importance list
                                article_words = set(preprocessed_text.split())
                                important_features = feature_importance[
                                    feature_importance['feature'].isin(article_words)
                                ].sort_values('abs_importance', ascending=False).head(10)
                                
                                if not important_features.empty:
                                    fig, ax = plt.subplots(figsize=(10, 5))
                                    colors = ['red' if x > 0 else 'green' for x in important_features['importance']]
                                    sns.barplot(x='importance', y='feature', data=important_features, palette=colors)
                                    plt.title('Important Words in This Article')
                                    plt.xlabel('Importance (Positive = Fake, Negative = Real)')
                                    plt.ylabel('Word')
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                else:
                                    st.info("No significant words found in the article for feature importance analysis.")
                    else:
                        st.error("Error making prediction. Please try again.")
    
    elif page == "Model Performance":
        st.title("Model Performance")
        
        # Create tabs for original and combined models
        tabs = st.tabs(["Original Models", "Combined Models"])
        
        # Tab for original models
        with tabs[0]:
            # Load original model comparison data
            model_comparison = load_model_comparison(combined=False)
            
            if model_comparison is not None:
                st.markdown("### Original Model Comparison")
                st.info("These models were trained on the original dataset.")
                
                # Display metrics table
                st.dataframe(model_comparison[['model_name', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']])
                
                # Plot metrics comparison
                st.markdown("#### Metrics Comparison")
                
                # Melt the dataframe for easier plotting
                plot_df = pd.melt(
                    model_comparison, 
                    id_vars=['model_name'], 
                    value_vars=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                    var_name='metric',
                    value_name='score'
                )
                
                # Create plot
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(x='metric', y='score', hue='model_name', data=plot_df)
                plt.title('Original Model Performance Comparison')
                plt.xlabel('Metric')
                plt.ylabel('Score')
                plt.ylim(0, 1)
                plt.legend(title='Model')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display best model
                best_model_idx = model_comparison['f1'].idxmax()
                best_model = model_comparison.loc[best_model_idx, 'model_name']
                
                st.markdown(f"### Best Original Model: {best_model}")
                st.markdown("Performance metrics:")
                
                # Create metrics display
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Accuracy", f"{model_comparison.loc[best_model_idx, 'accuracy']:.4f}")
                
                with col2:
                    st.metric("Precision", f"{model_comparison.loc[best_model_idx, 'precision']:.4f}")
                
                with col3:
                    st.metric("Recall", f"{model_comparison.loc[best_model_idx, 'recall']:.4f}")
                
                with col4:
                    st.metric("F1 Score", f"{model_comparison.loc[best_model_idx, 'f1']:.4f}")
                
                with col5:
                    st.metric("ROC AUC", f"{model_comparison.loc[best_model_idx, 'roc_auc']:.4f}")
            else:
                st.info("Original model comparison data not available.")
        
        # Tab for combined models
        with tabs[1]:
            # Load combined model comparison data
            model_comparison_combined = load_model_comparison(combined=True)
            
            if model_comparison_combined is not None:
                st.markdown("### Combined Model Comparison")
                st.success("These models were trained on the combined dataset (fake.csv + true.csv).")
                
                # Display metrics table
                st.dataframe(model_comparison_combined[['model_name', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']])
                
                # Plot metrics comparison
                st.markdown("#### Metrics Comparison")
                
                # Melt the dataframe for easier plotting
                plot_df = pd.melt(
                    model_comparison_combined, 
                    id_vars=['model_name'], 
                    value_vars=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                    var_name='metric',
                    value_name='score'
                )
                
                # Create plot
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(x='metric', y='score', hue='model_name', data=plot_df)
                plt.title('Combined Model Performance Comparison')
                plt.xlabel('Metric')
                plt.ylabel('Score')
                plt.ylim(0, 1)
                plt.legend(title='Model')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display best model
                best_model_idx = model_comparison_combined['f1'].idxmax()
                best_model = model_comparison_combined.loc[best_model_idx, 'model_name']
                
                st.markdown(f"### Best Combined Model: {best_model}")
                st.markdown("Performance metrics:")
                
                # Create metrics display
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Accuracy", f"{model_comparison_combined.loc[best_model_idx, 'accuracy']:.4f}")
                
                with col2:
                    st.metric("Precision", f"{model_comparison_combined.loc[best_model_idx, 'precision']:.4f}")
                
                with col3:
                    st.metric("Recall", f"{model_comparison_combined.loc[best_model_idx, 'recall']:.4f}")
                
                with col4:
                    st.metric("F1 Score", f"{model_comparison_combined.loc[best_model_idx, 'f1']:.4f}")
                
                with col5:
                    st.metric("ROC AUC", f"{model_comparison_combined.loc[best_model_idx, 'roc_auc']:.4f}")
            else:
                st.info("Combined model comparison data not available.")
            
            # Feature importance for Logistic Regression
            feature_importance = load_feature_importance()
            
            if feature_importance is not None:
                st.markdown("### Feature Importance Analysis")
                st.markdown("Words that influence the prediction of fake news:")
                
                # Plot top positive and negative features
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
                
                # Top positive features (indicative of fake news)
                top_positive = feature_importance[feature_importance['importance'] > 0].head(15)
                sns.barplot(x='importance', y='feature', data=top_positive, palette='Reds_r', ax=ax1)
                ax1.set_title('Top Features Indicative of Fake News')
                ax1.set_xlabel('Coefficient')
                ax1.set_ylabel('Feature')
                
                # Top negative features (indicative of real news)
                top_negative = feature_importance[feature_importance['importance'] < 0].head(15)
                sns.barplot(x='importance', y='feature', data=top_negative, palette='Blues_r', ax=ax2)
                ax2.set_title('Top Features Indicative of Real News')
                ax2.set_xlabel('Coefficient')
                ax2.set_ylabel('Feature')
                
                plt.tight_layout()
                st.pyplot(fig)
        if model_comparison is None:
            st.info("Model comparison data not available. Please train models first.")
    
    elif page == "About":
        st.title("About This Project")
        
        st.markdown("""
        ### Project Overview
        
        This Fake News Detection System is designed to help identify fake news articles in the Indian context. 
        It uses Natural Language Processing (NLP) and Machine Learning techniques to analyze news content and 
        classify it as either real or fake based on its linguistic patterns and content features.
        
        ### Datasets
        
        The system offers two sets of models:
        
        1. **Original Models**: Trained on the original fake news dataset.
        2. **Combined Models**: Trained on a combined dataset from `fake.csv` and `true.csv` files, offering improved accuracy and robustness.
        
        ### Methodology
        
        1. **Data Collection**: The system is trained on datasets of labeled news articles from Indian sources.
        

        2. **Text Preprocessing**: News articles are cleaned and preprocessed using techniques such as:
           - Tokenization
           - Stopword removal
           - Lemmatization
        

        3. **Feature Extraction**: The system extracts features from the text using:
           - TF-IDF vectorization
           - Word embeddings
           - Named entity recognition
        

        4. **Model Training**: Multiple machine learning models are trained and evaluated:
           - Traditional ML models (Naive Bayes, Logistic Regression, Random Forest, SVM)
           - Deep learning models (LSTM, BERT)
        

        5. **Evaluation**: Models are evaluated using metrics such as accuracy, precision, recall, F1 score, and ROC AUC.
        
        ### Limitations
        
        - The system's accuracy depends on the quality and diversity of the training data.
        - It may not catch all types of misinformation, especially subtle forms.
        - Context and external knowledge are sometimes necessary for accurate fact-checking.
        - The system should be used as a tool to assist human judgment, not replace it.
        
        ### Future Improvements
        
        - Incorporate fact-checking APIs for verification against known facts
        - Add source credibility analysis
        - Implement image analysis for detecting manipulated media
        - Develop multilingual support for Indian languages
        
        ### Contact
        
        For questions, feedback, or suggestions, please contact:
        
        - Email: example@example.com
        - GitHub: [github.com/username/fake-news-detection](https://github.com/username/fake-news-detection)
        """)

# Run the app
if __name__ == "__main__":
    main()