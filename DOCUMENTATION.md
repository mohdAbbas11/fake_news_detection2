# Fake News Detection System - Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Installation](#installation)
4. [Project Structure](#project-structure)
5. [Data Collection](#data-collection)
6. [Text Preprocessing](#text-preprocessing)
7. [Model Training](#model-training)
8. [Streamlit Application](#streamlit-application)
9. [Performance Metrics](#performance-metrics)
10. [Troubleshooting](#troubleshooting)
11. [Future Improvements](#future-improvements)

## Project Overview

The Fake News Detection System is designed to identify and classify fake news articles in the Indian context using Natural Language Processing (NLP) and Machine Learning techniques. The system analyzes the linguistic patterns, content features, and contextual information of news articles to determine their authenticity.

The project includes:
- Data collection from multiple sources
- Text preprocessing and feature extraction
- Training and evaluation of multiple machine learning models
- A user-friendly Streamlit web application for real-time fake news detection

## System Architecture

The system follows a modular architecture with the following components:

1. **Data Collection Module**: Gathers news articles from various sources and creates a labeled dataset.

2. **Preprocessing Module**: Cleans and transforms raw text data into a format suitable for machine learning.

3. **Model Training Module**: Trains multiple machine learning models on the preprocessed data and selects the best-performing model.

4. **Inference Module**: Uses the trained model to make predictions on new, unseen news articles.

5. **Web Application**: Provides a user interface for interacting with the system.

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Virtual environment (recommended)

### Setup Instructions

1. **Clone the repository or download the project files**

2. **Create and activate a virtual environment (recommended)**:

   **Windows**:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```

   **Linux/Mac**:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

4. **Download required NLTK data**:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

5. **Download spaCy model**:
   ```
   python -m spacy download en_core_web_sm
   ```

### Running the Project

**Using the provided scripts**:

- **Windows**: Double-click on `run_project.bat` or run it from the command line.
- **Linux/Mac**: Make the script executable with `chmod +x run_project.sh` and then run `./run_project.sh`.

**Manually**:

1. **Download datasets** (optional if you already have data):
   ```
   python data/download_datasets.py
   ```

2. **Train models**:
   ```
   python models/train_simple_model.py
   ```

3. **Run the Streamlit app**:
   ```
   cd app
   streamlit run app.py
   ```

**Command-line options**:

The `run_project.py` script accepts several command-line arguments:

- `--skip-install`: Skip installing requirements
- `--skip-nltk`: Skip downloading NLTK data
- `--skip-spacy`: Skip downloading spaCy model
- `--skip-datasets`: Skip downloading datasets
- `--skip-training`: Skip training models
- `--app-only`: Only run the Streamlit app

Example: `python run_project.py --skip-training --app-only`

## Project Structure

```
fake_news_detection/
├── app/                    # Streamlit application
│   └── app.py              # Main application file
├── data/                   # Data directory
│   ├── raw/                # Raw datasets
│   ├── processed/          # Processed datasets
│   └── download_datasets.py # Script to download datasets
├── models/                 # Trained models
│   └── train_simple_model.py # Script to train models
├── notebooks/              # Jupyter notebooks
│   ├── 01_data_collection.ipynb
│   ├── 02_text_preprocessing.ipynb
│   └── 03_model_training.ipynb
├── .venv/                  # Virtual environment (created during setup)
├── requirements.txt        # Project dependencies
├── README.md              # Project overview
├── DOCUMENTATION.md       # Detailed documentation
├── run_project.py         # Main script to run the project
├── run_project.bat        # Windows batch script
└── run_project.sh         # Linux/Mac shell script
```

## Data Collection

The system uses multiple datasets to ensure robust training:

1. **Indian Fake News Dataset**: A dataset specifically focused on Indian news articles, containing both fake and real news.

2. **Kaggle's Fake and Real News Dataset**: A large dataset with labeled fake and real news articles from various sources.

3. **FakeNewsNet**: A comprehensive repository of fake news articles with social context information.

The data collection process is handled by the `download_datasets.py` script, which:
- Downloads datasets from their respective sources
- Extracts and organizes the data in the `data/raw/` directory
- Provides functions for downloading from URLs, extracting zip files, and downloading from Kaggle

For detailed exploration of the datasets, refer to the `01_data_collection.ipynb` notebook.

## Text Preprocessing

Text preprocessing is a crucial step in preparing the raw text data for machine learning models. The preprocessing pipeline includes:

1. **Text Cleaning**:
   - Converting text to lowercase
   - Removing URLs and HTML tags
   - Removing punctuation and special characters
   - Removing extra whitespace

2. **Tokenization**: Breaking text into individual words or tokens.

3. **Stopword Removal**: Removing common words that don't carry significant meaning (e.g., "the", "and", "is").

4. **Lemmatization**: Reducing words to their base or dictionary form.

The preprocessing functions are implemented in both the Streamlit app (`app.py`) and the model training script (`train_simple_model.py`) to ensure consistency between training and inference.

For a detailed exploration of text preprocessing techniques, refer to the `02_text_preprocessing.ipynb` notebook.

## Model Training

The system trains multiple machine learning models to classify news articles as fake or real:

1. **Traditional Machine Learning Models**:
   - Naive Bayes
   - Logistic Regression
   - Random Forest
   - Support Vector Machine (SVM)

2. **Deep Learning Models** (implemented in the notebooks):
   - LSTM (Long Short-Term Memory)
   - BERT (Bidirectional Encoder Representations from Transformers)

The model training process includes:

1. **Feature Extraction**: Converting preprocessed text into numerical features using TF-IDF vectorization.

2. **Model Training**: Training each model on the training dataset.

3. **Evaluation**: Evaluating models using metrics such as accuracy, precision, recall, F1 score, and ROC AUC.

4. **Model Selection**: Selecting the best-performing model based on F1 score.

5. **Feature Importance Analysis**: For interpretable models like Logistic Regression, analyzing which features (words) are most indicative of fake or real news.

The trained models are saved in the `models/` directory for later use in the Streamlit application.

For a detailed exploration of model training and evaluation, refer to the `03_model_training.ipynb` notebook.

## Streamlit Application

The Streamlit application provides a user-friendly interface for interacting with the fake news detection system. The application includes:

1. **Home Page**: An introduction to the system and its features.

2. **Detect Fake News Page**: Where users can input news articles and get predictions.
   - Text input area for news articles
   - Model selection dropdown
   - Analysis results with prediction, confidence score, and processing time
   - Text analysis with word frequency and feature importance visualization

3. **Model Performance Page**: Displays performance metrics for all trained models.
   - Comparison table with accuracy, precision, recall, F1 score, and ROC AUC
   - Bar chart comparing metrics across models
   - Feature importance analysis for interpretable models

4. **About Page**: Information about the project, methodology, limitations, and future improvements.

The application is designed to be intuitive and informative, providing not just predictions but also explanations and insights into the decision-making process.

## Performance Metrics

The system evaluates models using several performance metrics:

1. **Accuracy**: The proportion of correct predictions among the total number of predictions.

2. **Precision**: The proportion of true positive predictions among all positive predictions (measures false positive rate).

3. **Recall**: The proportion of true positive predictions among all actual positives (measures false negative rate).

4. **F1 Score**: The harmonic mean of precision and recall, providing a balance between the two.

5. **ROC AUC**: The area under the Receiver Operating Characteristic curve, measuring the model's ability to distinguish between classes.

These metrics are calculated during model training and displayed in the Streamlit application for transparency and comparison.

## Troubleshooting

### Common Issues and Solutions

1. **Installation Issues**:
   - **Problem**: Error installing requirements.
   - **Solution**: Try installing packages individually or check for conflicts in your Python environment.

2. **Dataset Download Issues**:
   - **Problem**: Error downloading datasets.
   - **Solution**: Check your internet connection or download datasets manually from the provided URLs.

3. **Kaggle API Issues**:
   - **Problem**: Error using Kaggle API.
   - **Solution**: Ensure you have set up Kaggle API credentials correctly. See [Kaggle API documentation](https://github.com/Kaggle/kaggle-api).

4. **Model Training Issues**:
   - **Problem**: Out of memory error during model training.
   - **Solution**: Reduce the dataset size or use a machine with more memory.

5. **Streamlit App Issues**:
   - **Problem**: Streamlit app crashes or doesn't start.
   - **Solution**: Check that all dependencies are installed and that the models have been trained successfully.

### Logging

The system includes print statements for logging progress and errors. For more detailed logging, consider adding a logging configuration to the scripts.

## Future Improvements

Potential enhancements for the system include:

1. **Multilingual Support**: Extending the system to support Indian languages beyond English.

2. **Advanced Deep Learning Models**: Implementing more sophisticated models like transformers with attention mechanisms.

3. **Explainable AI**: Enhancing the explanation capabilities to provide more insights into why an article is classified as fake or real.

4. **Real-time Monitoring**: Adding capabilities to monitor and analyze news sources in real-time.

5. **Fact-checking Integration**: Integrating with fact-checking APIs or databases to verify claims in news articles.

6. **Source Credibility Analysis**: Incorporating information about the credibility of news sources.

7. **User Feedback Loop**: Allowing users to provide feedback on predictions to improve the system over time.

8. **Mobile Application**: Developing a mobile app version for wider accessibility.

9. **API Endpoint**: Creating an API endpoint for integration with other applications.

10. **Deployment**: Deploying the system to a cloud platform for public access.

---

This documentation provides a comprehensive overview of the Fake News Detection System. For specific implementation details, refer to the code and comments in the respective files.