# Fake News Detection for India

This project aims to detect fake news in the Indian context using Natural Language Processing (NLP) and Machine Learning techniques. The system analyzes news articles and classifies them as either real or fake based on their content, linguistic patterns, and other features.

## Project Structure

```
fake_news_detection/
├── app/                # Streamlit web application
├── data/               # Dataset storage
├── models/             # Trained models
├── notebooks/          # Jupyter notebooks for analysis and training
├── README.md           # Project documentation
└── requirements.txt    # Dependencies
```

## Features

- Data collection from reliable Indian news sources
- Text preprocessing and feature extraction
- Machine learning model training for fake news classification
- Interactive web application for real-time news verification
- Performance metrics and model evaluation

## Setup and Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Download required NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```
5. Download spaCy model:
   ```
   python -m spacy download en_core_web_sm
   ```

## Usage

### Data Collection and Preprocessing

Run the notebooks in the `notebooks/` directory to collect and preprocess data.

### Model Training

Execute the model training notebook to train the fake news detection model.

### Web Application

Start the Streamlit application:
```
cd app
streamlit run app.py
```

## Dataset

The project uses a combination of datasets including:
- News articles from verified Indian news sources
- Labeled fake news datasets specific to the Indian context
- User-reported suspicious news articles

## Model Performance

The model's performance is evaluated using metrics such as:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## License

MIT