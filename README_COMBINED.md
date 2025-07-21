# Fake News Detection System

## Overview
This project implements a machine learning system for detecting fake news articles. It uses a combined dataset of fake and real news articles to train multiple classification models and selects the best performing one for predictions.

## Dataset
The system uses two datasets:
- `fake.csv`: Contains 23,481 fake news articles with columns for title, text, subject, and date.
- `true.csv`: Contains 21,417 real news articles with the same column structure.

The combined dataset has 44,898 articles with a balanced distribution between fake and real news.

## Models
The system trains and evaluates four different machine learning models:
1. **Naive Bayes**: A probabilistic classifier based on Bayes' theorem.
2. **Logistic Regression**: A linear model for binary classification.
3. **Random Forest**: An ensemble learning method using multiple decision trees.
4. **Support Vector Machine**: A discriminative classifier using a separating hyperplane.

## Performance
Based on the evaluation metrics, the Random Forest model performed the best with:
- Accuracy: 99.76%
- Precision: 99.77%
- Recall: 99.77%
- F1 Score: 99.77%
- ROC AUC: 99.99%

The other models also performed well:
- Naive Bayes: 92.74% accuracy, 93.10% F1 score
- Logistic Regression: 98.62% accuracy, 98.67% F1 score
- Support Vector Machine: 99.40% accuracy, 99.42% F1 score

## Files
- `train_models_combined.py`: Script to train and evaluate models on the combined dataset.
- `predict_combined.py`: Script to use the trained model for predicting if a news article is fake or real.
- `inspect_both_csv.py`: Script to inspect and analyze the structure of the CSV files.
- `inspect_csv_detailed.py`: Script for detailed inspection of the datasets.

## Model Artifacts
- Trained models are saved in the `models/` directory.
- The best model is identified in `models/best_model_combined.txt`.
- Model comparison metrics are saved in `data/processed/model_comparison_combined.csv`.
- Feature importance for Logistic Regression is saved in `data/processed/feature_importance_combined.csv`.
- Confusion matrices for each model are saved as PNG files in the `data/processed/` directory.

## Usage

### Training Models
To train models on the combined dataset:
```bash
python train_models_combined.py
```

### Making Predictions
To use the trained model for predictions:
```bash
python predict_combined.py
```

You can modify the example articles in `predict_combined.py` to test the model on different news articles.

## Text Preprocessing
The system applies the following preprocessing steps to the text data:
1. Converting to lowercase
2. Removing URLs and HTML tags
3. Removing punctuation
4. Tokenizing the text
5. Removing stopwords
6. Lemmatizing tokens

## Feature Extraction
The system uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert the preprocessed text into numerical features for the machine learning models.

## Future Improvements
- Implement cross-validation for more robust model evaluation.
- Explore deep learning models like LSTM or BERT for potentially better performance.
- Add a web interface for easy interaction with the system.
- Implement explainable AI techniques to understand model decisions.
- Expand the dataset with more recent news articles for better generalization.