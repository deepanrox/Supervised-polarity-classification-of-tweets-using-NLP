# SUPERVISED POLARITY CLASSIFICATION OF TWEETS USING NLP

A comprehensive machine learning project that performs supervised polarity classification (sentiment analysis) on Twitter data using Natural Language Processing (NLP) techniques.

## 📋 Project Overview

This project implements a supervised learning approach for sentiment analysis using:
- **Dataset**: Sentiment140 dataset (1.6M tweets)
- **Algorithm**: Logistic Regression with TF-IDF vectorization
- **NLP Techniques**: Text preprocessing, stemming, stopword removal
- **Supervised Learning**: Binary classification (Positive/Negative)
- **Accuracy**: ~77.8% on test data

## 🎯 Research Objective

The primary goal is to develop a supervised learning model that can accurately classify tweets into positive and negative sentiment categories using advanced NLP techniques and machine learning algorithms.

## 🚀 Key Features

- **Supervised Learning Approach**: Uses labeled training data for model training
- **Advanced NLP Pipeline**: Text preprocessing, tokenization, stemming, stopword removal
- **TF-IDF Vectorization**: Converts text to numerical features
- **Logistic Regression**: Binary classification model
- **Comprehensive Evaluation**: Accuracy, classification report, confusion matrix
- **Interactive Prediction**: Real-time sentiment analysis
- **Model Persistence**: Save and load trained models


## 📊 Dataset

### Sentiment140 Dataset
- **Source**: Kaggle (https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Size**: 1.6 million tweets
- **Labels**: Binary sentiment (0 = Negative, 1 = Positive)
- **Format**: CSV with columns: target, id, date, flag, user, text

### Dataset Statistics
- **Total tweets**: 1,600,000
- **Positive tweets**: 800,000 (50%)
- **Negative tweets**: 800,000 (50%)
- **Balanced dataset**: Equal distribution of classes

## 🔬 Methodology

### 1. Data Preprocessing
- **Text Cleaning**: Remove special characters, convert to lowercase
- **Tokenization**: Split text into words
- **Stopword Removal**: Remove common English stopwords
- **Stemming**: Reduce words to root form using Porter Stemmer

### 2. Feature Engineering
- **TF-IDF Vectorization**: Convert text to numerical features
- **Feature Selection**: Limit to top 5000 features for efficiency
- **Dimensionality**: High-dimensional sparse matrix representation

### 3. Model Training
- **Algorithm**: Logistic Regression
- **Split Ratio**: 80% training, 20% testing
- **Stratification**: Maintain class balance in splits
- **Random State**: 42 for reproducibility

### 4. Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: Positive predictive value
- **Recall**: Sensitivity/true positive rate
- **F1-Score**: Harmonic mean of precision and recall


## 📈 Results

### Model Performance
- **Training Accuracy**: 77.8%
- **Test Accuracy**: 77.8%
- **Precision (Positive)**: 0.78
- **Recall (Positive)**: 0.78
- **F1-Score (Positive)**: 0.78

### Classification Report
```
              precision    recall  f1-score   support

    Negative       0.78      0.78      0.78    160000
    Positive       0.78      0.78      0.78    160000

    accuracy                           0.78    320000
   macro avg       0.78      0.78      0.78    320000
weighted avg       0.78      0.78      0.78    320000
```

## 🔍 NLP Pipeline

1. **Text Preprocessing**
   - Remove special characters and numbers
   - Convert to lowercase
   - Tokenize into words

2. **Feature Extraction**
   - Remove stopwords
   - Apply Porter stemming
   - TF-IDF vectorization

3. **Model Training**
   - Logistic Regression with L2 regularization
   - Cross-validation for hyperparameter tuning
   - Model evaluation and selection

## 🎯 Applications

This supervised polarity classification system can be applied to:

- **Social Media Monitoring**: Analyze public sentiment on social platforms
- **Brand Analysis**: Monitor brand perception and customer satisfaction
- **Market Research**: Understand consumer opinions and trends
- **Customer Service**: Automate sentiment analysis of customer feedback
- **Political Analysis**: Monitor public opinion on political topics

## 🚀 Future Enhancements

1. **Advanced Models**: Implement BERT, RoBERTa, or other transformer models
2. **Multi-class Classification**: Extend to neutral sentiment classification
3. **Real-time Processing**: Develop streaming sentiment analysis
4. **Cross-lingual Support**: Extend to multiple languages
5. **Deep Learning**: Implement neural networks for better performance

## 📝 Dependencies

- numpy>=1.21.0
- pandas>=1.3.0
- nltk>=3.6.0
- scikit-learn>=1.0.0
- kaggle>=1.5.0
- matplotlib>=3.5.0
- seaborn>=0.11.0


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Sentiment140 Dataset**: Go et al. (2009) for the comprehensive dataset
- **NLTK Library**: Natural Language Toolkit contributors
- **Scikit-learn**: Machine learning library developers
- **Kaggle**: For hosting the dataset and providing the platform

## 📚 References

1. Go, A., Bhayani, R., & Huang, L. (2009). Twitter sentiment classification using distant supervision.
2. Porter, M. F. (1980). An algorithm for suffix stripping.
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python.

