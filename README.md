# News Classification Project

This repository contains a news classification project developed as a class. The primary goal of this project is to classify news articles into their respective categories (such as politics, sports, economy, etc.) based on their headlines and content. The data for this project was scraped from one of Turkey's leading news websites, resulting in a comprehensive dataset of 1.4 million news articles..

### Project Overview

The project involves several key steps:

**Data Collection**: A total of 1.4 million news headlines and articles were scraped from Turkey's top news website. This extensive dataset serves as the foundation for training our classification model.

**Data Preprocessing**: The collected news articles were cleaned and preprocessed. This involved removing special characters, converting text to lowercase, removing Turkish stop words, and applying stemming to reduce words to their base form.

**Feature Extraction**: We used the TF-IDF (Term Frequency-Inverse Document Frequency) technique to convert the text data into numerical features that can be used by machine learning algorithms. This method helps in identifying the most important words in each news category.

**Model Training**: Two different machine learning models were trained:

**Naive Bayes Classifier**: This model was initially trained on the dataset to classify the news articles.
**Support Vector Machine (SVM)**: An SVM model was also trained to improve the classification accuracy.
**Handling Class Imbalance**: To address the issue of class imbalance in the dataset, we used the SMOTE (Synthetic Minority Over-sampling Technique) method. This technique oversamples the minority classes to balance the dataset before training the model.

**Evaluation**: The models were evaluated based on various performance metrics such as accuracy, precision, recall, and F1-score. The results were analyzed to determine the effectiveness of the models in classifying news articles.
