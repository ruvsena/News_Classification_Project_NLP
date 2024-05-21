#!/usr/bin/env python
# coding: utf-8

# In[1]:
pip install streamlit
pip install nltk
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk import download as nltk_download
#from nltk.stem import TurkishStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import re

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


# Gerekli NLTK veri setlerini indir
import nltk
nltk.download('stopwords')
nltk.download('punkt')


# In[3]:


stop_words = set(stopwords.words('turkish'))
# Türkçe stop words'leri yükleme


# In[4]:


# Veri setini yükleme
veri = pd.read_csv('data.csv')



# In[5]:


class_distribution = veri['Sınıf'].value_counts()
print(class_distribution)


# In[6]:


data = veri.sample(n=100000, random_state=32)


# In[8]:


class_distribution = data['Sınıf'].value_counts()
print(class_distribution)


# In[9]:


data['Haber Gövdesi'] =data['Haber Gövdesi'].str.replace('[^\w\s]', '',regex=True)# Özel karakterleri kaldırma
data['Haber Gövdesi'] =data['Haber Gövdesi'].apply(lambda x: x.lower())  # Küçük harfe dönüştürme


# In[10]:


from snowballstemmer import TurkishStemmer
# Türkçe stop words'leri yükleme
stop_words = set(stopwords.words('turkish'))
stemmer = TurkishStemmer()

# Ek stop words 
additional_stop_words = {'bir', 've', 'bu', 'de', 'da','ile', 'için', 'kadar', 'çok', 'en', 'olarak','var', 'daha', 'ne', 'her', 'oldu', 'sonra'}
stop_words.update(additional_stop_words)


print("Toplam stop words sayısı:", len(stop_words))
print("Stop words'lerden bazıları:", list(stop_words)[:10]) 



# In[11]:


print(data['Haber Gövdesi'].head())


# In[12]:


def preprocess_text(text):
    tokens = word_tokenize(text)
    # Stemming işlemi
    stemmed_tokens = [stemmer.stemWord(token) for token in tokens]
    # Stop words'leri kaldırma
    filtered_tokens = [token for token in stemmed_tokens if token not in stop_words]
    return ' '.join(filtered_tokens)


# In[13]:


data['Haber Gövdesi'] = data['Haber Gövdesi'].apply(preprocess_text)



# In[15]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectors = {}

# Her sınıf için TF-IDF vektörlerini hesapla
for category in data['Sınıf'].unique():
    documents_in_category = data[data['Sınıf'] == category]['Haber Gövdesi']
    
    # TF-IDF vektörlerini hesaplayın
    vectorizer = TfidfVectorizer(max_features=10000)  
    tfidf_matrix = vectorizer.fit_transform(documents_in_category)
    
    tfidf_vectors[category] = (vectorizer, tfidf_matrix)

# En önemli terimleri yazdırma
def get_top_tfidf_terms(tfidf_matrix, vectorizer, top_n=20):
    mean_tfidf = tfidf_matrix.mean(axis=0).A1
    top_indices = np.argsort(mean_tfidf)[::-1][:top_n]
    top_terms = [(vectorizer.get_feature_names_out()[i], mean_tfidf[i]) for i in top_indices]
    return top_terms

# Her sınıf için en önemli terimleri yazdırma
for category, (vectorizer, tfidf_matrix) in tfidf_vectors.items():
    print(f"{category} sınıfının en önemli terimleri:")
    top_terms = get_top_tfidf_terms(tfidf_matrix, vectorizer)
    for term, score in top_terms:
        print(f"Terim: {term}, TF-IDF Skoru: {score}")
    print("\n")


# In[17]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Her sınıf için en önemli terimleri görselleştirme
for category, (vectorizer, tfidf_matrix) in tfidf_vectors.items():
    top_terms = get_top_tfidf_terms(tfidf_matrix, vectorizer)
    terms, scores = zip(*top_terms)
    
    plt.figure(figsize=(10, 6))
    plt.barh(terms, scores, color='skyblue')
    plt.xlabel('TF-IDF Skoru')
    plt.title(f"{category} sınıfının en önemli terimleri")
    plt.gca().invert_yaxis()  
    plt.grid(True)
    plt.show()


# In[ ]:





# In[23]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

X = data['Haber Gövdesi']
y = data['Sınıf']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# TF-IDF vektörleştirme
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

nbModel = MultinomialNB()
nbModel.fit(X_train_tfidf, y_train)

y_test_predicted = nbModel.predict(X_test_tfidf)

# Performans metriğini yazdırma
from sklearn.metrics import classification_report
print(classification_report(y_test, y_test_predicted))


# In[24]:


from imblearn.over_sampling import SMOTE
from collections import Counter

# SMOTE kullanarak veriyi dengeleme
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_tfidf, y_train)

# Yeni dağılımı yazdırma
print("Yeni sınıf dağılımı:", Counter(y_resampled))

# Naive Bayes modelini yeniden eğitme
nbModel = MultinomialNB()
nbModel.fit(X_resampled, y_resampled)

# Test verileri üzerinde modelin performansını değerlendirme
y_test_predicted = nbModel.predict(X_test_tfidf)

# Performans metriğini yazdırma
print(classification_report(y_test, y_test_predicted, zero_division=1))


# In[26]:


accuracy = accuracy_score(y_test, y_test_predicted)
print(f"Accuracy: {accuracy}")


# In[ ]:


from sklearn.svm import SVC

X_train_tfidf = vectorizer.transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# SVM modelini oluşturma ve eğitme
svm_model = SVC(C=1.0, kernel='linear', class_weight='balanced', random_state=42)
svm_model.fit(X_train_tfidf, y_train)

y_test_predicted = svm_model.predict(X_test_tfidf)

# Performans metriğini yazdırma
print(classification_report(y_test, y_test_predicted, zero_division=1))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




