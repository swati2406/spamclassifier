#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 07:49:11 2022

@author: swati

Task- Spam Classifier

Process-
1. read the file
2. Data preprocessing
3. Train model
4. Test model
5. Pickle model file for deployment
"""

#reading file
import pandas as pd
messages = pd.read_csv('/home/swati/nltk_data/smsspamcollection/SMSSpamCollection',sep='\t',names=['label','messages'])

#data preprocessing
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = []
lemmatizer = WordNetLemmatizer()
for i in range(len(messages)):
    lines = re.sub('[^A-Za-z]',' ',messages.iloc[i,1])
    lines = lines.lower()
    words = lines.split(' ')
    words =[lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    words = ' '.join(words)
    corpus.append(words)
    
tfidf= TfidfVectorizer(max_features=5000)
X= tfidf.fit_transform(corpus).toarray()    
# pickling the text processing
import pickle

y=pd.get_dummies(messages['label'],drop_first=True)

#Training and testing model
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=42)

from sklearn.naive_bayes import MultinomialNB
NB_model = MultinomialNB().fit(X_train, y_train)
y_prediction = NB_model.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test, y_prediction))
print(confusion_matrix(y_test, y_prediction))

#pickling model file for deployment
pickle.dump(NB_model,open('SpamClassifier.pickle','wb'))
