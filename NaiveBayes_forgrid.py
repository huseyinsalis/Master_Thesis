# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 02:36:25 2020

@author: user1
"""
import pandas as pd

#Preporocessing
dataset = pd.read_csv('TotalGridData.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Splitting Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

naive = make_pipeline(MinMaxScaler(), GaussianNB())
scores = cross_val_score(naive, X_train, y_train, cv = 10)

naive.fit(X_train, y_train)

y_pred = naive.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Naive Bayes :")
print(cm)

from sklearn.metrics import accuracy_score
accscore = accuracy_score(y_test, y_pred)
print(accscore)

