# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 02:22:13 2020

@author: user1
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

#Preporocessing
dataset = pd.read_csv('TotalGridData.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Splitting Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify =y)

logreg2=LogisticRegression(max_iter=100000)
#scores = cross_val_score(logreg2, X_train, y_train, cv = 10)
#print(np.mean(scores))

logreg2.fit(X_train,y_train)
#Predict the test data
y_pred = logreg2.predict(X_test)

# from sklearn.model_selection import cross_val_score

# knnpipe = make_pipeline(MinMaxScaler(), KNeighborsClassifier(n_neighbors = 1, leaf_size=1, metric = 'minkowski', p = 2))
# scores = cross_val_score(knnpipe, X_train, y_train, cv = 10)
# print(np.mean(scores))

#Confusion matrix and accuracy for the predictions

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
accscore = accuracy_score(y_test, y_pred)
print(accscore)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred)) 

test11_18 = pd.read_csv('grid_11_18_old.csv')
X11_18 = test11_18.iloc[:,:].values
y_probabilities=logreg2.predict_proba(X11_18)
print(y_probabilities)