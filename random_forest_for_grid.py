# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 14:56:13 2020

@author: user1
"""

import pandas as pd

#Preporocessing
dataset = pd.read_csv('TotalGridData.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

#Splitting Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()

from sklearn.model_selection import GridSearchCV

param_grid = { 
    'n_estimators': [10, 100, 200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'criterion' :['gini', 'entropy']
}

CV_rfc = GridSearchCV(estimator=classifier, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)

print(CV_rfc.best_params_)

classifier = RandomForestClassifier(criterion= 'entropy', max_features= 'auto', n_estimators= 500)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Kernel rbf SVM :")
print(cm)

from sklearn.metrics import accuracy_score
accscore = accuracy_score(y_test, y_pred)
print(accscore)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred)) 