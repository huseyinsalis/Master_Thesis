# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 02:22:13 2020

@author: user1
"""

import pandas as pd

from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

#Preporocessing
dataset = pd.read_csv('TotalGridData.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Create an scaler object
sc = StandardScaler()


# Create a logistic regression object with an L2 penalty
decisiontree = DecisionTreeClassifier()

# Create a pipeline of three steps. First, standardize the data.
# Second, tranform the data with PCA.
# Third, train a Decision Tree Classifier on the data.
pipe = Pipeline(steps=[('sc', sc),
                       ('decisiontree', decisiontree)])

pipe.fit(X_train,y_train)
#Predict the test data
y_pred = pipe.predict(X_test)

#Confusion matrix and accuracy for the predictions

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
accscore = accuracy_score(y_test, y_pred)
print(accscore)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred)) 