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


#Scale the data KNN algorithm needs scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Linear SVM :")
print(cm)

from sklearn.metrics import accuracy_score
accscore = accuracy_score(y_test, y_pred)
print(accscore)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
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

param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
  
# fitting the model for grid search 
grid.fit(X_train, y_train) 
# print best parameter after tuning 
print(grid.best_params_) 

# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_) 
y_pred = grid.predict(X_test) 

# print classification report 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Best Kernel rbf SVM :")
print(cm)

from sklearn.metrics import accuracy_score
accscore = accuracy_score(y_test, y_pred)
print(accscore)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred)) 

 