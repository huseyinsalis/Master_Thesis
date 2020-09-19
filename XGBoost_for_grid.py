# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 16:02:11 2020

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

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

classifier = XGBClassifier(objective="multi:softprob", nthread=4, random_state=42)
parameters = {
    'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05]
}

grid_search = GridSearchCV(
    estimator=classifier,
    param_grid=parameters,
    n_jobs = -1,
    cv = 10,
    verbose=True
)
grid_search.fit(X, y)
print(grid_search.best_estimator_)

classifier = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              learning_rate=0.1, max_delta_step=0, max_depth=3,
              min_child_weight=1, missing=None, n_estimators=180, n_jobs=1,
              nthread=4, objective='multi:softprob', random_state=42,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)
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