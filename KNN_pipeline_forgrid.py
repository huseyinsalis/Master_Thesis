# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 00:43:10 2020

@author: user1
"""

import pandas as pd
import numpy as np

#Preporocessing
dataset = pd.read_csv('TotalGridData.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Splitting Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
 
knnpipe = make_pipeline(MinMaxScaler(), KNeighborsClassifier(n_neighbors = 1, leaf_size=1, metric = 'minkowski', p = 2))
scores = cross_val_score(knnpipe, X_train, y_train, cv = 10)
print(np.mean(scores))