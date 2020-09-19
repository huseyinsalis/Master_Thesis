# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 08:55:24 2020

@author: user1
"""

# permutation feature importance with knn for classification

from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from matplotlib import pyplot
# define dataset
import pandas as pd

#Preporocessing
dataset = pd.read_csv('GridData.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# define the model
model = KNeighborsClassifier()
# fit the model
model.fit(X, y)
# perform permutation importance
results = permutation_importance(model, X, y, scoring='accuracy')
# get importance
importance = results.importances_mean
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()