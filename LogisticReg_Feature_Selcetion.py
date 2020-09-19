# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 08:42:20 2020

@author: user1
"""

# logistic regression for feature importance

from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
import pandas as pd

#Preporocessing
dataset = pd.read_csv('GridData.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

model = LogisticRegression()
# fit the model
model.fit(X, y)
# get importance
importance = model.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()