# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 08:52:13 2020

@author: user1
"""

# xgboost for feature importance on a classification problem
from xgboost import XGBClassifier
from matplotlib import pyplot
# define dataset
import pandas as pd
from xgboost import plot_importance

import numpy as np

#Preporocessing
dataset = pd.read_csv('TotalGridData.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# define the model
model = XGBClassifier()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
print(importance)
sort_index = np.argsort(importance)
print(sort_index)
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
# plot feature importance
plot_importance(model)
params = {'legend.fontsize': 8,
          'legend.handlelength': 2}
pyplot.rcParams.update(params)
pyplot.rc('ytick', labelsize=8) 
pyplot.rc('xtick', labelsize=8) 
pyplot.show()