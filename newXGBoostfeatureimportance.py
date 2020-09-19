# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 00:17:59 2020

@author: user1
"""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, plot_importance
from matplotlib import pyplot
#Preporocessing
dataset = pd.read_csv('TotalGridData.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


model = XGBClassifier()
model.fit(X, y)

sorted_idx = np.argsort(model.feature_importances_)[::-1]

# model.get_score(importance_type='gain')

# for index in sorted_idx:
#     print([X.columns[index], model.feature_importances_[index]]) 


plot_importance(model, max_num_features = 15)
pyplot.show()