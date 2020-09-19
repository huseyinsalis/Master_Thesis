# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 13:49:25 2020

@author: user1
"""

# ANOVA feature selection for numeric input and categorical output
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
# generate dataset
#Preporocessing
dataset = pd.read_csv('GridData2.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# define feature selection
fs = SelectKBest(score_func=f_classif, k=2)
# apply feature selection
X_selected = fs.fit_transform(X, y)
print(X_selected.shape)