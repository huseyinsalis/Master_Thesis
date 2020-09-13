# -*- coding: utf-8 -*-
"""
Created on Tue May 26 21:24:23 2020

@author: user1
"""

# ANOVA feature selection for numeric input and categorical output

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Preporocessing
dataset = pd.read_csv('X_impedances.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# define feature selection
fs = SelectKBest(score_func=f_classif, k=2)
# apply feature selection
X_selected = fs.fit_transform(X, y)
print(X_selected.shape)