# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 15:06:50 2020

@author: user1
"""
import pandas as pd
from pylab import rcParams
import seaborn as sb
from scipy.stats.stats import kendalltau
# Data Visualisation Settings 

rcParams['figure.figsize'] = 5,4
sb.set_style('whitegrid')

# Import the data

dataset = pd.read_csv('GridData1oldcorrected.csv')

corr = dataset.corr(method='kendall')
rcParams['figure.figsize'] = 14.7,8.27
sb.heatmap(corr, 
           xticklabels=corr.columns.values, 
           yticklabels=corr.columns.values, 
           cmap="YlGnBu",
          annot=True)

