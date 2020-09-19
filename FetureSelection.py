# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 12:46:30 2020

@author: user1
"""

from sklearn.feature_selection import f_classif
import numpy as np
import pandas as pd

#Preporocessing
dataset = pd.read_csv('Correctedresult.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

for i in range(1,19):
     X1=np.concatenate([X[:100,:],X[i*100:(i+1)*100,:]], axis =0)
     y1=np.concatenate([y[:100],y[i*100:(i+1)*100]], axis =0)
     f_score, f_p_value = f_classif(X1,y1)
     print('F - score score  {} '.format(i), f_score)

avgs = [ [ None for i in range(36) ] for j in range(19) ]
for i in range(0,36):
    for j in range(0,19):
        avgs[j][i]=np.average(X[j*100:(j+1)*100,i]);        


# f_score1= f_score
# max1= np.argmax(f_score1)
# f_score1[max1]= 0
# max2= np.argmax(f_score1)
# f_score1[max2]= 0
# max3= np.argmax(f_score1)
# #print('pairwise_tukeyhsd',pairwise_tukeyhsd)