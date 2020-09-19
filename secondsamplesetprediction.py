# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 19:00:29 2020

@author: user1
"""

dataset2 = pd.read_csv('Correctedresult2.csv')
Xtest = dataset.iloc[:, :-1].values
ytest = dataset.iloc[:, -1].values

ypred = model.predict(Xtest)
predictions2 = [round(value) for value in ypred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))