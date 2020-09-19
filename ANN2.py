# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:27:33 2020

@author: user1
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import matplotlib.pyplot as plt
    
dataset = pd.read_csv('TotalGridData.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

model = Sequential()
model.add(Dense(64, input_shape=(36,), activation='tanh'))
model.add(Dense(32, activation='tanh'))
model.add(Dense(24, activation='tanh'))
model.add(Dense(19, activation='softmax'))

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])


history = model.fit(X_train, y_train, epochs=100,
                    validation_data=(X_test, y_test)
                    )

scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
# plot_loss_accuracy(history)

test11_18 = pd.read_csv('grid_11_18_old.csv')
X11_18 = test11_18.iloc[:,:].values
y_probabilities=model.predict_proba(X11_18)
print(y_probabilities)