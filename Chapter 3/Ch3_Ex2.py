#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 14:46:59 2022

Write a function that can shift an MNIST image in any direction (left, right, 
up, down), but 1 pixel. Then, for each image in the training set, create four 
shifted copies (one per direction) and add them to the training set. Finally, 
train your best model on this expanded training set and measure its accuracy 
on the test set. This technique of artificially growing or expanding the 
training set is called data augmentation or training set expansion.

- MNIST image shifter done - in MNIST_shifter.py
- Shifted data saved as .npy files 

TO DO:
- Import best model params from exe 1
- build model 
- train model
- compare performance with model from exe 1


@author: lewis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

# importing models
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier

X = np.load('MNIST_X_shifted.npy', allow_pickle=True)
y = np.load('MNIST_y_shifted.npy', allow_pickle=True) 

sss = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 0)
for train_index, test_index in sss.split(X,y):
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    
knn = KNeighborsClassifier
knn.fit(X_train, y_train)