#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 11:09:39 2022

Try to build a classifier for the MNIST dataset that achieves over 97% 
accuracy on the test set. Hint: the KNeighborsClassifier works quite well
for this task; you just need to find good hyperparameter values (try a grid
search on the weights and n_neighbors hyperparameters).

@author: lewis
"""
# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
#from sklearn.model_selection import 

# importing models
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier

# Importing data
from sklearn import datasets
digits = datasets.load_digits()

# splitting the data into train & test sets
X, y = digits.data, digits.target # digits.data.shape is (n_samples, n_features)

sss = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 0)
for train_index, test_index in sss.split(X,y):
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]

#X_train, X_test, y_train, y_test = train_test_split(
  #  X, y, test_size = 0.2, random_state = 0)

# Built-in data set, so no missing values, no cleaing needed.

