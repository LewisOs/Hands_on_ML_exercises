#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:38:58 2022

A script to generate a pixel shifted version of the MNIST data.

@author: lewis
"""

import numpy as np
from ImageShifter import image_shifter
from sklearn.datasets import fetch_openml

mnist = fetch_openml(name='mnist_784')

X_data, y_data = mnist.data, mnist.target.astype('float64')

X_left = np.zeros((70000, 784))
y_left = np.zeros(70000)

X_right = np.zeros((70000, 784))
y_right = np.zeros(70000,)

X_up = np.zeros((70000, 784))
y_up = np.zeros(70000,)

X_down = np.zeros((70000, 784))
y_down = np.zeros(70000,)

ys = [y_left, y_right, y_up, y_down]  

print("Shifting starting...")

for i in range(len(mnist.data)):
    X_left[i], X_right[i], X_up[i], X_down[i] = image_shifter(X_data[i])
    for y in ys:
        y[i] = y_data[i]
    if i % 1000 == 0:
        print(f"{i} of 70000 images shifted so far.")

print("Image shifting complete.")

X_shifted = np.row_stack((X_data, X_left, X_right, X_up, X_down))
y_shifted = np.concatenate((y_data, y_left, y_right, y_up, y_down))