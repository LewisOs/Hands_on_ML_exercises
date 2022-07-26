#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 14:48:08 2022

@author: lewis
"""

import numpy as np

def image_shifter(image, x_size=28, y_size=28):
    """Takes a 1D np.array representing a grayscale image and returns four 
    edited versions, each shifted up, down, left or right by 1 pixel in 2D 
    space. N.b. images are returned as 1D np.arrays. 
    
    Shifting is donevby duplicating a column/row of one side of the image 
    and delete the opposite most column/row"""
    
    image_2D = np.array(image).reshape(x_size, y_size)
    
    left = np.column_stack((image_2D[:, 1:], image_2D[:, -1]))
    left = left.flat
    
    right = np.column_stack((image_2D[:, 0], image_2D[:, 0:-1]))
    right = right.flat
    
    up = np.row_stack((image_2D[1:, :], image_2D[-1, :]))
    up = up.flat
    
    down = np.row_stack((image_2D[0, :], image_2D[0:-1, :]))
    down = down.flat
    
    return left, right, up, down
    
    