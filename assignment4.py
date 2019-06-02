# -*- coding: utf-8 -*-
# Name: André Moreira Souza
# NUSP: 9778985
# Course Code: SCC0251
# Semester: 2019/1
# Assignment: 4 - Colour image processing and segmentation
#%%
import numpy as np
import random
import imageio
import matplotlib.pyplot as plt # ! Remove before submitting

#%%
# * Defining functions for assignment


# * Root mean squared error (RMSE) for comparison
def rmse(img_i, img_r):
    return np.sqrt((1/(img_i.shape[0] * img_i.shape[1])) * np.sum(np.square(img_i.astype(float) - img_r.astype(float))))

#%%
# * Main function
if __name__ == "__main__":
    # get user input
    ipath, rpath = str(input()).strip(), str(input()).strip() # i = input image, r = reference image
    img_i, img_r = imageio.imread(ipath), imageio.imread(rpath) # reading images
    pixel_mode = int(input(), base=10) # 1 -> RGB; 2 -> RGBxy; 3 -> luminance; 4 -> luminance, x, y;
    k = int(input(), base=10) # k = number of clusters
    n = int(input(), base=10) # n = number of iterations
    S = input()

    # Checking input values' validity
    if(pixel_mode not in [1, 2, 3, 4]): raise ValueError("Invalid input value for pixel_mode. Should be in [1, 2, 3, 4]")
    if(k <= 0): raise ValueError("k should be a positive integer value -> k > 0")
    if(n <= 0): raise ValueError("n should be a positive integer value -> n > 0")
