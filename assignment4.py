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

# * Performs k-means algorithm of an input image
def k_means(img, pixel_mode, k, n):
    img_ret = np.zeros(img.shape, dtype=np.float)
    ids = np.sort(random.sample(range(0, img_i.shape[0]*img_i.shape[1]), k))
    return img_ret

# * Root mean squared error (RMSE) for comparison
def rmse(img_i, img_r):
    return np.sqrt((1/(img_i.shape[0] * img_i.shape[1])) * np.sum(np.square(img_i.astype(float) - img_r.astype(float))))

# * Computes the euclidean distance of two images
def euclidean_distance(img1, img2):
    if(len(img1.shape) == 3):
        return np.sum(np.square(img1.astype(float)-img2.astype(float)))/img1.shape[2]
    elif(len(img1.shape) == 2):
        return np.sum(np.square(img1.astype(float)-img2.astype(float)))
    else: raise TypeError("Invalid shape for img1")

# * Normalizes input image between min_value and max_value
def normalize(img, min_value=0, max_value=255):
    return min_value + \
        (img-img.min())*(max_value-min_value) / (img.max() - img.min())

#%%
# * Main function
if __name__ == "__main__":
    # get user input and check validity
    ipath, rpath = str(input()).strip(), str(input()).strip() # i = input image, r = reference image
    img_i, img_r = imageio.imread(ipath), imageio.imread(rpath) # reading images
    pixel_mode = int(input(), base=10) # 1 -> RGB; 2 -> RGBxy; 3 -> luminance; 4 -> luminance, x, y;
    if(pixel_mode not in [1, 2, 3, 4]): raise ValueError("Invalid input value for pixel_mode. Should be in [1, 2, 3, 4]")
    k = int(input(), base=10) # k = number of clusters
    if(k <= 0): raise ValueError("k should be a positive integer value -> k > 0")
    n = int(input(), base=10) # n = number of iterations
    if(n <= 0): raise ValueError("n should be a positive integer value -> n > 0")
    S = input() # User-defined seed
    random.seed(S)

    # Compute k-means and output
    # TODO: Uncomment next line to test k_means function
    print('%.4f' % rmse(k_means(img_i, pixel_mode, k, n), img_r))

#%%