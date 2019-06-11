# -*- coding: utf-8 -*-
# Name: André Moreira Souza
# NUSP: 9778985
# Course Code: SCC0251
# Semester: 2019/1
# Assignment: 4 - Colour image processing and segmentation
# %%
import numpy as np
import random
import imageio

# %%
# * Defining functions for assignment


def img_k_means(img, pixel_mode, k, n, S=None):
    """ Performs k-means algorithm of an input image.

    The result segmented image I is a m x n image,
    with labels defined by I(x, y) in {1, ..., k},
    that is, each pixel receives a label relative to it's cluster.
    """
    # Generating dataset for k-means
    dataset = dataset_gen_from_img(img, pixel_mode)
    # Selecting initial centroids (with S as the seed)
    random.seed(S)
    centroids = dataset[np.sort(
        random.sample(range(0, img.shape[0] * img.shape[1]), k))]

    # Initializing array of cluster id for each element of dataset
    clusters = np.zeros(dataset.shape[0], dtype=int)

    # Iterating
    for _ in range(k):
        # For each element of dataset,
        #   compute cluster based on minimum distance
        for idx, i in enumerate(dataset):
            clusters[idx] = np.nanargmin([
                np.linalg.norm(centroid - i, ord=2) for centroid in centroids
            ])
        # Update centroids with mean of its elements
        # centroids = calc_centroids(dataset, clusters, k)
        new_centroids = np.array(
            [np.nanmean(dataset[clusters == idx], axis=0) for idx in range(k)])

        # If there is no change to the centroids, break out of loop
        if np.allclose(centroids, new_centroids, equal_nan=True):
            break
        centroids = new_centroids

    return clusters.reshape(img.shape[:2])


def rmse(img_i, img_r):
    """ Root mean squared error (RMSE) for comparison."""
    return np.sqrt(
        (1 / (img_i.shape[0] * img_i.shape[1])) *
        np.sum(np.square(img_i.astype(float) - img_r.astype(float))))


def normalize(img, min_value=0, max_value=255):
    """ Normalizes input image between min_value and max_value."""
    return (min_value + (img - img.min()) * (max_value - min_value) /
            (img.max() - img.min()))


def dataset_gen_from_img(img, pixel_mode):
    """ Generates dataset from image, according to the selected pixel_mode

    Result can be used as an argument to k_means function
    pixel_mode: 1 -> RGB; 2 -> RGBxy; 3 -> luminance; 4 -> luminance, x, y;
    """
    dataset = None
    if (pixel_mode == 1):
        # R,G,B
        dataset = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
    elif (pixel_mode == 2):
        # R,G,B,x,y
        dataset = []
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                dataset.append(
                    [img[x, y, 0], img[x, y, 1], img[x, y, 2], x, y])
        dataset = np.array(dataset)
    elif (pixel_mode == 3):
        # Luminance
        dataset = np.array([
            0.299 * point[0] + 0.587 * point[1] + 0.114 * point[2]
            for point in img.reshape(img.shape[0] * img.shape[1], img.shape[2])
        ])
    elif (pixel_mode == 4):
        # Luminance, x, y
        dataset = []
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                dataset.append([
                    0.299 * img[x, y, 0] + 0.587 * img[x, y, 1] +
                    0.114 * img[x, y, 2], x, y
                ])
        dataset = np.array(dataset)
    else:
        raise ValueError("Invalid value for pixel_mode")
    return dataset.astype(np.double)


# %%
# * Main function
if __name__ == "__main__":
    # get user input and check validity
    # i = input image, r = reference image
    ipath, rpath = str(input()).strip(), str(input()).strip()
    # reading images
    if ipath.endswith('.npy'):
        img_i = np.load(ipath)
    else:
        img_i = imageio.imread(ipath)
    if rpath.endswith('.npy'):
        img_r = np.load(rpath)
    else:
        img_r = imageio.imread(rpath)
    # 1 -> RGB; 2 -> RGBxy; 3 -> luminance; 4 -> luminance, x, y;
    pixel_mode = int(input(), base=10)
    if (pixel_mode not in [1, 2, 3, 4]):
        raise ValueError(
            "Invalid input value for pixel_mode. Should be in [1, 2, 3, 4]")
    k = int(input(), base=10)  # k = number of clusters
    if (k <= 0):
        raise ValueError("k should be a positive integer value -> k > 0")
    n = int(input(), base=10)  # n = number of iterations
    if (n <= 0):
        raise ValueError("n should be a positive integer value -> n > 0")
    S = input()  # User-defined seed
    random.seed(S)  # Testing errors for S

    # Compute k-means, normalize and compare with reference image
    print('%.4f' % rmse(
        normalize(img_k_means(img_i, pixel_mode, k, n, S), 0, 255), img_r))
