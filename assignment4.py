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
    # Declaring return image
    img_ret = np.zeros(img.shape, dtype=np.uint8)
    # Generating dataset for k-means
    dataset = dataset_gen_from_img(img, pixel_mode)
    # Selecting initial centroids (with S as the seed)
    random.seed(S)
    ids = np.sort(random.sample(range(0, img_i.shape[0] * img_i.shape[1]), k))
    centroids = [dataset[i] for i in ids]

    # Iterating
    for _ in range(n):
        # Initializing sets of points for each cluster
        clusters = [[] for _ in range(k)]
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                # Computing Euclidean distance to each centroid
                pixel = dataset[x * img.shape[0] + y]
                dists = np.array([
                    np.linalg.norm(pixel - centroid) for centroid in centroids
                ])
                # Adding pixel to closest cluster and marking in the image
                min_dist = np.argmin(dists)
                clusters[min_dist].append(pixel)
                img_ret[x, y] = min_dist + 1
        # Recalculating centroids
        clusters = np.array(clusters)
        centroids = np.array([cluster.mean(axis=0) for cluster in clusters])

    return img_ret


def rmse(img_i, img_r):
    """ Root mean squared error (RMSE) for comparison."""
    return np.sqrt(
        (1 / (img_i.shape[0] * img_i.shape[1])) *
        np.sum(np.square(img_i.astype(float) - img_r.astype(float))))


def normalize(img, min_value=0, max_value=255):
    """ Normalizes input image between min_value and max_value."""
    return min_value \
        + (img-img.min())*(max_value-min_value) / (img.max() - img.min())


def dataset_gen_from_img(img, pixel_mode):
    """ Generates dataset from image, according to the selected pixel_mode

    Result can be used as an argument to k_means function
    pixel_mode: 1 -> RGB; 2 -> RGBxy; 3 -> luminance; 4 -> luminance, x, y;
    """
    dataset = None
    if (pixel_mode == 1):
        # R,G,B
        dataset = np.array([
            img[x, y, :] for x in range(img.shape[0])
            for y in range(img.shape[1])
        ],
                           dtype=np.double)
    elif (pixel_mode == 2):
        # R,G,B,x,y
        dataset = np.array([[img[x, y, 0], img[x, y, 1], img[x, y, 2], x, y]
                            for x in range(img.shape[0])
                            for y in range(img.shape[1])],
                           dtype=np.double)
    elif (pixel_mode == 3):
        # Luminance
        dataset = np.array([
            0.299 * img[x, y, 0] + 0.587 * img[x, y, 1] + 0.114 * img[x, y, 2]
            for x in range(img.shape[0]) for y in range(img.shape[1])
        ],
                           dtype=np.double)
    elif (pixel_mode == 4):
        # Luminance, x, y
        dataset = np.array([[
            0.299 * img[x, y, 0] + 0.587 * img[x, y, 1] + 0.114 * img[x, y, 2],
            x, y
        ] for x in range(img.shape[0]) for y in range(img.shape[1])],
                           dtype=np.double)
    else:
        raise ValueError("Invalid value for pixel_mode")
    return dataset


# %%
# * Main function
if __name__ == "__main__":
    # get user input and check validity
    # i = input image, r = reference image
    ipath, rpath = str(input()).strip(), str(input()).strip()
    # reading images
    img_i = imageio.imread(ipath)
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

    # Compute k-means and output
    print('%.4f' % rmse(
        normalize(img_k_means(img_i, pixel_mode, k, n, S), 0, 255).astype(
            np.uint8), img_r))
