# To compute the average precision at level tau
# We need the latent representations of my data
# We need to create the distance matrix.
# Using only training data, we need to compute a value
# of tau (positive real number) that maximizes
# the average precision.
# To compute the average precision select a point
# For every point in the training data locate all
# neighbors that are inside a ball with radius equal
# to tau. Count how many of these neighbors have
# the same label as the query point and divide
# this count by the number of neighbors in the ball
# After finding the best value for tau
# use the latent representations of the test data
# to compute the accuracy of the classifier.
# The classifier will be a majority vote classifier
# focusing on a ball with radius tau.

import os
import pickle
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

# ----------------------- CONFIG FILE ----------------------------- #
path_train_latent = './70_by_70_pca/train_pca_BB_Buy_autoencoder.pkl'
path_test_latent = './70_by_70_pca/test_pca_BB_Buy_autoencoder.pkl'
path_train_positive = './70_by_70_images_2010_2017/BB_Buy/'
path_test_positive = './70_by_70_images_2018/BB_Buy/'
tau = 1.0
# ------------------- END CONFIG FILE ----------------------------- #

def label_files(files: pd.Series, path_positive: str):
    files_pos_class = os.listdir(path_positive)
    labels = np.zeros(files.shape)
    for i, f in enumerate(files):
        f = f.replace('\\', '/')
        if f.split('/')[-1] in files_pos_class:
            labels[i] = 1.0
    return labels

def get_average_accuracy(dist:  np.ndarray,
                         tau: float,
                         source_labels: np.ndarray,
                         neighbors_labels: np.ndarray,
                        train_vs_train: bool = False):
    # number of samples
    n = source_labels.shape[0]
    avg = 0.0
    for i in range(dist.shape[0]):
        idx = np.where(dist[i, :] <= tau)[0]
        # Considers the fact that, when using
        # only train data, the nearest neighbor to a point
        # is the point itself
        if train_vs_train:
            agree = sum(neighbors_labels[idx] == source_labels[i]) - 1
        else:
            agree = sum(neighbors_labels[idx] == source_labels[i])
            
        avg = avg +  agree / idx.shape[0]
    return avg / n

with open(path_train_latent, 'rb') as f:
    latent_train = pickle.load(f)
    train_files = latent_train['file']
    latent_train = latent_train.iloc[:, 1:].to_numpy()

with open(path_test_latent, 'rb') as f:
    latent_test = pickle.load(f)
    test_files = latent_test['file']
    latent_test = latent_test.iloc[:, 1:].to_numpy()

# Compute Euclidean distances between points in the trainig/test set
# Printing a message in this step would be nice
train_dist = cdist(latent_train, latent_train)
test_dist = cdist(latent_test, latent_train)

# Get labels for train/test dataset
train_labels = label_files(train_files, path_train_positive)
test_labels = label_files(test_files, path_test_positive)

# Average accuracy between points in training data
avg_train = get_average_accuracy(train_dist,
                                tau,
                                train_labels,
                                train_labels,
                                True)
