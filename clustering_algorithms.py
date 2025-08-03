# Foundations of Data Mining - Practical Task 1
# Version 2.0 (2023-11-02)
###############################################
# Template for a custom clustering library.
# Classes are partially compatible to scikit-learn.
# Aside from check_array, do not import functions from scikit-learn, tensorflow, keras or related libraries!
# Do not change the signatures of the given functions or the class names!

import numpy as np
from sklearn.utils import check_array


class CustomKMeans:
    def __init__(self, n_clusters=8, max_iter=300, random_state=None):
        """
        Creates an instance of CustomKMeans.
        :param n_clusters: Amount of target clusters (=k).
        :param max_iter: Maximum amount of iterations before the fitting stops (optional).
        :param random_state: Initialization for randomizer (optional).
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X: np.ndarray, y=None):
        """
        This is the main clustering method of the CustomKMeans class, which means that this is one of the methods you
        will have to complete/implement. The method performs the clustering on vectors given in X. It is important that
        this method saves the centroids in "self.cluster_centers_" and the labels (=mapping of vectors to clusters) in
        the "self.labels_" attribute! As long as it does this, you may change the content of this method completely
        and/or encapsulate the necessary mechanisms in additional functions.
        :param X: Array that contains the input feature vectors
        :param y: Unused
        :return: Returns the clustering object itself.
        """
        # Input validation:
        X = check_array(X, accept_sparse='csr')

        # Calculation of cluster centers:
        self.cluster_centers_ = self.initialize_centers(X)

        # Determination of labels:
        for _ in range(self.max_iter):
            # Assignment step: Assign each data point to the nearest cluster
            distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
            self.labels_ = np.argmin(distances, axis=1)

            # Update step: Update the cluster centers based on the mean of points in each cluster
            new_centers = np.array([X[self.labels_ == i].mean(axis=0) for i in range(self.n_clusters)])

            # Check for convergence
            if np.all(new_centers == self.cluster_centers_):
                break

            self.cluster_centers_ = new_centers

        return self

    def initialize_centers(self, X):

        # Shuffle the data and take the first n_clusters points as initial cluster centers
        np.random.seed(self.random_state)
        indices = np.arange(X.shape[0])        # Creates an array of indices from 0 to the number of data points in X
        np.random.shuffle(indices)             # Shuffles the array of indices in-place
        return X[indices[:self.n_clusters]]    # Takes n_clusters indices and uses them to index the data in X.

    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Calls fit() and immediately returns the labels. See fit() for parameter information.
        """
        self.fit(X)
        return self.labels_


class CustomDBSCAN:
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        """
        Creates an instance of CustomDBSCAN.
        :param min_samples: Equivalent to minPts. Minimum amount of neighbors of a core object.
        :param eps: Short for epsilon. Radius of considered circle around a possible core object.
        :param metric: Used metric for measuring distances (optional).
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_ = None

    def fit(self, X: np.ndarray, y=None):
        """
        This is the main clustering method of the CustomDBSCAN class, which means that this is one of the methods you
        will have to complete/implement. The method performs the clustering on vectors given in X. It is important that
        this method saves the determined labels (=mapping of vectors to clusters) in the "self.labels_" attribute! As
        long as it does this, you may change the content of this method completely and/or encapsulate the necessary
        mechanisms in additional functions.
        :param X: Array that contains the input feature vectors
        :param y: Unused
        :return: Returns the clustering object itself.
        """
        # Input validation:
        X = check_array(X, accept_sparse='csr')

        # Determination of labels:
        self.labels_ = np.full(X.shape[0], -1)  # Initialize all points as noise (-1)
        cluster_label = 0

        for i in range(X.shape[0]):
            if self.labels_[i] != -1:
                continue  # Skip points that have already been assigned to a cluster

            neighbors = self._get_neighbors(X, i)

            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1  # Mark as noise
            else:
                cluster_label += 1
                self._expand_cluster(X, i, neighbors, cluster_label)

        return self

    def _get_neighbors(self, X, idx):
        """
        Get the neighbors of a point within the specified epsilon distance.
        """
        distances = np.linalg.norm(X - X[idx], axis=1)
        return np.where((0 < distances) & (distances <= self.eps))[0]

    def _expand_cluster(self, X, core_idx, neighbors, cluster_label):
        """
        Expand the cluster by adding connected points to the cluster.
        """
        self.labels_[core_idx] = cluster_label

        for neighbor in neighbors:
            if self.labels_[neighbor] == -1:
                self.labels_[neighbor] = cluster_label
                new_neighbors = self._get_neighbors(X, neighbor)

                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.concatenate([neighbors, new_neighbors])

    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Calls fit() and immediately returns the labels. See fit() for parameter information.
        """
        self.fit(X)
        return self.labels_
