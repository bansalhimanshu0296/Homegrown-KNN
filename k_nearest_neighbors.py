# k_nearest_neighbors.py: Machine learning implementation of a K-Nearest Neighbors classifier from scratch.
#
# Submitted by: Himanshu Himanshu -- hhimansh
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff

import numpy as np
from numpy.lib.function_base import select
from numpy.random.mtrand import random
from utils import euclidean_distance, manhattan_distance


class KNearestNeighbors:
    """
    A class representing the machine learning implementation of a K-Nearest Neighbors classifier from scratch.

    Attributes:
        n_neighbors
            An integer representing the number of neighbors a sample is compared with when predicting target class
            values.

        weights
            A string representing the weight function used when predicting target class values. The possible options are
            {'uniform', 'distance'}.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model and
            predicting target class values.

        _y
            A numpy array of shape (n_samples,) representing the true class values for each sample in the input data
            used when fitting the model and predicting target class values.

        _distance
            An attribute representing which distance metric is used to calculate distances between samples. This is set
            when creating the object to either the euclidean_distance or manhattan_distance functions defined in
            utils.py based on what argument is passed into the metric parameter of the class.

    Methods:
        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_neighbors = 5, weights = 'uniform', metric = 'l2'):
        # Check if the provided arguments are valid
        if weights not in ['uniform', 'distance'] or metric not in ['l1', 'l2'] or not isinstance(n_neighbors, int):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the KNearestNeighbors model object
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._X = None
        self._y = None
        self._distance = euclidean_distance if metric == 'l2' else manhattan_distance

    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """
        # fit function of knn is very easy just setting variable to class variable so it can be used in predict
        self._X = X
        self._y = y
    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """
        # taking empty result set to store classofication for each X point
        results = []

        # Iteratting through every point in X to find nearest neighbor
        for X_test in X:

            # taking empty list tos tore distance of each point with their category as tupple 
            distance_result = []

            # Iterating through every point of X_train
            for i in range(len(self._X)):

                # calculating distance of two given points
                distance = self._distance(self._X[i], X_test)

                # appending distance and their class as tupple to list
                distance_result.append((distance, self._y[i]))
            
            # sorting list with key as distance
            distance_result = sorted(distance_result, key= lambda distance: distance[0])

            # slicing no of neighbours needed to calculate the class 
            distance_result = distance_result[:self.n_neighbors]

            # making an empty category dictionary to store votes for each category
            categories = {}

            # checking if wieght is uniform
            if self.weights == "uniform":

                # Iterating through each neighbour
                for result in distance_result:

                    # if category is not in list initialising it
                    if result[1] not in categories:
                        categories[result[1]] = 0
                    
                    # Increasing vote of category by 1
                    categories[result[1]] += 1
            else:

                # this if weights are ditance
                # Iterating through each neighbour
                for result in distance_result:

                    # if category is not in list initialising it
                    if result[1] not in categories:
                        categories[result[1]] = 0

                    # if distance of neighbour is not 0 if not increase vote of that category by distance else 1 
                    if result[0] != 0:
                        categories[result[1]] += 1/result[0]
                    else:
                        categories[result[1]] += 1

            # append the key with maximum value to result list for that particular point
            results.append(max(categories, key = categories.get))

        # returning result array
        return results