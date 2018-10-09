# -*- coding: utf-8 -*-
#CS450 03Prove
#author: Shawn Lilly

import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score # to get accuracy
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # Built in KNN


class KNNClassifier: # actual KNN class itself
    def __init__(self, k, data=None, target=None): # constructor
        if data is None:
            data = []
        if target is None:
            target = []
        self.target = target
        self.k = k
        self.data = data
        
    
    def fit(self, data, target): # makes the model 
        self.target = target
        self.data = data
        return self
    
    def predict(self, testData): # find out which neighbors are clossest
        nInputs = np.shape(testData)[0]
        closest = np.zeros(nInputs)
        
        # all the maths for finding the nearest neighbors
        for n in range(nInputs):
            # find distances
            distance = np.sum((self.data - testData[n, :])**2, axis = 1)
            # find the nearest neighbours
            indices = np.argsort(distance, axis = 0)
            classes = np.unique(self.target[indices[:self.k]])
            
            
            if len(classes) == 1: # the length of the class
                closest[n] = np.unique(classes)
            else:
                counts = np.zeros(max(classes) + 1)
                for i in range(self.k):
                    counts[self.target[indices[i]]] += 1
                closest[n] = np.max(counts)
        return closest
    
    def score(self, xTest, yTest, sampleWeight=None): # find acuracy
        return accuracy_score(yTest, self.predict(xTest), sampleWeight = sampleWeight)

"""
all the magic for the car autism and mpg datasets
"""
def main():
    carData, carTest = carDataPreP()
    #autData, autTest = autDataPreP()
    mpgData, mpgTest = mpgDataPreP()
    
    n_neighbors = 5
    
    performScoreClassifier('Car', KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', n_jobs=-1, ), carData, carTest)
    performScoreClassifier('M.P.G.', KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance', n_jobs=-1, ), mpgData, mpgTest)


def performScoreClassifier(name, classifier, data, test):
    print(name)
    # k-fold Cross Validation
    scores = cross_val_score(classifier, data, test, cv=10)
    accuracyOutput(scores)
    
# get acuracy
def accuracyOutput(scores):
    print("Accuracy: {0:.0f}% +/-{1:.2f}".format(scores.mean() * 100, scores.std() * 2))



#dealing with non numerical data for car dataset
def carDataPreP():
    # get data set from url
    carData = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car"
                               ".data", header=None)
    
    # Set column names on data frame
    columns = "buying maint doors persons lug_boot safety target".split()
    carData.columns = columns
    
    # turn the car data to namber values
    columnNum = {"buying": {"vhigh": 4, "high": 3, "med": 2, "low": 1},
                 "maint": {"vhigh": 4, "high": 3, "med": 2, "low": 1},
                 "doors": {"2": 1, "3": 2, "4": 3, "5more": 4},
                 "persons": {"2": 1, "4": 2, "more": 3},
                 "lug_boot": {"small": 1, "med": 2, "big": 3},
                 "safety": {"low": 1, "med": 2, "high": 3},
                 "target": {"unacc": 1, "acc": 2, "good": 3, "vgood": 4}}
    
    carData.replace(columnNum, inplace=True)
    
    # split data as well as targets into separate data frames
    carTargets = carData.iloc[:, 6:]
    carData = carData.iloc[:, :6]
    
    # turn the data and target data frames into lists
    carDataArray = carData.as_matrix()
    carTargetsArray = carTargets.as_matrix()
    carTargetsArray = carTargetsArray.flatten()
    
    return carDataArray, carTargetsArray

#placeholder for autism dataset
"""
def autDataPreP(): 
    # get data set from txt
    
    # Set column names on data frame
    columns = "buying maint doors persons lug_boot safety target".split()
    autData.columns = columns
    
    # turn the car data to namber values
    
    # split data as well as targets into separate data frames
    autTargets = autData.iloc[:, 6:]
    autData = autData.iloc[:, :6]
    
    # turn the data and target data frames into lists
    autDataArray = carData.as_matrix()
    autTargetsArray = carTargets.as_matrix()
    autTargetsArray = carTargetsArray.flatten()
    
    return autDataArray, autTargetsArray
"""

#dealing with missing data for miles per gallon
def mpgDataPreP():
    # get data set from url
    mpgData = pd.read_csv(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
            delim_whitespace=True, na_values='?')
    
    # Set column names on data frame
    columns = "mpg cyl disp hp weight accel year origin model".split()
    mpgData.columns = columns
    
    # replace and remove missing values and connected rows rows
    mpgData.dropna(inplace=True)
    mpgData.drop(columns='model', inplace=True)
    
    # split it into data and targets
    mpgTargets = mpgData.iloc[:, :1]
    mpgData = mpgData.iloc[:, 1:8]
    
    mpgDataArray = mpgData.as_matrix()
    mpgTargArray = mpgTargets.as_matrix()
    mpgTargArray = mpgTargArray.flatten()
    
    return mpgDataArray, mpgTargArray


if __name__ == '__main__':
    main()