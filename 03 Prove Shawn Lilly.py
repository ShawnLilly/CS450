# -*- coding: utf-8 -*-
#CS450 02Prove
#author: Shawn Lilly

import numpy as np 
from sklearn import datasets # for iris dataset
from sklearn.metrics import accuracy_score # to get accuracy
from sklearn.neighbors import KNeighborsClassifier # Built in KNN
from sklearn.model_selection import train_test_split # function to make training easier



"""
K nearest neighbor will go through comparing the iris flower to K of its nearest
neighbors to predict what type it is.
"""
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
    
    def predict(self, test_data): # find out which neighbors are clossest
        n_inputs = np.shape(test_data)[0]
        closest = np.zeros(n_inputs)
        
        # all the maths for finding the nearest neighbors
        for n in range(n_inputs):
            # find distances
            distance = np.sum((self.data - test_data[n, :])**2, axis = 1)
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
    
    def score(self, x_test, y_test, sample_weight=None): # find acuracy
        return accuracy_score(y_test, self.predict(x_test), sample_weight = sample_weight)

"""
where my KNN and default KNN are called and compared
both are about the same up untill 4 neighbors
"""
def main():
    
    Myk = 5 # my KNN number of neighbors
    BuiltK = 5 # built in KNN number of neighbors
    
    iris = datasets.load_iris() # get iris data set
    irisD = iris.data
    irisT = iris.target
    x_train, x_test, y_train, y_test = train_test_split(irisD, irisT, test_size = 0.3)
    
    
    my_knn = KNNClassifier(Myk)
    my_knn_model = my_knn.fit(x_train, y_train)
    #print my KNN
    print('My KNN accuracy: {:.0f}%'.format(100*my_knn_model.score(x_test, y_test))) 
    
    sklearn_nkk = KNeighborsClassifier(n_neighbors = BuiltK)
    sklearn_nkk_model = sklearn_nkk.fit(x_train, y_train)
    sklearn_nkk_model.predict(x_test)
    #print built in KNN
    print('Built in KNN accuracy: {:.0f}%'.format(100*sklearn_nkk_model.score(x_test, y_test)))

if __name__ == '__main__':
    main()