# -*- coding: utf-8 -*-
#CS450 01Prove
#author: Shawn Lilly
"""
    Created on Fri Sep 21 14:49:40 2018
     This is a redo of my first atempt to get the hard coded classifier working
"""
import numpy as np
from sklearn import datasets # for iris dataset
from sklearn.metrics import accuracy_score # to get accuracy
from sklearn.naive_bayes import GaussianNB # classifier
from sklearn.model_selection import train_test_split  # function to make training easier


class HardcodedClassifier:
    def __init__(self): # constructor
        pass
    
    def fit(self, data, target): # placeholder
        return HardcodedClassifier()
    
    def predict(self, data_test): # placeholder 
        return np.zeros((data_test.shape[0]), dtype = int)
    
    def score(self, x_test, y_test, sample_weight=None): # find acuracy
        return accuracy_score(y_test, self.predict(x_test), sample_weight = sample_weight)


"""
where all the magic happens
"""
def main():
    iris = datasets.load_iris() # get iris data set
    irisD = iris.data
    irisT = iris.target  
    # Test set 30%
    data_train, data_test, target_train, target_test = train_test_split(irisD, irisT, test_size = .3)

    model_iris_gaussian_nb(data_test, data_train, target_test, target_train)
    model_iris_hardcoded(data_test, data_train, target_test, target_train)
    
    # GaussianNB classifier use
def model_iris_gaussian_nb(data_test, data_train, target_test, target_train):
    gnb_classifier = GaussianNB()
    gnb_classifier.fit(data_train, target_train)
    gnb_score = gnb_classifier.score(data_test, target_test)
    # print the gaussianNB acuracy
    print("(GaussianNB) Predicted accuracy: {:.0f}%".format(gnb_score * 100))
    
    # Hardcoded clasifier use
def model_iris_hardcoded(data_test, data_train, target_test, target_train):
    hc_classifier = HardcodedClassifier()
    hc_classifier.fit(data_train, target_train)
    hcc_score = hc_classifier.score(data_test, target_test)
    # print my hardcoded acuracy
    print("(Hardcoded) Predicted accuracy: {:.0f}%".format(hcc_score * 100))


if __name__ == '__main__':
    main()