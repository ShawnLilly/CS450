# -*- coding: utf-8 -*-
"""   
    Created on Sat Sep 21 14:49:40 2018
      I found most of this online and have just been messing around with it trying
      to figure it out. I couldn't fnd out how to use the defauld iris data so i used
      a persons online file because he explaind how to mess with it. I struggled a lot
      but this was my best atempt.
"""
# Load the libraries
import numpy as np
import pandas as pa
from sklearn import datasets
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


#class myclassifier:
    #add fit and predict methods
    
# Load in the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pa.read_csv(url, names=names)

# this will Split-out the dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.30 # 30 percent to test on
seed = 10 #randomizing value
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
#above randomizes and makes matrix

# these are the Test options and evaluation metric
seed = 7
scoring = 'accuracy'
#Accuracy =      #TP + #FP
           #TP + #FP + #TN + #FN

#Make predictions on validation dataset
knn = KNeighborsClassifier() #this is where I should implament my classifier
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(confusion_matrix(Y_validation, predictions))