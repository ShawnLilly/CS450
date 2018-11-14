# -*- coding: utf-8 -*-
#CS450 08Prove
#author: Shawn Lilly

import numpy as np
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoidDerivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        # input for nural network
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4) 
        self.weights2   = np.random.rand(4,1)    
        # output of nuralNetwork             
        self.y          = y
        self.output     = np.zeros(self.y.shape)
        
    # calculate the predictid value    
    def feedForward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    # go back updating the values for better predictions.
    def backProp(self):
        # apply the chain rule to find derivitivs of the loss function 
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoidDerivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoidDerivative(self.output), self.weights2.T) * sigmoidDerivative(self.layer1)))

        # update the weights 
        self.weights1 += d_weights1
        self.weights2 += d_weights2


if __name__ == "__main__":                      #Movie dataset
                                        #drama| actor |  budget| success   
    movie = np.array([[0,0,1],          # No |  Not   |  High  | No
                  [0,1,1],              # No | Famous |  High  | Yes
                  [1,0,0],              # Yes|  Not   |  Low   | Yes
                  [1,1,1],              # Yes| Famous |  High  | No
                  [0,0,0]])             # No |  Not   |  low   | No
    success = np.array([[0],[1],[1],[0],[0]])
    nn = NeuralNetwork(movie,success)
    
                                            
    #should loan dataset                                   
    shouldLoan = np.array([[2,1,1],         
                  [2,1,0],              
                  [2,0,1],             
                  [2,0,0],
                  [1,1,2],            
                  [1,0,0],             
                  [1,1,0],
                  [1,0,1],
                  [0,1,1],
                  [0,1,0],
                  [0,0,1],
                  [0,0,0]])    
    success2 = np.array([[1],[1],[1],[0],[1],[0],[1],[0],[1],[0],[0],[0]])
    nn1 = NeuralNetwork(movie,success)
    nn2 = NeuralNetwork(shouldLoan,success2)

    for i in range(1500): # num of iterations for movie
        nn1.feedForward()
        nn1.backProp()
        
    for i in range(1500): # num of iterations for loan
        nn2.feedForward()
        nn2.backProp()

    print('movie dataset')
    print(nn1.output)
    print('')
    print('loan dataset')
    print(nn2.output)
    
