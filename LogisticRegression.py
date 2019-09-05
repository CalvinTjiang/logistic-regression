import numpy as np 

class LogisticRegression:
    def __init__(self, total_features=1):
        pass
    
    def predict(self, x):
        '''
        Predict an array of output from an array of input data using sigmoid function 
        
        Parameters:
        x : array of data -> numpy array with shape = (n * m)

        Return :
        array of predicted output -> numpy array with shape = (n)
        '''

    def cost(self, x, y):
        '''
        calculate the error rate of the current weight compared with x and y

        Parameters:
        x : array of data -> numpy array with shape = (n * m)
        y : array of output -> numpy array with shape = (n * m)

        Return :
        error rate of the current weight -> float
        '''

    def gradient_descent(self, x, y, learning_rate=None, batch_size=None, total_epochs=None):
        '''
        apply gradient descent to reduced the error rate of the current weight
        by using derivative of the cost function
        
        Parameters:
        x : array of data -> numpy array with shape = (n * m)
        y : array of output -> numpy array with shape = (n * m)
        learning_rate : learning rate of the gradient descent -> positive float
        batch_size : size of each batch -> positive int
        total_epoch : total x and y being fully iterated -> positive int

        Return:
        None
        '''