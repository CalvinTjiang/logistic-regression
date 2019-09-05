import numpy as np 
import numpy.ma as ma 
class LogisticRegression:
    def __init__(self, total_features=1):
        self.total_features = total_features
        self.weight = np.random.rand(total_features)
    
    def predict(self, x):
        '''
        Predict an array of output from an array of input data using sigmoid function 
        
        Parameters:
        x : array of data -> numpy array with shape = (n * m)

        Return :
        array of predicted output -> numpy array with shape = (n)
        '''
         # Check if x have a same number of features with weight
        if len(x[0]) != self.total_features:
            raise Exception("Number of features in x is not equal with total feature in weight!")
        
        # Calculate Z
        z = x @ self.weight[np.newaxis, :].T

        # Return Prediction
        prediction = (1 / (1 + np.exp(-z)))
        return prediction

    def cost(self, x, y):
        '''
        calculate the error rate of the current weight compared with x and y

        Parameters:
        x : array of data -> numpy array with shape = (n * m)
        y : array of output -> numpy array with shape = (n * m)

        Return :
        error rate of the current weight -> float
        '''
        prediction = self.predict(x)
        one = np.ones(len(y))
        
        # Calculate Difference for y[i] = 1
        a = np.multiply(y, ma.log(prediction).filled(0))

        # Calculate Difference for y[i] = 0
        b = np.multiply(one - y, ma.log(one - prediction).filled(0))
        difference = a + b
        total_cost = -np.sum(difference, axis=0) / len(y)
        return total_cost


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