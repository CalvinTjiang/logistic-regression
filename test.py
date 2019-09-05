import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
from LogisticRegression import *
import timeit

def generate_data(total_features, slope, randomness, range_y):
    '''
    Generate a data
        
    Parameters:
    x : array of data -> numpy array with shape = (n * m)

    Return :
    array of predicted output -> numpy array with shape = (n)
    '''
    range_y = np.arange(*range_y)
    y = np.zeros(len(range_y))
    y[:len(range_y)//2] = np.ones(len(range_y)//2)
    random_data = np.random.rand(len(y), total_features)

    # Update each feature with slope and randomness
    random_data[:,0] = np.ones(len(y))
    for i in range(0, total_features - 1):
        random_data[:,i + 1] += (range_y * slope[i]) + np.random.uniform(-randomness[i], randomness[i], (len(y)))
    return (random_data[:,0 : total_features], y)

# Generate random data
x, y = generate_data(3, [-0.5, 0.2], [0.3, -1], (0, 10, 0.05))

# Initialize the LogisticRegression object
logistic_regression = LogisticRegression(3)
logistic_regression.learning_rate = 0.3
logistic_regression.batch_size = 25
logistic_regression.total_epochs = 40

print(f'---Gradient Descent---')
print(f'Initial cost : {logistic_regression.cost(x, y)}')

# Start gradient Descent on first model
start = timeit.default_timer()
logistic_regression.gradient_descent(x, y)
logistic_regression.gradient_descent(x, y)
logistic_regression.gradient_descent(x, y)
logistic_regression.gradient_descent(x, y)
logistic_regression.gradient_descent(x, y)
logistic_regression.gradient_descent(x, y)
logistic_regression.gradient_descent(x, y)
logistic_regression.gradient_descent(x, y)
taken = (timeit.default_timer() - start)

print(f'Time Taken : {taken}')
print(f'Result Cost : {logistic_regression.cost(x, y)} \n')

# Predict the output
prediction = logistic_regression.predict(x)

# Initialize 3D subplot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Draw the point
ax.scatter(x[:,1], x[:,2], y)
ax.scatter(x[:,1], x[:,2], prediction, marker='^')

# Add the label
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Output')

# Show the 3D scatterplot
plt.show()


