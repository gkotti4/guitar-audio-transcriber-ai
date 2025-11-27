import numpy as np

# region Activation Functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1 / (np.exp(x))

def softmax(x):
    z = x - np.max(x, axis=1, keepdims=True) # stabilize - subtract the maxes out of the 
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# endregion
