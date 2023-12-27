import numpy as np
from .activation_functions import Activation

class hamming_layer():
    def __init__(self, n_inputs, n_neurons, activation):
        # pointer for the next layer
        self.next_layer = None
        # weights and bias for current layer
        self.weights = None
        self.bias = None
        # layer's activation function
        self.activation = Activation(activation)
    
    def forward(self, X):
        Z = X.dot(self.weights) + self.bias # results of the layer aka predictions
        Z /= Z.max()
        X = self.activation.function(Z) # transformed data for the next layer
        return Z, X
    
    # adding next layer
    def add_layer(self, child):
        self.next_layer = child
        return self
    
    
class maxnet_layer():
    def __init__(self, n_inputs, n_neurons, activation, eps = None):
        # pointer for the next layer
        self.next_layer = None
        # weights and bias for current layer
        
        if eps is None:
            eps = np.random.uniform(0, 1/n_neurons)
            
        self.weights = np.full((n_neurons, n_inputs), -eps)
        np.fill_diagonal(self.weights, 1)
        self.bias = 0
        # layer's activation function
        self.activation = Activation(activation)
    
    def forward(self, X):
        Z = X.dot(self.weights) + self.bias # results of the layer aka predictions
        X = self.activation.function(Z) # transformed data for the next layer
        return Z, X
    
    # adding next layer
    def add_layer(self, child):
        self.next_layer = child
        return self
    
class kohonen_layer():
    def __init__(self, n_inputs, n_neurons):
        # random weights with values from uniform PDF
        self.weights = np.random.uniform(0.5 - 1/np.sqrt(n_neurons), 0.5 + 1/np.sqrt(n_neurons), size = (n_neurons, n_inputs))
    
    # function for adding neurons
    def add_neuron(self, x):
        self.weights = np.vstack((self.weights, x))
    
    # updating weights
    def update_weights(self, x, min_ind, lr, radius):
        # euclidean distance
        n_distances = np.linalg.norm(self.weights[min_ind] - self.weights, axis = 1)
        # if radius is too small updating only one weight
        if radius < 1e-3:
            self.weights[min_ind] += lr * (x - self.weights[min_ind])
        # otherwise updating weights for all the neighbors with scaling
        neighbours = n_distances <= radius
        h = np.exp(-n_distances[neighbours]/(2 * np.square(radius)))
        self.weights[neighbours] += lr * np.multiply(h.reshape(-1,1), (x - self.weights[neighbours]))
    
    # adding next layer
    def add_layer(self, child):
        self.next_layer = child
        return self