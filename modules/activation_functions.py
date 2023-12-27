import numpy as np
from scipy.special import erf

class Activation:
    def __init__(self, name, alpha=1):
        self.name = name
        if name == 'sigmoid':
            self.function = self.exponent
            self.derivative = self.exponent_der
        elif name == 'tanh':
            self.function = lambda x: np.tanh(x)
            self.derivative = lambda x: 1 - np.square(np.tanh(x))
        elif name == 'softmax':
            self.function = self.softmax
            self.derivative = self.softmax_der
        elif name == 'relu':
            self.function = lambda x: np.maximum(0, x)
            self.derivative = lambda x: x >= 0
        elif name == 'lrelu':
            self.function = lambda x: np.maximum(0.0001, x) 
            self.derivative = lambda x: np.where(x >= 0, 1, 0.0001)
        elif name == 'elu':
            self.function = lambda x: np.where(x >= 0, x, alpha*(np.exp(x) - 1))
            self.derivative = lambda x: np.where(x >= 0, 1, self.function(x) + alpha)
        elif name == 'ls':
            self.function = self.saturated_perceptron
        elif name == 'relu':
            self.function = lambda x: np.maximum(0, x)
        elif name == 'sgelu':
            self.function = lambda x: np.maximum(x * (1 + erf(x/np.sqrt(2)))/2, x)
        elif name == 'ssilu':
            self.function = lambda x: np.maximum(x * 1/(1 + np.exp(-x)), x)
        elif name == 'smish':
            self.function = lambda x: np.maximum(x * np.tanh(np.log(1 + np.exp(x))), x)
        
    def exponent(self, x):
        shift_x = x - np.max(x)
        return 1/(1 + np.exp(-shift_x))
    
    def exponent_der(self, x):
        exps = self.exponent(x)
        return exps * (1 - exps)
    
    def softmax(self, x):
        shift_x = x - np.max(x)
        exps = np.exp(shift_x)
        return exps/np.sum(exps)
    
    def softmax_der(self, x):
        array = self.softmax(x).reshape(-1,1)
        return np.diagflat(array) - np.dot(array, array.T)
    
    def saturated_perceptron(self, x):
        temp = x.copy()
        temp[temp < 0] = 0
        temp[temp > 1] = 1
        return temp