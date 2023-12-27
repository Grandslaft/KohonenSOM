import numpy as np

class Hamming_NN:
    # init here stands for writing down all existing layers and their relationships between each other
    def __init__(self, layers, E, classes = None, pt = 0.05):
        self.classes = classes
        self.predict_threshold = pt
        
        # if user gave 1 or less layers than we raise a value error
        if len(layers) < 2:
            raise ValueError("Not enough layers. Should be at least 2")
        
        # first layer in the array will always be an input layer
        self.layers = [layers[0]]
        
        # rest of the layers we link with each other and save linked layers in self.layers
        for layer in layers[1:]:
            self.layers[-1].add_layer(layer)
            if layer is not None and layer not in self.layers:
                self.layers.append(layer)
        
        self.n_inputs, n_neurons = E.shape
        self.layers[0].weights = E.T / 2
        self.layers[0].bias = n_neurons / 2

    def predict(self, X, max_epochs = np.inf):
        # second return is data for the next layer or Z, but with activation_function used on it
        _, A = self.layers[0].forward(X)

        # through all the epochs
        for _ in range(max_epochs):
            # going through maxnet layer
            _, A = self.layers[1].forward(A)
            # until only one nonzero value left
            if np.count_nonzero(A) != A.shape[0]:
                break
        return A.reshape(-1, self.n_inputs)
    
    # predict, but for each class separated
    def predict_class_separated(self, X, Y, max_epochs = np.inf):
        correctly_predicted = []
        accuracy_score = []
        for i in range(self.n_inputs):
            X_i = X[Y == i]
            predictions = self.get_predictions(self.predict(X_i, max_epochs))
            correctly_predicted.append(np.sum(predictions == i))
            accuracy_score.append(correctly_predicted[-1] / len(predictions))
            
        return np.array(accuracy_score), np.array(correctly_predicted)
    
    # there was an attempt to force model to not classify data, 
    # if it's not sure, but I failed, as you can see on the 
    # graphs down from here with vanishing and noising data
    # otherwise it works just fine the only thing, I can remove the code between #! wrong class
    def get_predictions(self, A):
        if self.classes is not None:
            return np.array(list(map(lambda i: self.classes[i], np.argmax(A, axis=1))))
        #! wrong class
        if (np.max(A, axis=1) <= self.predict_threshold).any():
            temp = np.argmax(A, axis=1)
            temp[np.max(A, axis=1) <= self.predict_threshold] = -1
            return temp
        #! wrong class
        return np.argmax(A, axis=1)