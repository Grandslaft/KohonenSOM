import numpy as np

class KohonenSOM:
    # init here stands for writing down all existing layers and their relationships between each other
    def __init__(self, layer):
        self.layer = layer
    
    def train(self, X, epochs = 10, lr = 0.1, radius = 0.1, lr_decay = 0.1, r_decay = 0.1, R_0 = None):
        # through epochs
        for epoch in range(epochs):
            # for every data, image in our case   
            for x in X:
                # find euclidean distance
                distances = np.linalg.norm(x - self.layer.weights, axis = 1)
                # find the 'shortest'
                min_d_ind = np.argmin(distances)
                
                # if it is greater than R_0, then we create a new neuron equal 
                # to the x with which the model tried to find a neighbor
                if R_0 is not None and distances[min_d_ind] > R_0:
                    self.layer.add_neuron(x)
                
                # updating weights
                self.layer.update_weights(x, min_d_ind, lr, radius)
            
            # decaying learning rate and radius    
            lr *= np.exp(-epoch*lr_decay)
            radius *= np.exp(-epoch*r_decay)

    