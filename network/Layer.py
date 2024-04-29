import numpy as np
from activation import *
import json

class Layer:

    seed = 123

    """
    Represents a single layer of the network
    ...

    Attributes
    -------
    layer_units : int
        number of "neurons" in the layer

    activation : str
        activation function of the layer (sigmoid, ReLU, linear, softmax, leaky_ReLU, tanh)

    """

    def __init__(self, layer_units, previous_units, activation, init_method):
        self.activation = eval(activation)

        self.layer_units = layer_units

        self.weights = (self.layer_units, previous_units)
        self.bias = np.zeros((self.layer_units, 1))
        self.init_parameters(init_method)

        self.z = np.zeros(self.layer_units)
        self.a = np.zeros(self.layer_units)
    
    def init_parameters(self, init_method):
        """
        takes the init_method as parameter to initialize weights and biases of the layer 
        """

        np.random.seed(Layer.seed)
        Layer.seed += 1

        if init_method == 'xavier':
            self.weights = np.random.randn(*self.weights.shape) * np.sqrt(1/self.weights.shape[1])
            
        elif init_method == 'he':
            self.weights = np.random.randn(*self.weights.shape) * np.sqrt(2/self.weights.shape[1])

        elif init_method == 'normal':
            self.weights = np.random.randn(*self.weights.shape)
            
        elif init_method == 'uniform':
            self.weights = np.random.uniform(-0.5, 0.5, size=self.weights.shape)

        else:
            raise ValueError("invalid init method")


    def forward_pass(self, input):
        """
        performs the forward pass (dot product of weights and inputs + bias).
        takes input data of shape [batch, input_units], stores output data [batch, layer_units]
        
        z[batch, layer_units] -> pre-activations
        a[batch, layer_units] -> layer outputs
        """

        self.z = np.matmul(self.weights, input) + self.bias
        self.a = self.activation(self.z)


