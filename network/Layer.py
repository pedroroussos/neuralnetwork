import numpy as np
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
        activation function of the layer (sigmoid, relu, linear, softmax, leaky_relu, tanh)

    """

    def __init__(self, layer_units, previous_units, activation, init_method):
        self.activation = activation

        self.layer_units = layer_units

        self.weights = np.zeros((self.layer_units, previous_units))
        self.bias = np.zeros((self.layer_units, 1))
        self.init_parameters(init_method)

        self.weight_momentum = np.zeros_like(self.weights)
        self.weight_velocity = np.zeros_like(self.weights)

        self.bias_momentum = np.zeros_like(self.bias)
        self.bias_velocity = np.zeros_like(self.bias)

        self.z = np.zeros(self.layer_units)
        self.a = np.zeros(self.layer_units)
        self.da = np.zeros((self.layer_units, self.layer_units))
    
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
        self.z = np.dot(self.weights, input) + self.bias
        self.activate()

    def activate(self):
        """ pass layer's pre-activations through activation funtion """

        if self.activation == 'sigmoid':
            self.a = 1/(1+np.exp(-self.z))
            self.da = self.a*(1-self.a)

        elif self.activation == 'softmax':
            exps = np.exp(self.z)
            self.a = exps / np.sum(exps, axis=0)
            self.da = np.zeros((self.layer_units, self.layer_units))

            n = self.a.shape[0]


            for i in range(self.a.shape[1]):
                tmp = np.tile(self.a[:,i].reshape(n,1), n)
                self.da += tmp * (np.identity(n) - np.transpose(tmp))

            self.da /= n

        elif self.activation == 'relu':
            self.a = np.maximum(0, self.z)
            self.da = (self.a > 0) * 1

        elif self.activation == 'leaky_relu':
            alpha = 0.01
            self.a = np.maximum(alpha * self.z, self.z)
            self.da = np.ones_like(self.a)
            self.da[self.a<0] = alpha

        elif self.activation == 'linear':
            self.a = self.z
            self.da = np.ones_like(self.a)

        elif self.activation == 'tanh':
            self.a = np.tanh(self.z)
            self.da = 1.-np.power(self.a, 2)

        else:
            raise ValueError("invalid activation function")



