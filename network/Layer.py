import numpy as np
from nptyping import NDArray

class Layer:
    """
    Represents a single layer of the network
    ...

    Attributes
    -------
    layer_units : int
        number of "neurons" of the layer
        
    previous_units : int
        number of "neurons" of the previous layer (or input, if layer is the first hidden)

    activation : str
        activation function of the layer (sigmoid, relu, linear, softmax, leaky_relu, tanh)
        
    init_method : str
        method used to initialize weights and biases of the layer
        
    weights : NDArray
        weights of the layer on matrix form
        
    bias : NDArray
        biases of the layer on matrix form
        
    z : NDArray
        pre-activations
        
    a : NDArray
        outputs of the layer
        
    da : NDArray
        derivative of the activation function with respect to the pre-activations
        
    w_m : NDArray
        weights gradient (used for adam optimizing computing)
   
    b_m : NDArray
        biases gradient (used for adam optimizing computing)   
         
    w_v : NDArray
        weights pointwise squared gradient (used for adam optimizing computing)
        
    b_v : NDArray
        biases pointwise squared gradient (used for adam optimizing computing)
               
    Methods
    -------
    init_parameters() -> None
        initializes weights and biases of the layer
    
    forward_pass(input: NDArray) -> None
        performs the forward pass (dot product of weights and inputs plus bias).
        takes input data of shape [batch, input_units], stores output data [batch, layer_units]
        
    activate() -> None
        pass layer's pre-activations through activation function, storing in self.a vector. Also computes
        the derivative of the activation function with respect to the pre-activations, storing in self.da vector
    
    """

    seed = 123

    def __init__(self, layer_units: int, previous_units: int, activation: str, init_method: str):

        self.activation = activation
        self.layer_units = layer_units
        self.previous_units = previous_units
        self.init_method = init_method

        self.weights = np.zeros((self.layer_units, self.previous_units))
        self.bias = np.zeros((self.layer_units, 1))

        self.init_parameters()

        self.z = np.zeros(self.layer_units)
        self.a = np.zeros(self.layer_units)
        self.da = np.zeros((self.layer_units, self.layer_units))

        self.w_m = np.zeros_like(self.weights)
        self.w_v = np.zeros_like(self.weights)

        self.b_m = np.zeros_like(self.bias)
        self.b_v = np.zeros_like(self.bias)
    
    def init_parameters(self) -> None:
        """ initializes weights and biases of the layer """

        np.random.seed(Layer.seed)
        Layer.seed += 1

        if self.init_method == 'xavier':
            self.weights = np.random.randn(*self.weights.shape) * np.sqrt(1/self.weights.shape[1])
            
        elif self.init_method == 'he':
            self.weights = np.random.randn(*self.weights.shape) * np.sqrt(2/self.weights.shape[1])

        elif self.init_method == 'normal':
            self.weights = np.random.randn(*self.weights.shape)
            
        elif self.init_method == 'uniform':
            self.weights = np.random.uniform(-0.5, 0.5, size=self.weights.shape)

        else:
            raise ValueError("invalid init method")


    def forward_pass(self, input: NDArray) -> None:
        """
        performs the forward pass (dot product of weights and inputs + bias).
        takes input data of shape [batch, input_units], stores output data [batch, layer_units]
        """
        self.z = np.dot(self.weights, input) + self.bias
        self.activate()

    def activate(self) -> None:
        """
        pass layer's pre-activations through activation function, storing in self.a vector
        also computes the derivative of the activation function with respect to the pre-activations,
        storing in self.da vector
        """

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



