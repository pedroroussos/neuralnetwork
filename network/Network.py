import numpy as np
from network.Layer import Layer
from nptyping import NDArray

class Network:
    """
    Represents the entire network.
    ...

    Attributes
    -------
    input_size : integer
        Length of input vector.

    loss_function : string
        Name of the loss function (bce: binary cross-entropy; cce: categorical cross-entropy;
        mse: mean squared error).

    init_method : string
        Method used to initialize weights across the network.

    batch_size : integer
        Training batch size.

    n_epochs : integer
        Number of training epochs.

    learning_rate : float
        Rate at which the parameters are updated.

    optimizer : string
        Optimizing method (sgd: stochastic gradient descent; adam: adaptive momentum).

    current_epoch : integer
        Stores the current epoch number the network is training at. It will be used in the adam optimizer


    Methods
    -------
    add_layer(units: int, activation: str) -> None
        Adds a layer with the specified number of units and activation function to the end of the network.

    forward_pass(input: npArray) -> None
        Performs a forward pass through the network given an input.

    compute_loss(y: NDArray) -> None
        Computes the loss of the network given the real value of y.

    compute_gradients() -> None
        Computes the backpropagation gradients.

    gradient_descent() -> None
        Updates the weights and biases of all layers using the specified optimizing method.

    train(x: NDArray, y: NDArray) -> None
        Performs the training loop on the network, given inputs (x) and outputs (y).

    predict(x: NDArray) -> NDArray
        Given an input to the network, returns the output.

    compute_accuracy(x: NDArray, y: NDArray) -> float
        Given inputs and outputs of a classification problem, returns the accuracy of the prediction.

    """

    def __init__(self,
                 input_size: int,
                 loss_function: str,
                 init_method: str,
                 batch_size: int,
                 n_epochs: int,
                 learning_rate: float,
                 optimizer: str):

        self.input_size = input_size
        self.loss_function = loss_function
        self.init_method = init_method
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer

        self.layers = {}
        self.input = np.zeros((self.input_size, self.batch_size))
        self.loss = 100
        self.dloss = np.array([])
        self.w_grad = {}
        self.b_grad = {}
        self.epoch_loss = []
        self.current_epoch = 0

    def add_layer(self, units:int, activation:str) -> None:
        if len(self.layers) == 0:
            self.layers[1] = Layer(units, self.input_size, activation, self.init_method)
        else:
            l = len(self.layers)
            self.layers[l+1] = Layer(units, self.layers[l].layer_units, activation, self.init_method)

    def forward_pass(self, input) -> None:
        """ performs forward pass through network given an input """
        self.input = input
        self.layers[1].forward_pass(self.input)

        for i in range(2, len(self.layers)+1):
            self.layers[i].forward_pass(self.layers[i-1].a)

    def compute_loss(self, y: NDArray) -> None:

        y_hat = self.layers[len(self.layers)].a

        if self.loss_function == 'bce':
            epsilon = 1e-7
            y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
            self.loss = np.sum(np.mean(-(1 - y) * np.log(1 - y_hat + 1e-7) - y * np.log(y_hat + 1e-7), axis=0))
            self.dloss = - (np.divide(y, y_hat) - np.divide(1 - y, 1 - y_hat))

        elif self.loss_function == 'mse':
            self.loss = np.mean(np.power(y_hat - y,2))/2
            self.dloss = y_hat-y

        elif self.loss_function == 'cce':
            self.loss = -1/len(y) * np.sum(np.sum(y * np.log(y_hat)))
            self.dloss = -y/(y_hat + 10**-100)

        else:
            raise ValueError('invalid loss function')

    def compute_gradients(self) -> None:
        global_grad = {
            len(self.layers): np.dot(self.layers[len(self.layers)].da, self.dloss)
        }

        local_grad = {
            1: self.input.T
        }

        for i in range(len(self.layers)-1, 0, -1):
            global_grad[i] = np.dot(self.layers[i + 1].weights.T, global_grad[i + 1]) * self.layers[i].da

        for i in range(2, len(self.layers)+1):
            local_grad[i] = self.layers[i-1].a.T

        for i in self.layers:
            self.w_grad[i] = global_grad[i]@local_grad[i]/self.batch_size
            self.b_grad[i] = np.expand_dims(np.mean(global_grad[i], axis=1), axis=1)

    def gradient_descent(self) -> None:
        pass
        if self.optimizer == 'sgd':
            for i in self.layers:
                self.layers[i].weights -= self.learning_rate * self.w_grad[i]
                self.layers[i].bias -= self.learning_rate * self.b_grad[i]

        elif self.optimizer == 'adam':
            beta1, beta2, epsilon = 0.9, 0.999, 1e-8

            for i in self.layers:
                self.layers[i].w_m = beta1*self.layers[i].w_m + (1-beta1)*self.w_grad[i]
                self.layers[i].w_v = beta2 * self.layers[i].w_v + (1 - beta2) * np.power(self.w_grad[i],2)
                self.layers[i].b_m = beta1*self.layers[i].b_m + (1-beta1)*self.b_grad[i]
                self.layers[i].b_v = beta2 * self.layers[i].b_v + (1 - beta2) * np.power(self.b_grad[i],2)

                w_m_hat = self.layers[i].w_m / (1 - np.power(beta1, self.current_epoch+1))
                w_v_hat = self.layers[i].w_v / (1 - np.power(beta2, self.current_epoch+1))
                b_m_hat = self.layers[i].b_m / (1 - np.power(beta1, self.current_epoch+1))
                b_v_hat = self.layers[i].b_v / (1 - np.power(beta2, self.current_epoch+1))

                self.layers[i].weights -= self.learning_rate * w_m_hat / (epsilon + np.sqrt(w_v_hat))
                self.layers[i].bias -= self.learning_rate * b_m_hat / (epsilon + np.sqrt(b_v_hat))

        else:
            raise ValueError('invalid optimizer')

    def train(self, x: NDArray, y: NDArray):
        for _ in range(self.n_epochs):
            self.current_epoch += 1
            loss_epoch = []
            for k in range(0, len(x), self.batch_size):
                self.forward_pass(x[:,k:k+self.batch_size])
                self.compute_loss(y[:,k:k+self.batch_size])
                self.compute_gradients()
                self.gradient_descent()
                loss_epoch.append(self.loss)

            self.epoch_loss.append(np.mean(loss_epoch))

    def predict(self, x: NDArray) -> None:
        self.forward_pass(np.expand_dims(x, axis=1))
        return self.layers[len(self.layers)].a


    def compute_accuracy(self, x: NDArray, y: NDArray) -> float:
        count_correct = 0
        for i in range(len(x)):
            count_correct += np.argmax(self.predict(x[i])) == np.argmax(y[i])
        return count_correct/len(y)