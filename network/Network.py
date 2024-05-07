import numpy as np
import json 
from network.Layer import Layer
from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import os

class Network:
    """
    Represents the whole network
    ...

    Attributes
    -------
    param : Hyperparameters
        object that stores all hyperparameters used to train the network, imported from .json file parameters.json

    """

    def __init__(self, input_size, loss_function, init_method, batch_size, n_epochs, learning_rate, optimizer):
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

    def add_layer(self, units, activation):
        if len(self.layers) == 0:
            self.layers[1] = Layer(units, self.input_size, activation, self.init_method)
        else:
            l = len(self.layers)
            self.layers[l+1] = Layer(units, self.layers[l].layer_units, activation, self.init_method)

    def forward_pass(self, input):
        """ performs forward pass through the whole network """
        self.input = input
        self.layers[1].forward_pass(self.input)

        for i in range(2, len(self.layers)+1):
            self.layers[i].forward_pass(self.layers[i-1].a)

    def compute_loss(self, y):

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

    def compute_gradients(self):
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

    def gradient_descent(self):
        pass
        if self.optimizer == 'sgd':
            for i in self.layers:
                self.layers[i].weights -= self.learning_rate * self.w_grad[i]
                self.layers[i].bias -= self.learning_rate * self.b_grad[i]

        elif self.optimizer == 'adam':
            beta1, beta2, epsilon = 0.9, 0.999, 1e-8

            for i in self.layers:
                self.layers[i].weight_momentum = beta1*self.layers[i].weight_momentum + (1-beta1)*self.w_grad[i]
                self.layers[i].weight_velocity = beta2 * self.layers[i].weight_velocity + (1 - beta2) * np.power(self.w_grad[i],2)
                self.layers[i].bias_momentum = beta1*self.layers[i].bias_momentum + (1-beta1)*self.b_grad[i]
                self.layers[i].bias_velocity = beta2 * self.layers[i].bias_velocity + (1 - beta2) * np.power(self.b_grad[i],2)

                self.layers[i].weights -= self.learning_rate * (self.layers[i].weight_momentum / (1 - beta1)) / (epsilon + np.sqrt(self.layers[i].weight_velocity / (1 - beta2)))
                self.layers[i].bias -= self.learning_rate * (self.layers[i].bias_momentum / (1 - beta1)) / (epsilon + np.sqrt(self.layers[i].bias_velocity / (1 - beta2)))

        else:
            raise ValueError('invalid optimizer')

    def train(self, X, y):
        for _ in range(self.n_epochs):
            loss_epoch = []
            for k in range(0, len(X), self.batch_size):
                self.forward_pass(X[:,k:k+self.batch_size])
                self.compute_loss(y[:,k:k+self.batch_size])
                self.compute_gradients()
                self.gradient_descent()
                loss_epoch.append(self.loss)

            self.epoch_loss.append(np.mean(loss_epoch))

    def predict(self, X):
        self.forward_pass(np.expand_dims(X, axis=1))
        return np.argmax(self.layers[len(self.layers)].a)

    def compute_accuracy(self, X_, y_):
        count_correct = 0
        for i in range(len(X_)):
            count_correct += self.predict(X_[i]) == np.argmax(y_[i])
        return count_correct/len(y_)