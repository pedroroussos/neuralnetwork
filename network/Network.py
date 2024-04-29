import numpy as np
import json 
from Layer import Layer

import os

class Network:
    """
    Represents the whole network
    ...

    Attributes
    -------
    param : dict
        dictionary that stores all hyperparameters used to train the network, imported from .json file parameters.json

    """

    def __init__(self, param):
        self.layers = {}
        self.param = param
        self.init_layers(param['init_method'])

    def init_layers(self, init_method):
        """
        initiate Layer class instances based on param dict data, stores layers on another dictionary (self.layers)
        """
        for i, l in self.param['layers'].items():
            i = int(i)
            if i>0:
                self.layers[i] = Layer(l['units'], self.param['layers'][str(i-1)]['units'], l['activation'], init_method)

    def forward_pass(self, input):
        """ performs forward pass through the whole network """
        self.layers[1].forward_pass(input)

        for i in range(2, len(self.layers)):
            self.layers[i].forward_pass(self.layers[i-1].a)

with open(f'parameters.json') as f:
    parameters = json.load(f)

n = Network(parameters)

x = np.ones((3,2))
n.forward_pass(x)

print(n.__dict__)