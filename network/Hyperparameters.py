import json
import os

class Hyperparameters:

    def __init__(self, file_path):
        with open(file_path) as f:
            param_dict = json.load(f)

        self.dataset_name = param_dict['dataset']
        self.layers = {int(i):layer  for i, layer  in param_dict['layers'].items()}
        self.loss = param_dict['loss']
        self.init_method = param_dict['init_method']
        self.optimizer = param_dict['optimizer']
        self.batch_size = param_dict['batch_size']
        self.n_epochs = param_dict['n_epochs']
        self.learning_rate = param_dict['learning_rate']

