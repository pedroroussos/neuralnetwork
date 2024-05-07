import numpy as np
import matplotlib.pyplot as plt
from network import *
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

def main() -> None:

    x, y = load_digits(return_X_y=True)

    le = LabelBinarizer()
    le.fit(y)
    y = le.transform(y)
    x /= 255
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=6, test_size=0.3, stratify=y)


    nn = Network(
        input_size = 64,
        loss_function = 'cce',
        init_method = 'xavier',
        batch_size = 1,
        n_epochs = 100,
        learning_rate = 0.01,
        optimizer = 'adam'
    )

    nn.add_layer(
        units = 128,
        activation = 'sigmoid'
    )

    nn.add_layer(
        units = 10,
        activation = 'softmax'
    )

    nn.train(x_train.T, y_train.T)
    print(nn.compute_accuracy(x_test, y_test))


if __name__=='__main__':
    main()
