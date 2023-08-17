import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))


def initialize_weights(layers_size: List[int], array_generator_func=np.random.randn):
    weights = []

    weights_dimensions = list(zip(layers_size[1:], layers_size))
    for dimension in weights_dimensions:
        if array_generator_func is np.zeros:
            w = array_generator_func(dimension)
        else:
            w = array_generator_func(*dimension)
        weights.append(w)

    return weights


def initialize_biases(layers_size: List[int], array_generator_func=np.zeros):
    biases = []

    for dimension in layers_size:
        b = array_generator_func((dimension, 1))
        biases.append(b)
    return biases


def plot(costs, hyperparams, file_name):
    sns.set()

    plt.title("Cost-Epoch Plot")
    num_epochs = len(costs)
    x = range(0, num_epochs)
    plt.plot(x, costs)
    plt.figtext(.6, .6, f"learning_rate = {hyperparams.get('learning_rate')}\n"
                        f"Num Epochs = {hyperparams.get('num_epochs')}\n"
                        f"Batch Size = {hyperparams.get('batch_size')}")
    plt.savefig(f'./output/{file_name}.png')


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
