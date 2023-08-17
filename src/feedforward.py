import numpy as np

from Loading_Datasets import train_set
from utils import sigmoid, initialize_weights, initialize_biases


def feedforward(training_set, weights, biases):
    w1, w2, w3 = weights
    b1, b2, b3 = biases

    hits = 0
    for image in training_set:
        x = image[0]
        y = image[1]

        z1 = np.dot(w1, x) + b1
        a1 = sigmoid(z1)

        z2 = np.dot(w2, a1) + b2
        a2 = sigmoid(z2)

        z3 = np.dot(w3, a2) + b3
        a3 = sigmoid(z3)

        if np.argmax(a3) == np.argmax(y):
            hits += 1

    accuracy = hits / len(training_set)
    print(f"Accuracy: {accuracy}")
    return accuracy


if __name__ == '__main__':
    layers_size = [102, 150, 60, 4]

    weights = initialize_weights(layers_size)
    biases = initialize_biases(layers_size[1:])

    feedforward(
        train_set,
        weights,
        biases
    )
