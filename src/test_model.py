from random import shuffle

from typing import List, Tuple, Dict, Union
import numpy as np

from Loading_Datasets import train_set, test_set
from utils import sigmoid, sigmoid_deriv, initialize_weights, initialize_biases, plot
from feedforward import feedforward


def backpropagation(training_set: List[Tuple[np.ndarray, np.ndarray]],
                    layers_size: List[int],
                    hyperparams: Dict[str, Union[int, float]]):
    learning_rate = hyperparams.get('learning_rate', 1.0)
    num_epochs = hyperparams.get('num_epochs', 10)
    batch_size = hyperparams.get('batch_size', 10)

    n, m = training_set[0][0].shape[0], len(training_set)

    w1, w2, w3 = initialize_weights(layers_size, array_generator_func=np.random.randn)
    b1, b2, b3 = initialize_biases(layers_size[1:], array_generator_func=np.zeros)

    costs = []

    for epoch in range(num_epochs):
        hits = 0
        cost = 0

        shuffle(training_set)

        num_batches = int(m / batch_size)
        for batch in range(num_batches):
            d_w1, d_w2, d_w3 = initialize_weights(layers_size, array_generator_func=np.zeros)
            d_b1, d_b2, d_b3 = initialize_biases(layers_size[1:], array_generator_func=np.zeros)

            for image in training_set[batch * batch_size:(batch + 1) * batch_size]:
                x = image[0]
                y = image[1]

                z1 = (w1 @ x) + b1
                a1 = sigmoid(z1)

                z2 = (w2 @ a1) + b2
                a2 = sigmoid(z2)

                z3 = (w3 @ a2) + b3
                a3 = sigmoid(z3)

                cost += sum(pow((a3 - y), 2))

                d_w3 += (2 * sigmoid_deriv(z3) * (a3 - y)) @ (np.transpose(a2))
                d_b3 += (2 * sigmoid_deriv(z3) * (a3 - y))

                d_a2 = np.transpose(w3) @ (2 * sigmoid_deriv(z3) * (a3 - y))

                d_w2 += (sigmoid_deriv(z2) * d_a2) @ (np.transpose(a1))
                d_b2 += (sigmoid_deriv(z2) * d_a2)

                d_a1 = np.transpose(w2) @ (sigmoid_deriv(z2) * d_a2)

                d_w1 += (sigmoid_deriv(z1) * d_a1) @ (np.transpose(x))
                d_b1 += (sigmoid_deriv(z1) * d_a1)

                if np.argmax(z3) == np.argmax(y):
                    hits += 1

            w1 -= learning_rate * (d_w1 / batch_size)
            w2 -= learning_rate * (d_w2 / batch_size)
            w3 -= learning_rate * (d_w3 / batch_size)

            b1 -= learning_rate * (d_b1 / batch_size)
            b2 -= learning_rate * (d_b2 / batch_size)
            b3 -= learning_rate * (d_b3 / batch_size)

        costs.append(cost / m)

    accuracy = hits / m
    print(f"Accuracy: {accuracy}", end='\t')

    plot(costs, hyperparams, 'test_model')

    weights = [w1, w2, w3]
    biases = [b1, b2, b3]
    return accuracy, weights, biases


if __name__ == "__main__":
    layers_size = [102, 150, 60, 4]
    hyperparams = {
        'learning_rate': 1,
        'num_epochs': 5,
        'batch_size': 10,
    }

    num_trials = 10
    train_accuracies = []
    test_accuracies = []
    for i in range(num_trials):
        print(f'Trial #{i}', end='\t')
        print('Train ', end='')
        train_accuracy, weights, biases = backpropagation(training_set=train_set,
                                                          layers_size=layers_size,
                                                          hyperparams=hyperparams)
        print('Test ', end='')
        test_accuracy = feedforward(test_set, weights, biases)

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    print(f"AVG Train Accuracy: {sum(train_accuracies) / num_trials}")
    print(f"AVG Test Accuracy: {sum(test_accuracies) / num_trials}")
