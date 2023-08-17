from random import shuffle

from typing import List, Tuple, Dict, Union
import numpy as np
import time
from tqdm import tqdm

from Loading_Datasets import train_set
from utils import sigmoid, sigmoid_deriv, initialize_weights, initialize_biases, plot


def backpropagation(training_set: List[Tuple[np.ndarray, np.ndarray]],
                    layers_size: List[int],
                    hyperparams: Dict[str, Union[int, float]],
                    convergence_cost):
    learning_rate = hyperparams.get('learning_rate', 1.0)
    batch_size = hyperparams.get('batch_size', 10)

    n, m = training_set[0][0].shape[0], len(training_set)

    w1, w2, w3 = initialize_weights(layers_size, array_generator_func=np.random.randn)
    b1, b2, b3 = initialize_biases(layers_size[1:], array_generator_func=np.zeros)

    costs = []
    cost = 1000000000

    epochs_taken = 0
    while cost > convergence_cost:
        epochs_taken += 1
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

    print(f'LR = {learning_rate} | Reached cost < {convergence_cost} after {epochs_taken} epochs')
    plot(costs, hyperparams, 'vectorization')
    return epochs_taken


if __name__ == "__main__":
    train_set = train_set
    layers_size = [102, 150, 60, 4]
    hyperparams = {
        'learning_rate': 5,
        'num_epochs': 20,
        'batch_size': 10,
    }

    convergence_cost = 5

    num_trials = 5
    epochs_taken_ls = []
    for i in range(num_trials):
        print(f'Trial#{i}\t', end='')
        epochs_taken = backpropagation(training_set=train_set,
                                       layers_size=layers_size,
                                       hyperparams=hyperparams,
                                       convergence_cost=convergence_cost)
        epochs_taken_ls.append(epochs_taken)

    print(f'AVG #Epochs taken to converge for lr={2}: {sum(epochs_taken_ls) / num_trials}')
