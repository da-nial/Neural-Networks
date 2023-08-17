from random import shuffle

from typing import List, Tuple, Dict, Union
import numpy as np
from tqdm import tqdm

import time
from Loading_Datasets import train_set
from utils import sigmoid, sigmoid_deriv, initialize_weights, initialize_biases, plot


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
        print(f'EPOCH#{epoch}', flush=True)
        hits = 0
        cost = 0

        shuffle(training_set)

        num_batches = int(m / batch_size)
        for batch in tqdm(range(num_batches)):
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

                for i in range(4):
                    for j in range(60):
                        d_w3[i, j] += a2[j][0] * sigmoid_deriv(z3[i][0]) * (2 * a3[i][0] - 2 * y[i][0])
                        d_b3[i][0] += sigmoid_deriv(z3[i][0]) * (2 * a3[i][0] - 2 * y[i][0])

                d_a2 = np.zeros((60, 1))
                for i in range(60):
                    for j in range(4):
                        d_a2[i, 0] += w3[j, i] * sigmoid_deriv(z3[j, 0]) * (2 * a3[j, 0] - 2 * y[j, 0])

                for i in range(60):
                    for j in range(150):
                        d_w2[i, j] += a1[j, 0] * sigmoid_deriv(z2[i, 0]) * (d_a2[i][0])
                        d_b2[i][0] += sigmoid_deriv(z2[i, 0]) * (d_a2[i][0])

                d_a1 = np.zeros((150, 1))
                for i in range(150):
                    for j in range(60):
                        d_a1[j, 0] += w2[j, i] * sigmoid_deriv(z2[j, 0]) * d_a1[j][0]

                for i in range(150):
                    for j in range(102):
                        d_w1[i, j] += x[j, 0] * sigmoid_deriv(z1[i, 0]) * d_a1[i][0]
                        d_b1[i][0] += sigmoid_deriv(z1[i, 0]) * d_a1[i][0]

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
    print(f"Accuracy: {accuracy}\n")
    plot(costs, hyperparams, 'backpropagation')


if __name__ == "__main__":
    train_set = train_set[:200]
    layers_size = [102, 150, 60, 4]
    hyperparams = {
        'learning_rate': 1,
        'num_epochs': 5,
        'batch_size': 10,
    }

    start = time.perf_counter()
    accuracy = backpropagation(training_set=train_set, layers_size=layers_size, hyperparams=hyperparams)
    end = time.perf_counter()

    exec_time = end - start
    print(f"Execution Time: {exec_time}s")
