from random import shuffle

from typing import List, Tuple, Dict, Union
import numpy as np
import time
from tqdm import tqdm

from Loading_Datasets import train_set
from utils import sigmoid, sigmoid_deriv, initialize_weights, initialize_biases, plot


def backpropagation(training_set: List[Tuple[np.ndarray, np.ndarray]],
                    layers_size: List[int],
                    hyperparams: Dict[str, Union[int, float]]):
    learning_rate = hyperparams.get('learning_rate', 1.0)
    num_epochs = hyperparams.get('num_epochs', 10)
    batch_size = hyperparams.get('batch_size', 10)
    momentum = hyperparams.get('momentum', 0.3)

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
        for batch in range(num_batches):
            d_w1, d_w2, d_w3 = initialize_weights(layers_size, array_generator_func=np.zeros)
            d_b1, d_b2, d_b3 = initialize_biases(layers_size[1:], array_generator_func=np.zeros)

            w1_velocity, w2_velocity, w3_velocity = np.copy(d_w1), np.copy(d_w2), np.copy(d_w3)

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

            w1_velocity = learning_rate * (d_w1 / batch_size) + (momentum * w1_velocity)
            w1 -= w1_velocity

            w2_velocity = learning_rate * (d_w2 / batch_size) + (momentum * w2_velocity)
            w2 -= w2_velocity

            w3_velocity = learning_rate * (d_w3 / batch_size) + (momentum * w3_velocity)
            w3 -= w3_velocity

            b1 -= learning_rate * (d_b1 / batch_size)
            b2 -= learning_rate * (d_b2 / batch_size)
            b3 -= learning_rate * (d_b3 / batch_size)

        costs.append(cost / m)

    accuracy = hits / m
    print(f"Accuracy: {accuracy}\n")
    plot(costs, hyperparams, 'momentum')
    return accuracy


if __name__ == "__main__":
    train_set = train_set[:200]
    layers_size = [102, 150, 60, 4]
    hyperparams = {
        'learning_rate': 1,
        'num_epochs': 20,
        'batch_size': 10,
        'momentum': 0.3,
    }

    num_trials = 10
    accuracies = []
    exec_times = []
    for i in range(num_trials):
        start = time.perf_counter()
        accuracy = backpropagation(training_set=train_set, layers_size=layers_size, hyperparams=hyperparams)
        end = time.perf_counter()

        exec_time = end - start
        accuracies.append(accuracy)
        exec_times.append(exec_time)

    print(f"AVG Accuracy: {sum(accuracies) / num_trials}")
    print(f"AVG Execution Time: {sum(exec_times) / num_trials}")
