from Loading_Datasets import train_set, test_set
from feedforward import feedforward
from test_model import backpropagation

if __name__ == "__main__":
    layers_size = [102, 150, 60, 8]
    hyperparams = {
        'learning_rate': 1,
        'num_epochs': 20,
        'batch_size': 10,
    }

    num_trials = 10
    train_accuracies = []
    test_accuracies = []
    for i in range(num_trials):
        print(f'Trial #{i}')
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
