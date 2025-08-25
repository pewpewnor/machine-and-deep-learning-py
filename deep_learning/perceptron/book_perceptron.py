import numpy as np


class Perceptron:
    """
    Literally only a single perceptron

    Inputs    Perceptron
    --------------------
    O       ->
    O       -> O
    O       ->
    --------------------
    """

    def __init__(self, num_features):
        self.weights = np.zeros(num_features)
        self.bias = 0

        self.misclassifications = []

    def net_input(self, X):
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1, 0)

    def fit(self, training_data, labels, eta, epochs):
        for _ in range(epochs):
            misclassifications_in_current_epoch = 0

            for inputs, label in zip(training_data, labels):
                delta = eta * (label - self.predict(inputs))

                # formula for delta_w is n * (y_expected - y_output) * x_input
                # formula for delta_b is n * (y_expected - y_output)
                delta_w = delta * inputs
                delta_b = delta
                self.weights += delta_w
                self.bias += delta_b

                misclassifications_in_current_epoch += int(delta != 0)
            self.misclassifications.append(misclassifications_in_current_epoch)
