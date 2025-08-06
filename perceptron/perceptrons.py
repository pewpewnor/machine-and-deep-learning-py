import numpy as np


class Perceptrons:
    def __init__(self, num_features, num_perceptrons):
        self.num_features = num_features
        self.num_perceptrons = num_perceptrons

        self.weights = np.random.randn(num_perceptrons, num_features)
        self.biases = np.random.randn(num_perceptrons)

    def forward(self, inputs):
        return (np.dot(self.weights, inputs) + self.biases >= 0).astype(int)

    def fit(self, training_data, epochs, eta):
        for _ in range(epochs):
            for inputs, label in training_data:
                outputs = self.forward(inputs)
                errors = np.full(self.num_perceptrons, label) - outputs

                delta_w = np.dot(eta * errors[:, np.newaxis], inputs[np.newaxis, :])
                delta_b = eta * errors

                self.weights += delta_w
                self.biases += delta_b
