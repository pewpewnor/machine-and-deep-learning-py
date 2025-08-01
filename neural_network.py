from typing import List

import numpy as np


class NeuralNetwork:
    def __init__(self, sizes: List[int]):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(
        self,
        training_data,
        epochs: int,
        mini_batch_size: int,
        eta: float,
        test_data=None,
    ):
        for epoch in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[i : i + mini_batch_size]
                for i in range(0, len(training_data), mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                accuracy = self.evaluate(test_data)
                print(f"Epoch {epoch + 1}: {accuracy}% accuracy")
            else:
                print(f"Epoch {epoch} complete")

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for image, label in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backpropagation(image, label)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        scaled_eta = eta / len(mini_batch)
        self.biases = [b - scaled_eta * nb for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w - scaled_eta * nw for w, nw in zip(self.weights, nabla_w)]

    def backpropagation(self, image, label):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # forward pass below

        a = image
        activations = [a]
        weighted_sums = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            weighted_sums.append(z)
            a = sigmoid(z)
            activations.append(a)

        # backward pass below

        # formula for finding error for last layer (L) which is output layer:
        # δ^L = ∇aC ⊙ σ'(z^L)
        delta = cost_derivative(activations[-1], label) * sigmoid_prime(
            weighted_sums[-1]
        )
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(self.num_layers - 1, 1, -1):
            z = weighted_sums[l]
            # formula for finding error for layers before the last layer (all hidden layers)
            # δ^l= ( (wl+1)^T . δ^(l+1) ) ⊙ σ'(z^l)
            delta = np.dot(self.weights[l + 1].T, delta) * sigmoid_prime(z)
            # the error is exactly the same as the rate of change of the cost with respect to any bias in the NN
            # formula: ∂C/∂b_lj = δlj
            nabla_b[l] = delta
            nabla_w[l] = np.dot(delta, activations[l - 1].T)

        return nabla_b, nabla_w

    def evaluate(self, test_data):
        correct = 0
        for image, label in test_data:
            prediction = np.argmax(self.feedforward(image))
            if prediction == np.argmax(label):
                correct += 1
        return round(correct / len(test_data) * 100, 2)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def cost_derivative(output_activations, y):
    return output_activations - y
