import numpy as np


class Neuron:
    def __init__(self, num_inputs):
        weights, bias = he_initialization(num_inputs)
        self.weights = weights
        self.bias = bias

    def forward(self, inputs):
        return relu_activation(self.weighted_sum(inputs))

    def weighted_sum(self, inputs):
        assert len(inputs) == len(self.weights)
        return np.dot(inputs, self.weights) + self.bias

    def __str__(self):
        return f"""Neuron information:
weights = {self.weights}
bias = {self.bias}"""


def he_initialization(num_inputs):
    weights = np.random.randn(num_inputs) * np.sqrt(2.0 / num_inputs)
    bias = 0.0
    return weights, bias


def relu_activation(x):
    return np.maximum(0, x)
