import numpy as np

from neuron import Neuron


class Layer:
    def __init__(self, num_inputs, num_neurons):
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]

    def forward(self, inputs):
        return np.array([neuron.forward(inputs) for neuron in self.neurons])

    def __str__(self):
        return f"""Layer information:
neurons = {self.neurons}"""
