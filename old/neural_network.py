import numpy as np
from layer import Layer


class NeuralNetwork:
    def __init__(self, num_inputs, hidden_layer, num_outputs):
        num_hidden_layers, num_hidden_layer_neurons = hidden_layer
        assert num_hidden_layers > 0
        # add first hidden layer
        self.layers = [Layer(num_inputs, num_hidden_layer_neurons)]
        # add rest of hidden layers
        for _ in range(num_hidden_layers - 1):
            self.layers.append(
                Layer(num_hidden_layer_neurons, num_hidden_layer_neurons)
            )
        # add output layer
        self.layers.append(Layer(num_hidden_layer_neurons, num_outputs))

    def forward(self, inputs):
        logits = self.layers[0].forward(inputs)
        for layer in self.layers[1:]:
            logits = layer.forward(logits)
        probabilities = softmax(logits)
        assert np.isclose(
            np.sum(probabilities), 1, rtol=1e-9, atol=1e-12
        )  # sum of probabilities must basically be 1 (100%)
        return probabilities


def softmax(logits):
    shifted_x = logits - np.max(logits)
    exp_x = np.exp(shifted_x)
    return exp_x / np.sum(exp_x)
