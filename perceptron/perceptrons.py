import numpy as np


class Perceptrons:
    def __init__(self, num_features, num_perceptrons):
        self.num_features = num_features
        self.num_perceptrons = num_perceptrons

        self.weights = np.random.randn(num_perceptrons, num_features)
        self.biases = np.random.randn(num_perceptrons)

        self.output_weights = np.random.randn(num_perceptrons)
        self.output_bias = np.random.randn()

    def forward(self, inputs):
        hidden_outputs = (np.dot(self.weights, inputs) + self.biases >= 0).astype(int)
        final_output = (
            np.dot(self.output_weights, hidden_outputs) + self.output_bias >= 0
        ).astype(int)
        return hidden_outputs, final_output

    def predict(self, inputs):
        _, final_output = self.forward(inputs)
        return final_output

    def fit(self, training_data, epochs, eta):
        for _ in range(epochs):
            for inputs, label in training_data:
                hidden_outputs, final_output = self.forward(inputs)

                if final_output == label:
                    continue

                # formula for delta_w is n * (y_expected - y_output) * x_input
                # formula for delta_b is n * (y_expected - y_output)

                delta_output_w = eta * (label - final_output) * hidden_outputs
                delta_output_b = eta * (label - final_output)
                self.output_weights += delta_output_w
                self.output_bias += delta_output_b

                for i in range(self.num_perceptrons):
                    if hidden_outputs[i] != label:
                        delta_hidden_w = eta * (label - hidden_outputs[i]) * inputs
                        delta_hidden_bias = eta * (label - hidden_outputs[i])
                        self.weights[i] += delta_hidden_w
                        self.biases[i] += delta_hidden_bias
