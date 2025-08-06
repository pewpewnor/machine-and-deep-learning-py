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

                # this if check is technically not necessary since if error = 0, then delta is also 0
                # but, this will reduce unnecessary training computational time
                if final_output == label:
                    continue

                # formula for delta_w is n * (y_expected - y_output) * x_input
                # formula for delta_b is n * (y_expected - y_output)

                output_error = label - final_output
                delta_output_w = eta * output_error * hidden_outputs
                delta_output_b = eta * output_error
                self.output_weights += delta_output_w
                self.output_bias += delta_output_b

                for i in range(self.num_perceptrons):
                    # this if check is not technically necessary for the same reason as before
                    if hidden_outputs[i] != label:
                        hidden_error = label - hidden_outputs[i]
                        delta_hidden_w = eta * hidden_error * inputs
                        delta_hidden_bias = eta * hidden_error
                        self.weights[i] += delta_hidden_w
                        self.biases[i] += delta_hidden_bias
