import numpy as np


class Perceptrons:
    """
    My perceptrons with a single hidden layer of multiple perceptrons which feed
    into a single perceptron (that is the output peceptron).
    Every single perceptron will have weights and a bias, the weighted sum will
    be converted to either 1 or 0.

    Inputs    Hidden Perceptrons  Output perceptron
    -----------------------------------------------
    O       -> O                ->
    O       -> O                -> O
    O       -> O                ->
    -----------------------------------------------
    """

    def __init__(self, num_features, num_perceptrons):
        self.num_features = num_features
        self.num_perceptrons = num_perceptrons

        self.hidden_weights = np.random.randn(num_features, num_perceptrons)
        self.hidden_biases = np.random.randn(num_perceptrons)

        self.output_weights = np.random.randn(num_perceptrons)
        self.output_bias = np.random.randn()

    def forward(self, inputs):
        hidden_outputs = (
            inputs @ self.hidden_weights + self.hidden_biases >= 0
        ).astype(int)
        final_output = (
            hidden_outputs @ self.output_weights + self.output_bias >= 0
        ).astype(int)
        return hidden_outputs, final_output

    def predict(self, inputs):
        _, final_output = self.forward(inputs)
        return final_output

    def fit(
        self,
        training_data,
        training_label,
        epochs,
        eta,
        validation_data=None,
        validation_label=None,
    ):
        for epoch in range(1, epochs + 1):
            for inputs, label in zip(training_data, training_label):
                hidden_outputs, final_output = self.forward(inputs)

                # this if check is technically not necessary since if error = 0, then delta is also 0
                # but, this will reduce unnecessary training computational time
                if final_output == label:
                    continue

                # updates for all hidden perceptrons
                for i in range(self.num_perceptrons):
                    # this if check is not technically necessary for the same reason as before
                    if hidden_outputs[i] == label:
                        continue

                    # formula for delta_w is n * (y_expected - y_output) * x_input
                    # formula for delta_b is n * (y_expected - y_output)

                    hidden_error = label - hidden_outputs[i]
                    delta_hidden_w = eta * hidden_error * inputs
                    delta_hidden_bias = eta * hidden_error
                    self.hidden_weights[:, i] += delta_hidden_w
                    self.hidden_biases[i] += delta_hidden_bias

                # update for the single output perceptron
                output_error = label - final_output
                delta_output_w = eta * output_error * hidden_outputs
                delta_output_b = eta * output_error
                self.output_weights += delta_output_w
                self.output_bias += delta_output_b

            # print validation accuracy after training on current epoch
            if validation_data is not None and validation_label is not None:
                epoch_accuracy = self.score(validation_data, validation_label)
                print(f"Epoch {epoch} validation accuracy: {epoch_accuracy}%")

    def score(self, test_data, test_label):
        correct_predictions = sum((self.predict(test_data) == test_label).astype(int))
        return round(correct_predictions / len(test_label) * 100, 2)
