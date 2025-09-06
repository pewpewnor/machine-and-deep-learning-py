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
        # perceptron formula:
        # if a * w < threshold (which is a * w + b < 0), then output is -1, else output is 1
        hidden_outputs = np.where(
            inputs @ self.hidden_weights + self.hidden_biases < 0, -1, 1
        )
        final_output = np.where(
            hidden_outputs @ self.output_weights + self.output_bias < 0, -1, 1
        )
        return hidden_outputs, final_output

    def predict(self, inputs):
        _, final_output = self.forward(inputs)
        return final_output

    def fit(self, train_x, train_y, epochs, eta, val_x=None, val_y=None, val_every=1):
        val_countdown = 0

        for epoch in range(1, epochs + 1):
            # shuffle the training data in every epoch so that they learn outside of training data order
            indices = np.random.permutation(len(train_y))
            train_x[:] = train_x[indices]
            train_y[:] = train_y[indices]

            for inputs, label in zip(train_x, train_y):
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
            if val_x is not None and val_y is not None:
                val_countdown += 1
                if val_countdown == val_every:
                    print(
                        f"Epoch {epoch} validation accuracy: {self.score(val_x, val_y)}%"
                    )
                    val_countdown = 0

    def score(self, test_x, test_y):
        correct_predictions = sum((self.predict(test_x) == test_y).astype(int))
        return round(correct_predictions / len(test_y) * 100, 2)
