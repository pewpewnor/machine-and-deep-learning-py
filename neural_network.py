import numpy as np


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    """
    we don't actually have to use softmax to see what digit did the neural network guess
    to do that, we can just grab the highest number out of all the output nodes
    softmax is simply math to convert an array of negative and positive values to each item being 0 to 1
    with highest value being closer to 1, and lowest value being closer to 0
    also, the sum of the new array would always be 1
    so we can interpret the result of softmax as probabilities where 0 is 0% and 1 is 100%
    """
    # stabilizing to prevent integer overflow since the exponent of e could be so big?
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)


class NeuralNetwork:
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        # simple biases and weights random initialization
        # sizes[1:] cuz excluding size for input layer
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # sizes[:-1] for x cuz we don't need to know the number of output nodes for weights
        # sizes[1:] for y cuz excluding size for input layer
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """
        a (activation, how activated it is) is array of inputs (for the layer)
        z is array of weighted sums (per layer)
        w is array of weights (per layer)
        b is array of biases (per layer)
        """
        a = np.array(a).reshape(-1, 1)
        self.prevz = []

        # for hidden layers
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(w, a) + b
            a = relu(z)
            self.prevz.append(z)

        """
        for output layer, don't use relu or other activation functions
        since e.g. relu will force negative numbers to 0
        which means that when we softmax the result of that relu, -5 will have the same probability value as -2
        but -5 should instead have a lower probability value than -2
        """
        w, b = self.weights[-1], self.biases[-1]
        z = np.dot(w, a) + b
        self.prevz.append(z)
        return softmax(z)

    def __str__(self):
        return f"""biases:
    {self.biases}
weights:
    {self.weights}
"""


def assert_nn():
    nn = NeuralNetwork([4, 3, 3, 2])
    print(nn)
    probabilities = nn.feedforward([1, 2, 3, 4])
    print(probabilities)
    print("highest:", np.max(probabilities), "which is index", np.argmax(probabilities))
    assert np.isclose(
        np.sum(probabilities), 1, rtol=1e-9, atol=1e-12
    )  # sum of probabilities must basically be 1 (100%)
    print()
    print()
    print("all asserts passed.")


if __name__ == "__main__":
    assert_nn()
