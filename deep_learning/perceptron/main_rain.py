import numpy as np
from matplotlib import pyplot as plt

from deep_learning.perceptron.my_perceptrons import Perceptrons


def main():
    perceptrons = Perceptrons(2, 5)
    training_data = [
        (np.array([95, 9]), 1),  # Very high humidity & clouds → rain
        (np.array([85, 4]), 0),  # High humidity, low clouds → no rain
        (np.array([80, 7]), 1),  # High both → rain
        (np.array([60, 2]), 0),  # Medium humidity, low clouds → no rain
        (np.array([92, 8.5]), 1),  # Very rainy conditions
        (np.array([55, 7.5]), 0),  # Medium humidity, high clouds → not always rain
        (np.array([78, 8.8]), 1),  # Rain likely
        (np.array([50, 5]), 0),  # Balanced but unclear → no rain
        (np.array([88, 3]), 0),  # High humidity, low clouds → no rain
        (np.array([93, 9.5]), 1),  # Definitely rain
        (np.array([20, 3]), 0),  # No Rain
        (np.array([80, 5]), 1),  # Rain if high humidity even if clouds are undecisive
        (np.array([30, 5]), 0),  # No rain if low humidity when clouds are undecisive
    ]

    # learning rate should be low (less than 1), so it can nudge little by little to the correct answer
    # the lower the learning rate, the more epochs are needed (more movement needed to make nudge it closer)
    # more epochs are needed for it to be properly fitted against training data
    perceptrons.fit(training_data, 10000, 0.1)

    # print all predictions based on humidity when clouds are 50
    x = list(range(0, 100))
    y = [perceptrons.predict(np.array([i, 5])) for i in x]
    plt.plot(x, y)
    plt.show()

    assert perceptrons.predict(np.array([7, 0.1])) == 0
    assert perceptrons.predict(np.array([80, 9])) == 1
    assert perceptrons.predict(np.array([90, 5])) == 1
    assert perceptrons.predict(np.array([0, 5])) == 0


if __name__ == "__main__":
    main()
