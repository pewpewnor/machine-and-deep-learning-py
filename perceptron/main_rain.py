import numpy as np
from matplotlib import pyplot as plt

from perceptron.my_perceptron import Perceptrons


def main():
    perceptrons = Perceptrons(2, 5)
    training_data = [
        (np.array([95, 90]), 1),  # Very high humidity & clouds → rain
        (np.array([85, 40]), 0),  # High humidity, low clouds → no rain
        (np.array([80, 70]), 1),  # High both → rain
        (np.array([60, 20]), 0),  # Medium humidity, low clouds → no rain
        (np.array([92, 85]), 1),  # Very rainy conditions
        (np.array([55, 75]), 0),  # Medium humidity, high clouds → not always rain
        (np.array([78, 88]), 1),  # Rain likely
        (np.array([50, 50]), 0),  # Balanced but unclear → no rain
        (np.array([88, 30]), 0),  # High humidity, low clouds → no rain
        (np.array([93, 95]), 1),  # Definitely rain
        (np.array([20, 30]), 0),  # No Rain
        (np.array([80, 50]), 1),  # Rain if high humidity even if clouds are undecisive
        (np.array([30, 50]), 0),  # No rain if low humidity when clouds are undecisive
    ]

    # learning rate should be low (less than 1), so it can nudge little by little to the correct answer
    # the lower the learning rate, the more epochs are needed (more movement needed to make nudge it closer)
    # more epochs are needed for it to be properly fitted against training data
    perceptrons.fit(training_data, 10000, 0.1)

    print(perceptrons.predict(np.array([7, 1])) == 0)

    # print all predictions based on humidity when clouds are 50
    x = list(range(0, 100))
    y = [perceptrons.predict(np.array([i, 50])) for i in x]
    plt.plot(x, y)
    plt.show()

    assert perceptrons.predict(np.array([7, 1])) == 0
    assert perceptrons.predict(np.array([80, 90])) == 1
    assert perceptrons.predict(np.array([90, 50])) == 1
    assert perceptrons.predict(np.array([0, 50])) == 0


if __name__ == "__main__":
    main()
