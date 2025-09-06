import numpy as np
from matplotlib import pyplot as plt

from deep_learning.perceptron.my_perceptrons import Perceptrons
from utils.feature_engineering.standard_scaler import StandardScaler


def main():
    perceptrons = Perceptrons(2, 3)
    train_data = [
        (np.array([95, 9]), 1),  # Very high humidity & clouds → rain
        (np.array([85, 4]), -1),  # High humidity, low clouds → no rain
        (np.array([80, 7]), 1),  # High both → rain
        (np.array([60, 2]), -1),  # Medium humidity, low clouds → no rain
        (np.array([92, 8.5]), 1),  # Very rainy conditions
        (np.array([55, 7.5]), -1),  # Medium humidity, high clouds → not always rain
        (np.array([78, 8.8]), 1),  # Rain likely
        (np.array([50, 5]), -1),  # Balanced but unclear → no rain
        (np.array([88, 3]), -1),  # High humidity, low clouds → no rain
        (np.array([93, 9.5]), 1),  # Definitely rain
        (np.array([20, 3]), -1),  # No Rain
        (np.array([70, 5]), 1),  # Rain if high humidity even if clouds are undecisive
        (np.array([90, 5]), 1),  # Rain if high humidity even if clouds are undecisive
        (np.array([100, 5]), 1),  # Rain if high humidity even if clouds are undecisive
        (np.array([30, 5]), -1),  # No rain if low humidity when clouds are undecisive
        (np.array([20, 5]), -1),  # No rain if low humidity when clouds are undecisive
    ]
    scaler = StandardScaler()
    train_x = scaler.fit_transform([x for x, _ in train_data])
    train_y = np.array([y for _, y in train_data])

    # learning rate should be low (less than 1), so it can nudge little by little to the correct answer
    # the lower the learning rate, the more epochs are needed (more movement needed to make nudge it closer)
    # more epochs are needed for it to be properly fitted against training data
    perceptrons.fit(
        train_x, train_y, 200, 0.01, val_x=train_x, val_y=train_y, val_every=10
    )

    # print all predictions based on humidity when clouds are 50
    plot_x = list(range(0, 100))
    plot_y = [perceptrons.predict(scaler.transform(np.array([i, 5]))) for i in plot_x]
    plt.plot(plot_x, plot_y)
    plt.show()

    test_data = [
        (np.array([7, 0.1]), -1),
        (np.array([80, 9]), 1),
        (np.array([90, 5]), 1),
        (np.array([80, 5]), 1),
        (np.array([0, 5]), -1),
        (np.array([40, 5]), -1),
    ]
    test_x = scaler.transform([x for x, _ in test_data])
    test_y = [y for _, y in test_data]
    for x, y in zip(test_x, test_y):
        assert perceptrons.predict(x) == y


if __name__ == "__main__":
    main()
