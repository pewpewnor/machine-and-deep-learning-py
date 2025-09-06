import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from deep_learning.perceptron.my_perceptrons import Perceptrons

iris_dataset = "./data/iris_dataset/iris.data"


def main():
    df = pd.read_csv(iris_dataset, header=None)
    df = df.iloc[:100]

    train_x = df.iloc[:, [0, 2]].values
    train_y = df.iloc[:, 4].values
    train_y = np.where(train_y == "Iris-setosa", -1, 1)

    train_x_setosa = train_x[:50]
    train_x_versicolor = train_x[50:]
    train_y_setosa = train_y[:50]
    train_y_versicolor = train_y[50:]

    def show_features():
        train_x_setosa_petal = train_x_setosa[:, 0]
        train_x_setosa_sepal = train_x_setosa[:, 1]
        train_x_versicolor_petal = train_x_versicolor[:, 0]
        train_x_versicolor_sepal = train_x_versicolor[:, 1]

        plt.scatter(
            train_x_setosa_petal,
            train_x_setosa_sepal,
            color="red",
            marker="s",
            label="setosa",
        )
        plt.scatter(
            train_x_versicolor_petal,
            train_x_versicolor_sepal,
            color="blue",
            marker="x",
            label="versicolor",
        )
        plt.xlabel("petal length [cm]")
        plt.ylabel("sepal length [cm]")
        plt.legend()
        plt.show()

    ppn = Perceptrons(train_x.shape[1], 10)
    ppn.fit(train_x, train_y, 9, 0.1, val_x=train_x, val_y=train_y)

    # def show_misclassifications_over_time():
    #     plt.plot(
    #         range(1, len(ppn.misclassifications) + 1),
    #         ppn.misclassifications,
    #         marker="o",
    #     )
    #     plt.xlabel("epochs")
    #     plt.ylabel("number of misclassifications")
    #     plt.show()

    def show_decision_regions():
        # plot the decision surface
        x1_min, x1_max = train_x[:, 0].min() - 1, train_x[:, 0].max() + 1
        x2_min, x2_max = train_x[:, 1].min() - 1, train_x[:, 1].max() + 1
        resolution = 0.02
        xx1, xx2 = np.meshgrid(
            np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
        )

        # Predict row by row
        grid_points = np.array([xx1.ravel(), xx2.ravel()]).T
        Z = np.array([ppn.predict(point) for point in grid_points])
        Z = Z.reshape(xx1.shape)

        # Make contour transparent and just show colors corresponding to -1 and 1
        plt.contourf(xx1, xx2, Z, alpha=0.5, colors=["red", "blue"])

        # Plot scatter points with correct -1 / 1 labels
        plt.scatter(
            train_x[train_y == -1, 0],
            train_x[train_y == -1, 1],
            color="red",
            marker="s",
            label="setosa",
        )
        plt.scatter(
            train_x[train_y == 1, 0],
            train_x[train_y == 1, 1],
            color="blue",
            marker="x",
            label="versicolor",
        )

        plt.xlabel("petal length [cm]")
        plt.ylabel("sepal length [cm]")
        plt.legend()
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        plt.show()

    show_decision_regions()


if __name__ == "__main__":
    main()
