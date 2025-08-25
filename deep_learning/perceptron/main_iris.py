import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from deep_learning.perceptron.my_perceptrons import Perceptrons

iris_dataset = "./data/iris_dataset/iris.data"


def main():
    df = pd.read_csv(iris_dataset, header=None)
    df = df.iloc[:100]

    labels = df.iloc[:, 4].values
    labels = np.where(labels == "Iris-setosa", 0, 1)
    features = df.iloc[:, [0, 2]].values

    labels_setosa = labels[:50]
    labels_versicolor = labels[50:]
    features_setosa = features[:50]
    features_versicolor = features[50:]

    def show_features():
        features_setosa_petal = features_setosa[:, 0]
        features_setosa_sepal = features_setosa[:, 1]
        features_versicolor_petal = features_versicolor[:, 0]
        features_versicolor_sepal = features_versicolor[:, 1]

        plt.scatter(
            features_setosa_petal,
            features_setosa_sepal,
            color="red",
            marker="s",
            label="setosa",
        )
        plt.scatter(
            features_versicolor_petal,
            features_versicolor_sepal,
            color="blue",
            marker="x",
            label="versicolor",
        )
        plt.xlabel("petal length [cm]")
        plt.ylabel("sepal length [cm]")
        plt.legend()
        plt.show()

    training_data = [(f, label) for f, label in zip(features, labels)]

    ppn = Perceptrons(features.shape[1], 10)
    ppn.fit(training_data, 100, 0.1)

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
        colors = ("red", "blue", "lightgreen", "gray", "cyan")
        cmap = ListedColormap(colors[: len(np.unique(labels))])

        # plot the decision surface
        x1_min, x1_max = features[:, 0].min() - 1, features[:, 0].max() + 1
        x2_min, x2_max = features[:, 1].min() - 1, features[:, 1].max() + 1
        resolution = 0.02
        xx1, xx2 = np.meshgrid(
            np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
        )
        Z = ppn.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        show_features()

    show_decision_regions()


if __name__ == "__main__":
    main()
