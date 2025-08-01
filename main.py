import math
from os.path import join

import matplotlib.pyplot as plt
import numpy as np

from mnist_loader import MnistDataloader
from neural_network import NeuralNetwork

input_dir_path = "./input"
training_images_filepath = join(input_dir_path, "train-images.idx3-ubyte")
training_labels_filepath = join(input_dir_path, "train-labels.idx1-ubyte")
test_images_filepath = join(input_dir_path, "t10k-images.idx3-ubyte")
test_labels_filepath = join(input_dir_path, "t10k-labels.idx1-ubyte")


def main():
    mnist_dataloader = MnistDataloader(
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
    )
    training_data, test_data = mnist_dataloader.load_data(
        shape="flat", one_hot_encode=True
    )

    validation_data = training_data[50000:]
    training_data = training_data[:50000]
    nn = NeuralNetwork([784, 100, 10])
    print("Stochastic gradient descent...")
    nn.SGD(training_data, 30, 10, 3.0, test_data=validation_data)
    print("Evaluating model accuracy...")
    print(f"Model has {nn.evaluate(test_data)}% accuracy")


main()


def show_images(data, cols, output_path="output.png"):
    rows = math.ceil(len(data) / cols)
    plt.figure(figsize=(5 * cols, 4 * rows))

    for index, (image, label) in enumerate(data, 1):
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap="Greys")
        if label:
            plt.title(label, fontsize=12)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()


def show_random_training_and_test_images(training_data, test_data):
    data_to_show = []
    for r in np.random.randint(1, 10000, 10):
        train_image, train_label = training_data[r]
        data_to_show.append((train_image, f"training image [{r}] = {train_label}"))

    for r in np.random.randint(1, 10000, 5):
        test_image, test_label = test_data[r]
        data_to_show.append((test_image, f"test image [{r}] = {test_label}"))

    show_images(data_to_show, 5)
