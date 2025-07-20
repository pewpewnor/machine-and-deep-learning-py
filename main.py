import math
from os.path import join

import matplotlib.pyplot as plt
import numpy as np

from mnist_loader import LabeledImage, MnistDataloader
from neural_network import NeuralNetwork

input_dir_path = "./input"
training_images_filepath = join(input_dir_path, "train-images.idx3-ubyte")
training_labels_filepath = join(input_dir_path, "train-labels.idx1-ubyte")
test_images_filepath = join(input_dir_path, "t10k-images.idx3-ubyte")
test_labels_filepath = join(input_dir_path, "t10k-labels.idx1-ubyte")


# Load MINST dataset
mnist_dataloader = MnistDataloader(
    training_images_filepath,
    training_labels_filepath,
    test_images_filepath,
    test_labels_filepath,
)
labeled_training_images, labeled_test_images = mnist_dataloader.load_data()


def test():
    nn = NeuralNetwork([784, 16, 10])
    for image, label in labeled_training_images[:10]:
        pass
        # p = nn.feedforward(image)

    correct = 0
    for image, label in labeled_test_images[:100]:
        p = nn.feedforward(image)
        guess = np.argmax(p) + 1
        if guess == label:
            correct += 1

    print("accuracy:", correct, "%")


test()


def show_images(labeled_images, cols, output_path="output.png"):
    rows = math.ceil(len(labeled_images) / cols)
    plt.figure(figsize=(5 * cols, 4 * rows))

    for index, labeled_image in enumerate(labeled_images, 1):
        plt.subplot(rows, cols, index)
        plt.imshow(labeled_image.image, cmap="Greys")
        if labeled_image.label:
            plt.title(labeled_image.label, fontsize=12)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()


def show_random_training_and_test_images():
    labeled_images_2_show = []
    for r in np.random.randint(1, 10000, 10):
        train_image, train_label = labeled_training_images[r]
        labeled_images_2_show.append(
            LabeledImage(
                train_image,
                f"training image [{r}] = {train_label}",
            )
        )

    for r in np.random.randint(1, 10000, 5):
        test_image, test_label = labeled_test_images[r]
        labeled_images_2_show.append(
            LabeledImage(
                test_image,
                f"test image [{r}] = {test_label}",
            )
        )

    show_images(labeled_images_2_show, 5)


# show_random_training_and_test_images()
