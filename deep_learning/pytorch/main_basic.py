import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=ToTensor()
)
test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=ToTensor()
)

train_loader = DataLoader(training_data, batch_size=10)
test_loader = DataLoader(test_data, batch_size=10)


def display_images():
    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }
    plt.figure(figsize=(15, 4))

    for images, labels in train_loader:
        for i in range(10):
            plt.subplot(1, 10, i + 1)
            plt.imshow(images[i].squeeze(), cmap="gray")
            plt.title(labels_map[labels[i].item()])
            plt.axis("off")
        break

    plt.show()


display_images()
