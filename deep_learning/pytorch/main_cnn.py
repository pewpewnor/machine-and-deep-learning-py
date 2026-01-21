import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_data = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=ToTensor()
)
test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=ToTensor()
)

train_loader = DataLoader(train_data, batch_size=10)
test_loader = DataLoader(test_data, batch_size=10)

from deep_learning.pytorch.cnn import CNN

model = CNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters())


def train_one_epoch():
    running_loss = 0
    last_loss = 0

    for i, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(imgs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999 or i == 0:
            last_loss = running_loss / 1000  # loss per batch
            print("  batch {} loss: {}".format(i + 1, last_loss))
            running_loss = 0.0

    return last_loss


def test():
    model.eval()

    all_correct = 0
    all_total = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            output = model(imgs)
            pred = torch.argmax(output, dim=1)
            assert len(pred) == len(labels)

            correct = (pred == labels).sum().item()
            all_correct += correct
            all_total += len(pred)

    print(f"Correct: {all_correct} / {all_total}")


train_one_epoch()

start = time.time()
test()
end = time.time()
print(end - start, "seconds")
