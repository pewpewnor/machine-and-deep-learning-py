from torch import nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Conv2d(200, 200, kernel_size=(3, 3)),
        )
        self.flatten = nn.Flatten()
        self.classification_head = nn.Sequential(
            nn.Linear(28 * 28, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        flat = self.flatten(x)
        probabilities = self.classification_head(flat)
        return probabilities
