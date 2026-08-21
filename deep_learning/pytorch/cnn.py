from torch import nn


class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Conv2d(1, 400, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3)),
            nn.Conv2d(400, num_classes, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
        )
        self.flatten = nn.Flatten()
        self.classification_head = nn.Sequential(
            nn.LazyLinear(10),
            nn.ReLU(inplace=True),
            nn.Linear(10, num_classes),
        )

    def forward(self, x):
        x = self.feature_map(x)
        x = self.flatten(x)
        return self.classification_head(x)
