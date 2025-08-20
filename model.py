import torch
import torch.nn as nn

class ClassificationCNN(nn.Module):
    def __init__(self):
        super(ClassificationCNN,self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 4, 5)
        self.conv2 = nn.Conv2d(4, 8, 5)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 11)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


