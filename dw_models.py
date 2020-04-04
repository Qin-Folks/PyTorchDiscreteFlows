import torch
from torch import nn


class PreNet(nn.Module):
    def __init__(self):
        super(PreNet, self).__init__()
        self.fc1 = nn.Linear(30, 30)
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, 60)
        self.fc4 = nn.Linear(60, 60)

        self.dim = 60

    def forward(self, x_in):
        x0, x1 = x_in[:, :self.dim // 2], x_in[:, self.dim // 2:]

        x0 = torch.sigmoid(self.fc1(x0))
        x1 = torch.sigmoid(self.fc2(x1))

        x = x0 + x1

        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x
