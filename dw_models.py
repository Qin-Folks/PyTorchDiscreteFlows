import torch
from torch import nn


class PreNet(nn.Module):
    def __init__(self):
        super(PreNet, self).__init__()
        self.fc1_1 = nn.Linear(30, 30)
        self.fc1_2 = nn.Linear(30, 30)
        self.fc2 = nn.Linear(30, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 60)

        self.dim = 60

    def forward(self, x_in):
        x0, x1 = x_in[:, :self.dim // 2], x_in[:, self.dim // 2:]

        x0 = torch.relu(self.fc1_1(x0))
        x1 = torch.relu(self.fc1_2(x1))

        x_in = x0 + x1

        x_in = torch.relu(self.fc2(x_in))
        x_in = torch.relu(self.fc3(x_in))
        x_in = self.fc4(x_in)
        x0, x1 = x_in[:, :self.dim // 2], x_in[:, self.dim // 2:]

        tmp = 0.02
        x0 = torch.softmax(x0/tmp, dim=-1)
        x1 = torch.softmax(x1/tmp, dim=-1)
        x = torch.stack((x0, x1), dim=1)
        return x
