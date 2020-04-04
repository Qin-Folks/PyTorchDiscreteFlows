import torch
from torch import nn


class ClassNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ClassNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, out_dim)

    def forward(self, x_in):
        x_in = torch.relu(self.fc1(x_in))
        x_in = torch.relu(self.fc2(x_in))
        x_in = torch.relu(self.fc3(x_in))
        x_in = torch.sigmoid(self.fc4(x_in))
        return x_in
