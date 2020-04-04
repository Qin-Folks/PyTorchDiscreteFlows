import torch
from torch import nn
from torch.nn import functional as F


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

        tmp = 1
        # x0_max, _ = torch.max(x0, dim=-1, keepdim=True)
        # x0_min, _ = torch.min(x0, dim=-1, keepdim=True)
        # print('(x0 - x0_min): ', (x0 - x0_min).shape)
        # print('(x0_max - x0_min): ', (x0_max - x0_min).shape)
        # print('(x1_max - x1_min).repeat(1, x1.shape[1]): ', (x0_max - x0_min).repeat(1, x0.shape[1]).shape)
        # x0 = (x0 - x0_min) / (x0_max - x0_min).repeat(1, x0.shape[1])
        #
        # x1_max, _ = torch.max(x1, dim=-1)
        # x1_min, _ = torch.min(x1, dim=-1)
        # x1 = (x1 - x1_min) / (x1_max - x1_min).repeat(1, x1.shape[1])

        x0 = torch.softmax(x0/tmp, dim=-1)
        x1 = torch.softmax(x1/tmp, dim=-1)
        x = torch.stack((x0, x1), dim=1)
        return x


class CouplingLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, mask):
        super().__init__()
        self.s_fc1 = nn.Linear(input_dim, hid_dim)
        self.s_fc2 = nn.Linear(hid_dim, hid_dim)
        self.s_fc3 = nn.Linear(hid_dim, output_dim)
        self.t_fc1 = nn.Linear(input_dim, hid_dim)
        self.t_fc2 = nn.Linear(hid_dim, hid_dim)
        self.t_fc3 = nn.Linear(hid_dim, output_dim)
        self.mask = mask

    def forward(self, x):
        x_m = x * self.mask
        s_out = torch.tanh(self.s_fc3(F.relu(self.s_fc2(F.relu(self.s_fc1(x_m))))))
        t_out = self.t_fc3(F.relu(self.t_fc2(F.relu(self.t_fc1(x_m)))))
        y = x_m + (1 - self.mask) * (x * torch.exp(s_out) + t_out)
        log_det_jacobian = s_out.sum(dim=1)
        return y, log_det_jacobian

    def backward(self, y):
        y_m = y * self.mask
        s_out = torch.tanh(self.s_fc3(F.relu(self.s_fc2(F.relu(self.s_fc1(y_m))))))
        t_out = self.t_fc3(F.relu(self.t_fc2(F.relu(self.t_fc1(y_m)))))
        x = y_m + (1 - self.mask) * (y - t_out) * torch.exp(-s_out)
        return x


class RealNVP(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, mask, n_layers=6):
        super().__init__()
        assert n_layers >= 2, 'num of coupling layers should be greater or equal to 2'

        self.modules = []
        self.modules.append(CouplingLayer(input_dim, output_dim, hid_dim, mask))
        for _ in range(n_layers - 2):
            mask = 1 - mask
            self.modules.append(CouplingLayer(input_dim, output_dim, hid_dim, mask))
        self.modules.append(CouplingLayer(input_dim, output_dim, hid_dim, 1 - mask))
        self.module_list = nn.ModuleList(self.modules)

    def forward(self, x):
        ldj_sum = 0  # sum of log determinant of jacobian
        for module in self.module_list:
            x, ldj = module(x)
            ldj_sum += ldj
        return x, ldj_sum

    def backward(self, z):
        for module in reversed(self.module_list):
            z = module.backward(z)
        return z