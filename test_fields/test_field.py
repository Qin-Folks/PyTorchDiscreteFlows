import torch
from torch.distributions.bernoulli import Bernoulli
from test_fields.model import ClassNet
import torch.utils.data as utils
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
import numpy as np

epoch = 10

base_dist_ = Bernoulli(torch.tensor([0.5, 0.5]))
input_ = base_dist_.sample((50000,))
print('input_: ', input_)
bch_sz = 512
dataset_ = utils.TensorDataset(input_)
dataloader_ = DataLoader(dataset_, num_workers=1, batch_size=bch_sz, shuffle=True, drop_last=False,
                         pin_memory=True)

class_net = ClassNet(1, 1).to('cuda')
criterion_ = nn.BCELoss()
optimizer = optim.SGD(class_net.parameters(), lr=5e-5)

def train():
    epoch_num = 100
    for epoch in range(1, epoch_num+1):
        running_loss = []
        sum_loss = 0
        for ite_idx, (a_data,) in enumerate(dataloader_):
            a_data = a_data.to('cuda')

            out_ = class_net(a_data)
            loss_ = criterion_(out_, a_data)
            loss_.backward()
            optimizer.step()
            running_loss.append(loss_.item())

        print('epoch: ', epoch)
        print('running loss: ', np.mean(running_loss))
        print('out: ', out_)


# train()
