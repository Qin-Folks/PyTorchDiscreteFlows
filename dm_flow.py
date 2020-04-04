import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import torch
import itertools

import os

import disc_utils
from MADE import *
from dw_models import PreNet
from nets import MLP
from TorchDiscCode import *

n_samps = 1000
mean = np.array([5,7] )
data = np.zeros((len(mean)*n_samps,2))
samps = multivariate_normal(mean, np.eye(2)*2).rvs(n_samps)
data = samps
plt.scatter(samps[:,0], samps[:,1])
plt.show()


def one_hotter(x, depth):
    idd = np.eye(depth)
    # print(idd[0])
    res = np.zeros((x.shape[0], x.shape[1], depth))
    # print(res.shape)
    for ind in range(len(x)):
        print('x[ind]: ', x[ind])
        for j, val in enumerate(x[ind]):
            if int(val) >= depth:
                val = depth - 1
                # print(val)
            res[ind, j, :] = idd[int(val)]

    return res


batch_size, sequence_length, vocab_size = 128, 2, 30

oh = one_hotter(data, vocab_size)


def oh_sample(batch_size):
    rand = np.random.choice(np.arange(len(oh)), batch_size)
    return oh[rand, :]


samps = oh_sample(200).argmax(-1)
plt.figure(figsize=(5, 5))
plt.scatter(samps[:, 0], samps[:, 1], alpha=0.5)
plt.show()

pre_net = PreNet()

epochs = 500
lr_rate = 0.0001

for a_e in range(epochs):
    x = torch.tensor(oh_sample(batch_size)).float()
    x = x.view(x.shape[0], -1)
    pre_out = pre_net(x)
    pre_out_reshape = pre_out.view(batch_size, 2, -1)
    base_ = torch.distributions.OneHotCategorical(probs=pre_out_reshape)

