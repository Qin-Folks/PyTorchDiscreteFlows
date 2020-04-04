import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import torch
import itertools

import os

import disc_utils
from MADE import *
from dw_models import PreNet, RealNVP
from nets import MLP
from TorchDiscCode import *
from torch import distributions


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

mask_1 = torch.from_numpy(np.zeros(30).astype(np.float32))
mask_2 = torch.from_numpy(np.ones(30).astype(np.float32))
mask = torch.cat((mask_1, mask_2))
real_nvp = RealNVP(60, 60, 256, mask, 8)

prior_z = distributions.MultivariateNormal(torch.zeros(60), torch.eye(60))

epochs = 600
lr_rate = 0.001
print_loss_every = 20

optimizer = torch.optim.Adam(list(pre_net.parameters()) + list(real_nvp.parameters()), lr=lr_rate)

pre_net.train()
real_nvp.train()
for a_e in range(epochs):
    optimizer.zero_grad()
    x = torch.tensor(oh_sample(batch_size)).float()
    x_reshape = x.view(x.shape[0], -1)
    pre_out = pre_net(x_reshape)
    log_prob_sum = 0

    for pre_idx, a_pre_out in enumerate(pre_out):
        a_base = torch.distributions.OneHotCategorical(probs=a_pre_out)
        a_log_prob = a_base.log_prob(x[pre_idx])
        log_prob_sum += torch.sum(a_log_prob)

    pre_out_flatten = pre_out.view(x.shape[0], -1)
    z, log_det_j_sum = real_nvp(pre_out_flatten)

    flow_loss = -(prior_z.log_prob(z) + log_det_j_sum).mean()
    pre_loss = -log_prob_sum
    loss = flow_loss + pre_loss
    loss.backward()
    optimizer.step()
    if a_e % print_loss_every == 0:
        print('epoch:', a_e, 'pre loss:', pre_loss.item(), 'flow loss: ', flow_loss.item(), 'loss: ', loss.item())

    # base_ = torch.distributions.OneHotCategorical(probs=pre_out_reshape)
    # base_log_prob = base_.log_prob(x)

# 测试性能
print('Concluded')
sample_sz = 20
pre_net.eval()
real_nvp.eval()
x = torch.tensor(oh_sample(sample_sz)).float()
x_ = x.detach().numpy().argmax(-1)
plt.scatter(x_[:, 0], x_[:, 1], label='Input x', alpha=0.3, marker='x')
pre_out = pre_net(x.view(x.shape[0], -1)).view(x.shape[0], -1)
z, log_det_j_sum = real_nvp(pre_out)
x_back = real_nvp.backward(z)
x_back[x_back < 0] = 0.0
x_back = x_back.view(sample_sz, 2, 30)
samples = []
equaled = 0
non_equaled = 0
for x_back_idx, a_x_back in enumerate(x_back):
    x_idxed = x_[x_back_idx]
    non_neg_x = torch.relu(a_x_back)
    a_base = torch.distributions.OneHotCategorical(probs=non_neg_x)
    a_sample = a_base.sample()
    print('x idxed: ', x_idxed, 'a sample: ', a_sample.argmax(-1).numpy())
    if np.equal(x_idxed, a_sample.argmax(-1).numpy()).all():
        equaled += 1
    else:
        non_equaled +=1
    samples.append(a_sample)
samples = torch.stack(samples, dim=0)
samples_ = samples.detach().numpy().argmax(-1)
plt.scatter(samples_[:, 0], samples_[:, 1], label='Input x', alpha=0.3, color='r')
plt.show()
print('equaled: ', equaled, 'non equaled: ', non_equaled)
