import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import torch
import itertools

import os

import disc_utils
from MADE import *
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


samps = oh_sample(10000).argmax(-1)
print('samps: ', samps[:10])
plt.figure(figsize=(5,5))
plt.scatter(samps[:,0], samps[:,1])

num_flows = 6 # number of flow steps. This is different to the number of layers used inside each flow
nh = 32 # number of hidden units per layer
vector_length = sequence_length*vocab_size
temperature = 0.1 # used for the straight-through gradient estimator. Value taken from the paper
disc_layer_type = 'bipartite' #'autoreg'
flows = []

for i in range(num_flows):
    if disc_layer_type == 'autoreg':
        layer = MADE(vocab_size, [nh, nh, nh], vocab_size,
                     num_masks=1, natural_ordering=True)
        # if want to also learn the scale:
        # put MADE(vocab_size, [nh, nh, nh], 2*vocab_size, num_masks=1, natural_ordering=True)

        # if natural ordering is false then this only works for up to 4 layers!!!!
        # TODO: fix this bug.

        disc_layer = DiscreteAutoregressiveFlow(layer, temperature, vocab_size)

    elif disc_layer_type == 'bipartite':
        layer = MLP(vector_length // 2, vector_length // 2, nh)
        # to get the scale also, set MLP(vector_length//2, vector_length, nh)
        # MLP defaults to the following architecture for each individual flow:
        '''
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )
        '''

        disc_layer = DiscreteBipartiteFlow(layer, i % 2, temperature, vocab_size, vector_length)
        # i%2 flips the parity of the masking. It splits the vector in half and alternates
        # each flow between changing the first half or the second.
    flows.append(disc_layer)

model = DiscreteAutoFlowModel(flows)

n_samps = 10000
data_samps = oh_sample(n_samps).argmax(-1)
mod_data_samps = data_samps+15
plt.scatter(data_samps[:,0], data_samps[:,1], label = 'original data')
plt.scatter(mod_data_samps[:,0], mod_data_samps[:,1], label = 'shifted data')
plt.legend()
plt.show()

import collections
import pandas as pd
mod_data_dim0 = collections.Counter(mod_data_samps[:,0])
mod_data_dim1 = collections.Counter(mod_data_samps[:,1])
dim0_probs = np.zeros((vocab_size))
dim1_probs = np.zeros((vocab_size))
for k, v in mod_data_dim0.items():
    dim0_probs[k] = v/n_samps
for k, v in mod_data_dim1.items():
    dim1_probs[k] = (v/n_samps)

dim0_probs += 0.000001
dim1_probs += 0.000001

# need to renormalize again...
dim0_probs = dim0_probs / np.sum(dim0_probs)
dim1_probs = dim1_probs / np.sum(dim1_probs)

mod_data_probs = np.vstack([dim0_probs, dim1_probs])

base = torch.distributions.OneHotCategorical(probs = torch.tensor(mod_data_probs).float() )
samps = base.sample((10000,)).argmax(-1)
plt.scatter(samps[:,0], samps[:,1], label = 'shifted data')
plt.show()

epochs = 500
learning_rate = 0.0001
print_loss_every = 20
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
losses = []
weights = []
base_log_probs = torch.log(base.probs)
model.train()
for e in range(epochs):
    x = torch.tensor(oh_sample(batch_size)).float()

    if disc_layer_type == 'bipartite':
        x = x.view(x.shape[0], -1)  # flattening vector

    optimizer.zero_grad()
    zs = model.forward(x)

    if disc_layer_type == 'bipartite':
        zs = zs.view(batch_size, 2, -1)  # flattening vector

    logprob = zs * base_log_probs.float()
    loss = -torch.sum(logprob) / batch_size

    loss.backward()
    optimizer.step()

    losses.append(loss.detach())

    if e % print_loss_every == 0:
        print('epoch:', e, 'loss:', loss.item())

plt.plot(losses)

model.eval()
x = torch.tensor(oh_sample(batch_size)).float()
if disc_layer_type == 'bipartite':
    x = x.view(batch_size, -1)
zs = model.forward(x)
z = zs
if disc_layer_type == 'bipartite':
    z = z.view(batch_size, 2, -1)
    x = x.view(batch_size, 2, -1)

x = x.detach().numpy().argmax(-1)
z = z.detach().numpy().argmax(-1)
p = base.sample((batch_size,)).argmax(-1)
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.scatter(p[:,0], p[:,1], c='g', s=5)
plt.scatter(z[:,0], z[:,1], c='r', s=5)
plt.scatter(x[:,0], x[:,1], c='b', s=5)
plt.legend(['prior', 'x->z', 'data'])
plt.axis('scaled')
plt.title('x -> z')
plt.xlim([0,vocab_size])
plt.ylim([0,vocab_size])

if disc_layer_type == 'bipartite':
    z = model.reverse(base.sample((batch_size,)).float().view(batch_size, -1))
    z = z.view(batch_size, 2, -1)
else:
    z = model.reverse(base.sample((batch_size,)).float())
z = z.detach().numpy().argmax(-1)
plt.subplot(122)
plt.scatter(x[:,0], x[:,1], c='b', s=5, alpha=0.5)
plt.scatter(z[:,0], z[:,1], c='r', s=5, alpha=0.3)
plt.legend(['data', 'z->x'])
plt.axis('scaled')
plt.title('z -> x')
plt.xlim([0,vocab_size])
plt.ylim([0,vocab_size])
plt.gcf().savefig('DemoResult.png', dpi=250)