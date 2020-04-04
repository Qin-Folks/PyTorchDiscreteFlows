import torch
import numpy as np
import torch.nn.functional as F
from torchvision.utils import save_image


def idx_to_one_hot(idx, dig_to_use):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    arg_idx = torch.tensor(list([dig_to_use.index(x) for x in idx]))
    n = len(dig_to_use)
    assert torch.max(arg_idx).item() < n
    if arg_idx.dim() == 0:
        arg_idx = arg_idx.unsqueeze(0)
    if arg_idx.dim() == 1:
        arg_idx = arg_idx.unsqueeze(1)

    onehot = torch.zeros(arg_idx.size(0), n)
    onehot = onehot.to(device)
    arg_idx = arg_idx.to(device)
    onehot.scatter_(1, arg_idx, 1)
    return onehot


def idx_to_multi_hot(idx, num_obj, dig_to_use, loc_sensitive=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n = len(dig_to_use)

    multi_hot = None
    arg_idx = []
    for an_idx in idx:
        arg_idx.append(list([dig_to_use.index(x) for x in an_idx]))
    arg_idx = torch.tensor(np.array(arg_idx))

    assert torch.max(arg_idx).item() < n
    for i in range(num_obj):
        an_idx = arg_idx[:, i].unsqueeze(-1)
        onehot = torch.zeros(an_idx.size(0), n)
        onehot = onehot.to(device)
        an_idx = an_idx.to(device)
        onehot.scatter_(1, an_idx, 1)
        if multi_hot is None:
            multi_hot = onehot
        else:
            if loc_sensitive:
                multi_hot = torch.cat((multi_hot, onehot), dim=-1)
            else:
                multi_hot += onehot
    return multi_hot




