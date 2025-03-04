import numpy as np
import torch
from torch.utils.data import Dataset


# ----------------------------- #
# Random-Variable Dataset model #
# ----------------------------- #
class RVDataset(Dataset):

    def __init__(self, P, shapex, shapey, valuex='gaussian', valuey='binary', only_index=True, seed=0, device='cpu'):
        super(RVDataset, self).__init__()

        # check inputs
        check_valuex = valuex in ('gaussian', 'binary')
        assert check_valuex, f'Invalid value for valuex: {valuex}. Allowed values: "gaussian" (default), "binary".'
        check_valuey = valuey in ('gaussian', 'binary')
        assert check_valuey, f'Invalid value for valuey: {valuey}. Allowed values: "gaussian", "binary" (default).'
        check_device = any([device == 'cpu', 'cuda:' in device])
        assert check_device, f'Invalid value for device: {device}. Allowed values: "cpu" (default), "cuda:<int>".'

        self.P = P
        self.shapex = shapex
        self.shapey = shapey
        self.valuex = valuex
        self.valuey = valuey
        self.only_index = only_index
        self.seed = seed
        np.random.seed(self.seed)

        # generate (x,y)
        if self.valuex == 'gaussian':
            x = torch.tensor(
                    np.random.randn(self.P, *self.shapex),
                    requires_grad=False,
                    dtype=torch.float64
            )
        else:
            x = torch.tensor(
                    np.random.choice((-1, 1), (P, *self.shapex)),
                    requires_grad=False,
                    dtype=torch.int32
            )
        if self.valuey == 'gaussian':
            y = torch.tensor(
                    np.random.randn(self.P, *self.shapey),
                    requires_grad=False,
                    dtype=torch.float64
            )
        else:
            y = torch.tensor(
                    np.random.choice((-1, 1), (P, *self.shapey)),
                    requires_grad=False,
                    dtype=torch.int32
            )
        if ('cuda' in device) and torch.cuda.is_available():
            x = x.to(device)
            y = y.to(device)
            self.device = device
        else:
            self.device = 'cpu'
        self.x = x
        self.y = y

    def __len__(self):
        return self.P

    def __getitem__(self, idx):
        item = idx if self.only_index else (self.x[idx], self.y[idx], idx)
        return item
