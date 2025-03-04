import numpy as np
import torch
from torch.utils.data import Dataset


# ----------------------------------- #
# Multi Grayscale Image Dataset model #
# ----------------------------------- #
class Multi_GIDataset(Dataset):

    def __init__(self, images_dir, nnshape, Ps, which_sets=['train'], symmetric=True, pmax=1., pmin=0., only_index=True, seed=0, device='cpu'):
        super(Multi_GIDataset, self).__init__()

        if isinstance(which_sets, str): which_sets = [which_sets]
        self.which_sets = which_sets
        self.datasets = {which_set: GIDataset(images_dir, nnshape, P, which_set, symmetric, pmax, pmin, only_index, seed, device) for (P, which_set) in (Ps, self.which_sets)}

    def __len__(self):
        return len(self.which_sets)

    def __getitem__(self, which_set):
        return self.datasets[which_set]



# ----------------------------- #
# Grayscale Image Dataset model #
# ----------------------------- #
class GIDataset(Dataset):

    def __init__(self, images_dir, nnshape, P, which_set='train', symmetric=True, pmax=1., pmin=0., only_index=True, seed=0, device='cpu'):
        super(GIDataset, self).__init__()

        # check inputs
        check_set = which_set in ('train', 'test', 'val')
        assert check_set, f'Invalid value for set type: {which_set}. Allowed values: "train" (default), "test", "val".'
        check_pixels = all([pmin>=0., pmax>pmin])
        assert check_pixels, f'Invalid value for pmax, pmin: {pmax}, {pmin}. The pixels maximum and minimum values should be such that: 0 <= pmin < pmax. Default: pmin=0, pmax=1.'
        check_device = any([device == 'cpu', 'cuda:' in device])
        assert check_device, f'Invalid value for device: {device}. Allowed values: "cpu" (default), "cuda:<int>".'
        
        self.images_dir = images_dir
        self.nnshape = nnshape
        self.P = P
        self.which_set = which_set
        self.symmetric = symmetric
        self.pmax = pmax
        self.pmin = pmin
        self.only_index = only_index
        self.seed = seed
        np.random.seed(self.seed)

        # load images and labels from images_dir
        with open(f"{self.images_dir}/{self.which_set}/images.npy", 'rb') as f:
            images = np.load(f)
        with open(f"{self.images_dir}/{self.which_set}/labels.npy", 'rb') as f:
            labels = np.load(f)

        # reshape images to be readable by nn model
        self.oshape = images.shape[1:]
        if self.oshape != self.nnshape:
            images = images.reshape((len(images),)+self.nnshape)

        # extract (x,y)
        self.idxs = np.random.choice(len(labels), self.P, replace=False)
        self.x = torch.tensor(images[idxs], requires_grad=False, dtype=torch.float64)
        self.y = torch.tensor(labels[idxs], requires_grad=False, dtype=torch.int32)
        del images, labels

        # symmetrize
        if symmetric:
            self.x = self._rescale(self.x, 1., -1., self.pmax, self.pmin)

        # move to device
        if ('cuda' in device) and torch.cuda.is_available():
            self.x = self.x.to(device)
            self.y = self.y.to(device)
            self.device = device
        else:
            self.device = 'cpu'

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        item = idx if self.only_index else (self.x[idx], self.y[idx], idx)
        return item

    def _rescale(self, x, M, m, b, a):
        return (M-m)*(x-a)/(b-a) + m

    def get_images(self, idxs, return_label=False):
        if isinstance(idxs, int): idxs = [idxs]
        image = self.x[idxs].reshape((len(idxs),)+self.oshape)
        if self.symmetric:
            image = self._rescale(image, self.pmax, self.pmin, 1., -1.)
        if not return_label: 
            return image
        else: 
            label = self.y[idx]
            return image, label
