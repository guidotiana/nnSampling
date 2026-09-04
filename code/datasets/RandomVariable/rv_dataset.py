import numpy as np
import torch
from torch.utils.data import Dataset


class RVDataset(Dataset):

	def __init__(self, P, shapex, shapey, rho='gaussian', seed=0, device='cpu'):
		super(RVDataset, self).__init__()

		# check inputs
		assert rho in ('gaussian', 'binary'), f'RVDataset.__init__(): invalid value for <rho> ({rho}). Allowed values: "gaussian" (default), "binary".'
		assert any([device=='cpu', 'cuda:' in device]), f'RVDataset.__init__(): invalid value for <device> ({device}). Allowed values: "cpu" (default), "cuda:<int>".'

		self.P = P
		self.shapex = shapex
		self.shapey = shapey
		self.rho = rho
		self.seed = seed

		np.random.seed(self.seed)

		# generate (x,y)
		if self.rho == 'gaussian':
			x = torch.tensor(
				np.random.randn(self.P, *self.shapex),
				requires_grad=False,
			)
		else:
			x = torch.tensor(
				np.random.choice((-1, 1), (P, *self.shapex)),
				requires_grad=False,
				dtype=torch.int32
			)
		y = torch.tensor(
			np.random.choice((-1, 1), (P, *self.shapey)),
			requires_grad=False,
			dtype=torch.int32
		)

		if ('cuda' in device) and torch.cuda.is_available():
			x = x.to(device)
			y = y.to(device)

		self.x = x
		self.y = y

	def __len__(self):
		return self.P

	def __getitem__(self, idx):
		return self.x[idx], self.y[idx], idx

	def to(self, device):
		device = device if isinstance(device, torch.device) else torch.device(device)
		if ("cpu" in device.type) or (("cuda" in device.type) and torch.cuda.is_available()):
			self.x = self.x.to(device)
			self.y = self.y.to(device)
