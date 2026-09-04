import numpy as np
import torch
from torchvision import datasets, transforms

def rescale(x, M, m, b, a):
	return (M-m)*(x-a)/(b-a) + m


download_d = "./download"
data_d = "."

class_to_list = {
	'cat': [3, -1],
	'dog': [5, +1],
}

tfm = transforms.Compose([
	transforms.ToPILImage(),
	transforms.Grayscale(num_output_channels=1),
	transforms.ToTensor(),
])
train_ds = datasets.CIFAR10(root=download_d, train=True, download=True, transform=tfm)
test_ds = datasets.CIFAR10(root=download_d, train=False, download=True, transform=tfm)

torch.manual_seed(0)
for ds, label in zip([train_ds, test_ds], ["train", "test"]):

	X = []
	Y = []

	for name,(idx,value) in class_to_list.items():
		mask = torch.tensor(ds.targets) == idx
		X += ds.data[mask].tolist()
		Y += [value]*mask.sum()

	X = torch.tensor(X, dtype=torch.float32).transpose(1,3)
	X = torch.tensor(
		[ds.transform(x).tolist() for x in X]
	)
	X = X.squeeze(1)

	X = rescale(
		X,
		M=1., m=-1.,
		a=1., b=0.,
	)
	Y = torch.tensor(Y, dtype=torch.int32)

	idxs = torch.randperm(len(X))
	X = X[idxs]
	Y = Y[idxs]

	print(f"Label <{label}>:")
	print(f" X -> {X.shape} (min={X.min().item():.1f}, max={X.max().item():.1f})")
	print(f" Y -> {Y.shape}\n")
	torch.save((X,Y), f"{data_d}/sgcifar2_{label}.pt")
