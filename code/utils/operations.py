import torch
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from time import process_time as ptime


# Find optimal permutation between two weights vectors
def find_permutation(wi, wj, check=True):
	if check:
		assert len(wi.keys())==len(wj.keys()), 'compute_q(): invalid inputs weights vectors. Dictionary lengths must coincide!'
		assert all([wi_layer in wj for wi_layer in wi]), 'compute_q(): invalid inputs weights vectors. Dictionary keys must coincide!'

	layers = [layer for layer in wi if 'bias' not in layer]
	for ilayer, layer in enumerate(layers):
		bias = f'{layer.rstrip("weight")}.bias'
		if bias in wi:
			wi[layer] = torch.concat(
				(wi[layer], wi[bias].unsqueeze(-1)),
				axis=-1
			)
			wj[layer] = torch.concat(
				(wj[layer], wj[bias].unsqueeze(-1)),
				axis=-1
			)

		wi[layer] = wi[layer].squeeze()
		wj[layer] = wj[layer].squeeze()
		assert wi[layer].ndim == 2, f'compute_q(): unexpected shape from layer {layer}: {tuple(wi[layer].shape)}. Expected (squeezed) dimensions: 2.'
		K, N_k = wi[layer].shape

		x, y = np.arange(K), np.arange(K)
		X, Y = np.meshgrid(x, y)

		cost = (wi[layer][Y.reshape(-1)]*wj[layer][X.reshape(-1)]).sum(axis=-1)
		cost = cost.reshape(K, K)
		row_ind, col_ind = linear_sum_assignment(cost.detach().numpy(), maximize=True)

		if ilayer < len(layers)-1:
			next_layer = layers[ilayer+1]
			wi[next_layer] = wi[next_layer][..., row_ind]
			wj[next_layer] = wj[next_layer][..., col_ind]

	return wi, wj

# Calculate squared modulus of a weight vector
def compute_mod2(w):
	mod2 = 0.
	for name in w:
		mod2 += (w[name]**2).sum()
	return mod2

# Calculate modulus of a weight vector
def compute_mod(w):
	mod2 = compute_mod2(w)
	return torch.sqrt(mod2)

# Calculate squared distance between two weight vectors
def compute_d2(wi, wj):
	d2 = 0.
	for name in wi:
		d2 += ((wi[name]-wj[name])**2.).sum()
	return d2

# Calculate distance between two weight vectors
def compute_d(wi, wj):
	d2 = compute_d2(wi, wj)
	return torch.sqrt(d2)

# Calculate similarity between two weight vectors
def compute_q(wi, wj, mods=None):
	if mods is None:
		modi = compute_mod(wi)
		modj = compute_mod(wj)
	else:
		modi, modj = mods
	dotprod = 0.
	for name in wi:
		dotprod += (wi[name]*wj[name]).sum()
	q = dotprod/(modi*modj)
	return q

# Calculate sum of two weight vectors
def wsum(wi, wj, requires_grad=False):
	if not requires_grad:
		return {name: (wi[name]+wj[name]).detach().clone() for name in wi}
	else:
		return {name: wi[name]+wj[name] for name in wi}

# Calculate difference of two weight vectors
def wdiff(wi, wj, requires_grad=False):
	if not requires_grad:
		return {name: (wi[name]-wj[name]).detach().clone() for name in wi}
	else:
		return {name: wi[name]-wj[name] for name in wi}

# Calculate the product between two weights vectors
def wprod(wi, wj, requires_grad=False):
	if not requires_grad:
		return {name: (wi[name]*wj[name]).detach().clone() for name in wi}
	else:
		return {name: wi[name]*wj[name] for name in wi}

# Multiply weight vector by constant
def kprod(w, k, requires_grad=False):
	if not requires_grad:
		return {name: (k*w[name]).detach().clone() for name in w}
	else:
		return {name: k*w[name] for name in w}

# Elevate weight vector elements to the k-th power
def kpow(w, k, requires_grad=False):
	if not requires_grad:
		return {name: (w[name]**k).detach().clone() for name in w}
	else:
		return {name: w[name]**k for name in w}

# Rescale the norm of a weight vector
def rescale(w, new_mod, old_mod=None, requires_grad=False):
	if old_mod is None: old_mod = compute_mod(w)
	return kprod(w, new_mod/old_mod, requires_grad=requires_grad)

# Produce a copy of the weight vector
def wcopy(w):
	return {name: w[name].detach().clone() for name in w}



""" ########################## """
""" OPERATIONS ON DICTIONARIES """
""" ########################## """

# Merge two dictionaries
def merge_dict(from_dict, into_dict, overwrite=True):
	if overwrite:
		for key in from_dict:
			into_dict[key] = from_dict[key]
	else:
		for key in from_dict:
			if key not in into_dict.keys():
				into_dict[key] = from_dict[key]
	return into_dict

# Merge many dictionaries
def merge_dicts(from_dicts, into_dict, overwrite=True):
	for from_dict in from_dicts:
		into_dict = merge_dict(from_dict, into_dict, overwrite)
	return into_dict

# Product between the values of two dictionaries (expected to be floats, tensors or arrays)
def prod_dicts(first_dict, second_dict, keys_from:str="first"):
	if keys_from == "first":
		return {first_dict[key]*second_dict[key] for key in first_dict}
	elif keys_from == "second":
		return {first_dict[key]*second_dict[key] for key in second_dict}
	else:
		raise ValueError(f"prod_dicts(): keys_from is supposed to be a string with values ('first', 'second'), but found {keys_from}.")

# Check that the keys of dictionary A are a subset of the keys of dictionary B
def is_subset(keys_A, keys_B):
	if len(keys_A) > len(keys_B):
		return False
	else:
		return all([k in keys_B for k in keys_A])



""" #################### """
""" OPERATIONS ON FLOATS """
""" #################### """

# Get order of magnitude of input number
def get_ofm(x):
	return abs(np.log10(abs(x)).astype(int))+1

# Evaluate if two numbers are close
def isclose(a, b, rel_tol=1e-08, abs_tol=0.0):
	return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

# Round up a x to be multiple of x0
def roundup(multiple, divisor):
	if multiple <= divisor:
		return divisor
	else:
		return round(multiple/divisor) * divisor

# Check the bounds of a variable x
def included(x, lim=[-math.inf, math.inf], eq=[0, 0]):
	return (x>=lim[0] if eq[0] else x>lim[0]) and (x<=lim[1] if eq[1] else x<lim[1])
