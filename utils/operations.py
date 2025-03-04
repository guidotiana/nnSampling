import torch
import numpy as np
from scipy.optimize import linear_sum_assignment


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
