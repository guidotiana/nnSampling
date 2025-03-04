import torch
import numpy as np
from math import sqrt, log2, acos

import sys
sys.path.append('..')
from model import ComMachineModel
from utils.general import isclose
from utils.operations import find_permutation, compute_mod2, compute_q, wsum, wdiff, kprod, rescale


# -------------------------------- #
# Geodesics between weight vectors #
# -------------------------------- #
class GeoLine:
    
    # load model, cost and metric functions, dataset and regularization
    def __init__(self, model, Cost, Metric, dataset, lamda):
        self.model = model
        self.Cost = Cost
        self.Metric = Metric
        self.dataset = dataset
        self.lamda = lamda
        
    # compute observables dictionary (loss, accuracy, q_cos) of <weights> (with respect to <to_weights>)
    def compute_observables(self, weights):
        if self.same_mod:
            mod2 = self.to_mod2
        else:
            mod2 = compute_mod2(weights).item()
        
        self.model.set_weights(weights)
        fx = self.model(self.dataset.x)
        cost = self.Cost(fx, self.dataset.y)
        loss = cost + (self.lamda/2.)*mod2
        metric = self.Metric(fx, self.dataset.y)
        q = compute_q(weights, self.to_weights, mods=(sqrt(mod2), sqrt(self.to_mod2)))
        
        return {'loss':loss.item(), 'cost':cost.item(), 'metric':metric.item(), 'q':q.item()}
    
    # initialize geodesic observables dictionary
    def init_obs(self, from_weights, N):
        new_obs = self.compute_observables(from_weights)
        obs = {
            key: [new_obs[key]] + [0.]*N for key in new_obs
        }
        return obs
    
    # compute geodesic observables dictionary based on <method> input
    def compute_geodesic(self, from_f:str, to_f:str, N:int=64, method:str='scaled line', lob_T:float=None, permute:bool=True, verbose:bool=True):
        ## check for invalid methods
        assert method in ('straight line', 'scaled line', 'recursive'), \
                "compute_geodesic(): invalid value for method variable. Allowed values: 'straight line', 'scaled line' (default), 'recursive'."
        
        ## if lob==True rescale lambda by T (lob_T!=None)
        if not lob_T is None:
            lamda_0 = self.lamda
            self.lamda *= lob_T

        ## load ends weights
        self.model.load(from_f)
        from_weights = self.model.copy()
        self.model.load(to_f)
        self.to_weights = self.model.copy()

        ## find optimal starting permutation
        if permute:
            from_weights, self.to_weights = find_permutation(from_weights, self.to_weights, check=True)
        
        ## STRAIGHT LINE (non-constant-modulus line, from from_weights to to_weights)
        if method == 'straight line':
            ### save <to_weights> squared modulus
            self.to_mod2 = compute_mod2(self.to_weights).item()
            self.same_mod = False
            
            ### compute straight line observables
            obs = self.init_obs(from_weights, N)
            obs = self.line(from_weights, obs, N)

        else:
            ### calculate mean ipersphere radius
            from_mod2 = compute_mod2(from_weights).item()
            to_mod2 = compute_mod2(self.to_weights).item()
            mod = (sqrt(to_mod2)+sqrt(from_mod2))/2.
            self.same_mod = True

            ### rescale from_weights and to_weights
            from_weights = rescale(from_weights, mod)
            self.to_weights = rescale(self.to_weights, mod)
            self.to_mod2 = mod**2.

            ### SCALED LINE (constant-modulus line with uneven angle distribution, from from_weights to to_weights)
            if method == 'scaled line':
                obs = self.init_obs(from_weights, N)
                obs = self.line(from_weights, obs, N)

            ### RECURSIVE (constant-modulus line with even angle distribution, from from_weights to to_weights)
            else:
                #### check if N is power of 2
                K = log2(N)
                if not isclose(K, round(K, 0)):
                    K = int(round(K, 0))
                    if verbose: print(f'compute_geodesic(): alert! Only powers of 2 are allowed for N variable when method=="recursive"! N={int(N)} has been changed to N={int(2**K)}.')
                N = int(2**K)
                if verbose: print(f'compute_geodesic(): models saved through geodesic line computation: {int(2+K)}. Check available memory!')

                obs = self.init_obs(from_weights, N)
                obs = self.recursive(from_weights, self.to_weights, obs, 0, N)
                new_obs = self.compute_observables(self.to_weights)
                for key in new_obs:
                    obs[key][N] = new_obs[key]

        ## add 'phi' and 'gamma' values along the line
        obs['phi'], obs['gamma'] = [], []
        for n, ob in enumerate(obs['q_cos']):
            phi = acos(ob) if not isclose(ob, 1.0) else 0.
            obs['phi'].append(phi)
            obs['gamma'].append(n/N)

        ## if lob==True, back to the starting lamda
        if not lob_T is None:
            self.lamda = lamda_0

        return obs
        
    # compute geodesic observables dictionary with the "straight line" or the "scaled line" method
    def line(self, from_weights, obs, N):
        delta_weights = wdiff(self.to_weights, from_weights)
        eps = 1./N
        eps_weights = kprod(delta_weights, eps)
        
        ## method = "scaled line"
        if self.same_mod:
            for i in range(N):
                from_weights = wsum(from_weights, eps_weights)
                scaled_weights = rescale(from_weights, sqrt(self.to_mod2))
                new_obs = self.compute_observables(scaled_weights)
                for key in new_obs:
                    obs[key][i+1] = new_obs[key]
        
        ## method = "straight line"
        else:
            for i in range(N):
                from_weights = wsum(from_weights, eps_weights)
                new_obs = self.compute_observables(from_weights)
                for key in new_obs:
                    obs[key][i+1] = new_obs[key]
        return obs
    
    # compute geodesic observables dictionary with the "recursive" method
    def recursive(self, weights_i, weights_j, obs, i, j):
        assert j > i, f'recursive(): something went wrong! Expected: j ({j}) > i ({i})! Exit!'
        if (j-i)%2 == 0:
            k = i + (j-i)//2
            delta_weights = wdiff(weights_j, weights_i)
            weights_k = wsum(weights_i, kprod(delta_weights, 0.5))
            weights_k = rescale(weights_k, sqrt(self.to_mod2))
            
            new_obs = self.compute_observables(weights_k)
            for key in new_obs:
                obs[key][k] = new_obs[key]
                
            obs = self.recursive(weights_i, weights_k, obs, i, k)
            obs = self.recursive(weights_k, weights_j, obs, k, j)
        return obs
