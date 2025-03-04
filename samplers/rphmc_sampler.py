import os, shutil
import torch
import numpy as np
from time import process_time as ptime

from utils.rng_state import load_torch_state, save_torch_state
from utils.operations import wsum, wdiff, kprod, rescale, wcopy, compute_d, compute_mod, compute_mod2

# Default precision
torch.set_default_dtype(torch.float64)


# ----------------------------------------------------- #
# Implicit-Center Replicated Hybrid Monte Carlo Sampler #
# ----------------------------------------------------- #
class RPHMCSampler():

    ## Initialize the sampler
    def __init__(self, models, Cost, Metric, dataset, generator, name:str='RPHMCSampler'):
        self.name = name

        self.y = len(models)
        keys = [f'r{i}' for i in range(self.y)]
        check_keys = [key in models.keys() for key in keys]
        assert self.y > 2, f'{self.name}.__init__(): invalid length of models dictionary (found {self.y}). The minimum number of models is 3 (replicas).'
        assert check_keys, f'{self.name}.__init__(): invalid keys in models dictionary (found {models.keys()}) given its length (found {self.y}). Expected keys: {keys}.'

        self.models = {key: models[key] for key in keys}
        self.Cost = Cost
        self.Metric = Metric
        self.dataset = dataset
        self.generator = generator


    ## Hybrid Monte Carlo sampling
    def sample(
            self,
            pars, results_dir, weights_dir, prefix,
            start=None, keep_going=True,
            save_step=10, check_step=10, print_step=1000, wsave_step=1000
    ):
        self.pars = pars
        self.results_dir = results_dir
        self.weights_dir = weights_dir
        self.prefix = prefix
        if self.pars['lob']:
            self.pars['lamda'] *= self.pars['T']

        if start is None:
            if not keep_going:
                self._init_all()
                self.t0 = ptime()
                self.data = {'move': 0}
                for ri in self.models:
                    obs = self._compute_observables(ri=ri, backward=False, extra=True)
                    for key in obs:
                        self.data[f'{ri}_{key}'] = obs[key].item()
                    self.data[f'{ri}_d'] = self.distances[ri].item()
                self.data['time'] = 0.
            self.check = {'move': self.data['move']}
            for ri in self.models:
                self.check[f'{ri}_am'] = 0
                self.check[f'{ri}_btb_d'] = 0.
                self.check[f'{ri}_step_dt'] = 0.
                self.check[f'{ri}_eff_v'] = 0.
        else:
            self.t0 = ptime()-start['data']['time']
            self.data = start['data'].copy()
            self.check = start['check'].copy()
            for ri in self.models:
                self.models[ri].load(start['files']['models'][ri])
            self.generator = load_torch_state(start['files']['generator'], self.generator)
            self._init_all()
        self._to_types()

        if self.data['move'] == 0:
            if save_step > 0: self._save(self.data, f'{self.results_dir}/data.dat', header=True)
            if check_step > 0: self._save(self.check, f'{self.results_dir}/check.dat', header=True)
            if wsave_step > 0: self._wsave(0)
        if print_step > 0:
            self._prepare_printing()
            self._status()

        for _ in range(self.pars['moves']):
            for ri in self.models:
                step_dt = ptime()
                wi = self.models[ri].copy(grad=False)
                self._compute_partial_center(ri)

                self._init_momenta()
                Ki = self._compute_K()
                Ui = self.data[f'{ri}_loss'] + (self.pars['gamma']/self.y)*sum(self.distances.values()).item()
                obf = self._integrate(ri=ri)
                Kf = self._compute_K()
                Uf = (obf['loss'] + (self.pars['gamma']/self.y)*sum(self.distances.values())).item()
                dE = Uf-Ui + Kf-Ki

                p = torch.rand(1, device=self.models['r0'].device, generator=self.generator).item()
                if p <= np.exp(-dE/self.pars['T']):
                    for key in obf:
                        self.data[f'{ri}_{key}'] = obf[key].item()
                    for rj in self.distances:
                        self.data[f'{rj}_d'] = self.distances[rj].item()
                    self.check[f'{ri}_btb_d'] = compute_d(self.models[ri].weights, wi).item()
                    self.check[f'{ri}_am'] += 1
                else:
                    self.models[ri].set_weights(wi)
                    self.mods[ri] = compute_mod(self.models[ri].weights)
                    self._adjourn_center(ri)
                    for rj in self.models:
                        self.distances[rj] = compute_d(self.models[rj].weights, self.center)
                    self.check[f'{ri}_btb_d'] = 0.

                self.check[f'{ri}_step_dt'] = ptime()-step_dt
                self.check[f'{ri}_eff_v'] = self.check[f'{ri}_btb_d']/self.check[f'{ri}_step_dt']

            self.data['move'] += 1
            self.data['time'] = ptime()-self.t0
            self.check['move'] += 1

            if save_step > 0 and self.data['move']%save_step == 0: self._save(self.data, f'{self.results_dir}/data.dat')
            if check_step > 0 and self.data['move']%check_step == 0: self._save(self.check, f'{self.results_dir}/check.dat')
            if wsave_step > 0 and self.data['move']%wsave_step == 0: self._wsave(self.data['move'])
            if print_step > 0 and self.data['move']%print_step == 0: self._status()


    ## Velocity verlet integration algorithm
    def _integrate(self, ri, return_obs=False):
        _ = self._compute_observables(ri=ri, backward=True, extra=False)
        old_grad = self.models[ri].copy(grad=True)
        self.models[ri].zero_grad()

        new_grad = {}
        for step in range(self.pars['isteps']):
            with torch.no_grad():
                for name in self.models[ri].weights:
                    self.models[ri].weights[name] += self.momenta[name]*self.pars['dt']/self.pars['m'] - old_grad[name]*self.pars['dt']**2./(2.*self.pars['m'])

            self.mods[ri] = compute_mod(self.models[ri].weights)
            self._adjourn_center(ri)
            for rj in self.models: 
                self.distances[rj] = compute_d(self.models[rj].weights, self.center)
            last_step = step == self.pars['isteps']-1
            obf = self._compute_observables(ri=ri, backward=True, extra=last_step)

            for name in self.models[ri].weights:
                new_grad[name] = self.models[ri].weights[name].grad.detach().clone()
                self.momenta[name] -= (new_grad[name]+old_grad[name])*self.pars['dt']/2.

            if not last_step: old_grad = wcopy(new_grad)
            self.models[ri].zero_grad()

        return obf


    ## Calculate observables (and gradient)
    def _compute_observables(self, ri, backward=True, extra=False):
        fx = self.models[ri](self.dataset.x)
        cost = self.Cost(fx, self.dataset.y)
        mod2 = self.mods[ri]**2.
        loss = cost + (self.pars['lamda']/2.)*mod2

        if backward:
            U = loss + (self.pars['gamma']/self.y)*sum(self.distances.values())
            U.backward(retain_graph=True)

        if extra:
            metric = self.Metric(fx, self.dataset.y)
            return {'loss':loss, 'cost':cost, 'mod2':mod2, 'metric':metric}
        else:
            return {'loss':loss, 'cost':cost, 'mod2':mod2}


    ## Initialize momenta
    def _init_momenta(self):
        self.momenta = {
                name: torch.randn(tuple(self.models['r0'].weights[name].shape), device=self.models['r0'].device, generator=self.generator) * np.sqrt(self.pars['T']*self.pars['m'])
                for name in self.models['r0'].weights
        }


    ## Calculate K
    def _compute_K(self):
        return 1./(2.*self.pars['m']) * compute_mod2(self.momenta).item()


    ## Initialize center, moduli and distances of the various replicas
    def _init_all(self):
        self.mods = {ri: compute_mod(self.models[ri].weights) for ri in self.models}
        self.center = kprod(self.models['r0'].weights, 0., requires_grad=False)
        for ri in self.models:
            self.center = wsum(self.center, self.models[ri].weights, requires_grad=True)
        self.center = kprod(self.center, 1./self.y, requires_grad=True)
        if self.pars['rescale']:
            self.center = rescale(self.center, sum(self.mods.values())/self.y, requires_grad=True)
        self.distances = {ri: compute_d(self.models[ri].weights, self.center) for ri in self.models}


    ## Calculate partial center, removing the i-th replica
    def _compute_partial_center(self, ri):
        self.pcenter = kprod(self.models['r0'].weights, 0., requires_grad=False)
        for rj in self.models:
            if rj != ri:
                self.pcenter = wsum(self.pcenter, self.models[rj].weights, requires_grad=True)


    ## Adjourn center position, adding to the partial center the adjourned i-th replica (and, eventually, rescaling)
    def _adjourn_center(self, ri):
        self.center = wsum(self.pcenter, self.models[ri].weights, requires_grad=True)
        self.center = kprod(self.center, 1./self.y, requires_grad=True)
        if self.pars['rescale']:
            self.center = rescale(self.center, sum(self.mods.values())/self.y, requires_grad=True)


    ## Save HMC sampling data or detailed balace check data
    def _save(self, dikt, filename, header=False):
        if header:
            with open(filename, 'w') as f:
                header, line = '', ''
                for key in dikt:
                    header = header + f'{key}\t'
                    line = line + f'{dikt[key]}\t'
                print(header[:-1], file=f)
                print(line[:-1], file=f)
        else:
            with open(filename, 'a') as f:
                line = ''
                for key in dikt: line = line + f'{dikt[key]}\t'
                print(line[:-1], file=f)


    ## Save HMC sampling models weights, and torch generator state
    def _wsave(self, move):
        for ri in self.models:
            self.models[ri].save(f'{self.weights_dir}/{ri}_{self.prefix}{move}.pt')
        if move == 0:
            save_torch_state(f'{self.results_dir}/torch_state.npy', self.generator)
            shutil.copy(f'{self.results_dir}/torch_state.npy', f'{self.results_dir}/prev_torch_state.npy')
        else:
            os.remove(f'{self.results_dir}/prev_torch_state.npy')
            os.rename(f'{self.results_dir}/torch_state.npy', f'{self.results_dir}/prev_torch_state.npy')
            save_torch_state(f'{self.results_dir}/torch_state.npy', self.generator)


    ## Print HMC sampling data
    def _status(self):
        self.data['time_h'] = self.data["time"] / 3600.
        for ri in self.models:
            self.check[f'{ri}_ar'] = self.check[f'{ri}_am'] / self.check['move'] if self.check['move'] > 0 else 0.

        line = ''
        for key, _, fp in self.printing_stuff['data']: line = f'{line}|{format(self.data[f"{key}"], f".{fp}f"):^11}'
        line = f'{line}|' + ''.join([' ']*5)
        for key, _, fp in self.printing_stuff['check']: line = f'{line}|{format(self.check[f"{key}"], f".{fp}f"):^9}'
        line = f'{line}|'

        self.data.pop('time_h')
        for ri in self.models:
            self.check.pop(f'{ri}_ar')
        print(f'{line}\n{self.separator}')


    ## Prepare printing of HMC data
    def _prepare_printing(self):
        data_stuff = [['move', 'move', 0]] + [[f'{ri}_loss', f'U ({ri})', 5] for ri in self.models] + [['time_h', 'time', 2]]
        check_stuff = [['move', 'move', 0]] + [[f'{ri}_ar', f'ar ({ri})', 3] for ri in self.models]
        self.printing_stuff = {
                'data': data_stuff,
                'check': check_stuff
        }

        self.separator = ''.join(['-']*(12*len(self.printing_stuff['data'])+1)) + ''.join([' ']*5) + ''.join(['-']*(10*len(self.printing_stuff['check'])+1))
        header = ''
        for _, symbol, _ in self.printing_stuff['data']: header = f'{header}|{symbol:^11}'
        header = f'{header}|' + ''.join([' ']*5)
        for _, symbol, _ in self.printing_stuff['check']: header = f'{header}|{symbol:^9}'
        header = f'{header}|'

        print(f'// {self.name} status register:')
        print(f'{self.separator}\n{header}\n{self.separator}')


    ## Convert to correct types each dictionary values
    def _to_types(self):
        types_and_keys = {
                'pars': [
                    (int, ['moves', 'isteps']),
                    (bool, ['rescale', 'lob']),
                ],
                'data': [
                    (int, ['move']),
                ],
                'check': [
                    (int, ['move']+[f'{ri}_am' for ri in self.models]),
                ]
        }
        self.pars = self._correct(self.pars, types_and_keys['pars'])
        self.data = self._correct(self.data, types_and_keys['data'])
        self.check = self._correct(self.check, types_and_keys['check'])

    def _correct(self, d, types_and_keys):
        for key in d:
            found = False
            for _type, keys in types_and_keys:
                if key in keys:
                    d[key] = _type(d[key])
                    found = True
                    break
            if not found:
                d[key] = float(d[key])
        return d
