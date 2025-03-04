import os, shutil
import torch
import numpy as np
from time import process_time as ptime

from utils.rng_state import load_torch_state, save_torch_state
from utils.operations import compute_d, compute_mod2, wcopy

# Default precision
torch.set_default_dtype(torch.float64)


# -------------------------- #
# Hybrid Monte Carlo Sampler #
# -------------------------- #
class HMCSampler():

    ## Initialize the sampler
    def __init__(self, model, Cost, Metric, dataset, generator, name:str='HMCSampler'):
        self.model = model
        self.Cost = Cost
        self.Metric = Metric
        self.dataset = dataset
        self.generator = generator
        self.name = name


    ## Perform Hybrid Monte Carlo sampling
    def sample(
            self, 
            pars, results_dir, weights_dir, prefix,
            start=None, keep_going=False,
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
                self.t0 = ptime()
                self.data = {'move': 0}
                obs = self._compute_observables(backward=False, extra=True)
                for key in obs:
                    self.data[key] = obs[key].item()
                self.data['time'] = 0.
            self.check = {
                    'move': self.data['move'],
                    'am': 0,
                    'btb_d': 0.,
                    'step_dt': 0.,
                    'eff_v': 0.,
            }
        else:
            self.t0 = ptime()-start['data']['time']
            self.data = start['data'].copy()
            self.check = start['check'].copy()
            self.model.load(start['files']['model'])
            self.generator = load_torch_state(start['files']['generator'], self.generator)
        self._to_types()

        if self.data['move'] == 0:
            if save_step > 0: self._save(self.data, f'{self.results_dir}/data.dat', header=True)
            if check_step > 0: self._save(self.check, f'{self.results_dir}/check.dat', header=True)
            if wsave_step > 0: self._wsave(0)
        if print_step > 0: 
            self._prepare_printing()
            self._status()
        self._to_types()

        wi = self.model.copy(grad=False)
        for _ in range(self.pars['moves']):
            step_dt = ptime()

            self._init_momenta()
            Ki = self._compute_K()
            obf = self._integrate()
            Kf = self._compute_K()
            dE = (obf['loss']-self.data['loss'] + Kf-Ki).item()
            
            p = torch.rand(1, device=self.model.device, generator=self.generator).item()
            if p <= np.exp(-dE/self.pars['T']):
                for key in obf:
                    self.data[key] = obf[key].item()
                self.check['am'] += 1
                self.check['btb_d'] = compute_d(self.model.weights, wi).item()
                wi = self.model.copy(grad=False)
            else:
                self.check['btb_d'] = 0.
                self.model.set_weights(wi)

            self.data['move'] += 1
            self.data['time'] = ptime()-self.t0
            self.check['move'] += 1
            self.check['step_dt'] = ptime()-step_dt
            self.check['eff_v'] = self.check['btb_d']/self.check['step_dt']

            if save_step > 0 and self.data['move']%save_step == 0: self._save(self.data, f'{self.results_dir}/data.dat')
            if check_step > 0 and self.data['move']%check_step == 0: self._save(self.check, f'{self.results_dir}/check.dat')
            if wsave_step > 0 and self.data['move']%wsave_step == 0: self._wsave(self.data['move'])
            if print_step > 0 and self.data['move']%print_step == 0: self._status()


    ## Velocity verlet integration algorithm
    def _integrate(self):
        _ = self._compute_observables(backward=True, extra=False)
        old_grad = self.model.copy(grad=True)
        self.model.zero_grad()

        new_grad = {}
        for step in range(self.pars['isteps']):
            with torch.no_grad():
                for name in self.model.weights:
                    self.model.weights[name] += self.momenta[name]*self.pars['dt']/self.pars['m'] - old_grad[name]*self.pars['dt']**2./(2.*self.pars['m'])

            last_step = step == self.pars['isteps']-1
            obf = self._compute_observables(backward=True, extra=last_step)

            for name in self.model.weights:
                new_grad[name] = self.model.weights[name].grad.detach().clone()
                self.momenta[name] -= (new_grad[name]+old_grad[name])*self.pars['dt']/2.

            if not last_step: old_grad = wcopy(new_grad)
            self.model.zero_grad()

        return obf


    ## Initialize momenta
    def _init_momenta(self):
        self.momenta = {
                name: torch.randn(tuple(self.model.weights[name].shape), device=self.model.device, generator=self.generator) * np.sqrt(self.pars['T']*self.pars['m']) 
                for name in self.model.weights
        }


    ## Calculate K
    def _compute_K(self):
        return 1./(2.*self.pars['m']) * compute_mod2(self.momenta).item()


    ## Calculate observables (and gradient)
    def _compute_observables(self, backward=True, extra=False):
        fx = self.model(self.dataset.x)
        cost = self.Cost(fx, self.dataset.y)
        mod2 = compute_mod2(self.model.weights)
        loss = cost + (self.pars['lamda']/2.)*mod2

        if backward:
            loss.backward(retain_graph=True)

        if extra:
            metric = self.Metric(fx, self.dataset.y)
            return {'loss':loss, 'cost':cost, 'mod2':mod2, 'metric':metric}
        else:
            return {'loss':loss, 'cost':cost, 'mod2':mod2}


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


    ## Save HMC sampling model weights, and torch generator state
    def _wsave(self, move):
        self.model.save(f'{self.weights_dir}/{self.prefix}{move}.pt')
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
        self.check['ar'] = self.check['am'] / self.check['move'] if self.check['move'] > 0 else 0.

        line = ''
        for key, _, fp in self.printing_stuff['data']: line = f'{line}|{format(self.data[f"{key}"], f".{fp}f"):^12}'
        line = f'{line}|' + ''.join([' ']*5)
        for key, _, fp in self.printing_stuff['check']: line = f'{line}|{format(self.check[f"{key}"], f".{fp}f"):^12}'
        line = f'{line}|'

        self.data.pop('time_h')
        self.check.pop('ar')
        print(f'{line}\n{self.separator}')


    ## Prepare printing of HMC data
    def _prepare_printing(self):
        self.printing_stuff = {
                'data':[
                    ['move', 'move', 0],
                    ['loss', 'U', 5],
                    ['cost', 'loss', 5],
                    ['metric', 'metric', 5],
                    ['mod2', 'mod2', 1],
                    ['time_h', 'time', 2],
                ],
                'check':[
                    ['move', 'move', 0],
                    ['ar', 'ar', 5],
                    ['eff_v', 'eff_v', 1],
                ]
        }

        self.separator = ''.join(['-']*(13*len(self.printing_stuff['data'])+1)) + ''.join([' ']*5) + ''.join(['-']*(13*len(self.printing_stuff['check'])+1))
        header = ''
        for _, symbol, _ in self.printing_stuff['data']: header = f'{header}|{symbol:^12}'
        header = f'{header}|' + ''.join([' ']*5)
        for _, symbol, _ in self.printing_stuff['check']: header = f'{header}|{symbol:^12}'
        header = f'{header}|'

        print(f'// {self.name} status register:')
        print(f'{self.separator}\n{header}\n{self.separator}')


    ## Convert to correct types each dictionary values
    def _to_types(self):
        types_and_keys = {
                'pars': [
                    (int, ['moves', 'isteps']),
                    (bool, ['lob']),
                ],
                'data': [
                    (int, ['move']),
                ],
                'check': [
                    (int, ['move', 'am']),
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
