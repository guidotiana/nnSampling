import os, shutil
import torch
import numpy as np
from time import process_time as ptime

from utils.rng_state import save_torch_state, load_torch_state
from utils.operations import compute_d, compute_mod2, compute_q, wcopy

# Default precision
torch.set_default_dtype(torch.float64)



# ----------------------------------------- #
# Double-Ratchet Hybrid Monte Carlo Sampler #
# ----------------------------------------- #
class DRHMCSampler():

    ## Init
    def __init__(self, models, Cost, Metric, dataset, generator, name:str='DRHMCSampler'):
        self.name = name
        assert tuple(models.keys()) == ('fe', 'se'), f'{self.name}.__init__(): invalid "models" dictionary keys. Expected keys: "fe", "se".'
        assert tuple(models['fe'].weights.keys()) == tuple(models['se'].weights.keys()), f'{self.name}.__init__(): mismatch between models["fe"] and models["se"] weights keys.'
        self.models = models
        self.Cost = Cost
        self.Metric = Metric
        self.dataset = dataset
        self.generator = generator


    ## Double-ratchet HMC sampling
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
                d = compute_d(self.models['fe'].weights, self.models['se'].weights)
                self.distances = {
                        'move': 0,
                        'dmin': d.item(),
                        'd': d,
                        'ratchet': torch.tensor([0.], device=self.models['fe'].device)
                }
                self.t0 = ptime()
                self.data = {'move': 0}
                for which_end in ('fe', 'se'):
                    obs = self._compute_observables(which_end, backward=False, extra=True)
                    for key in obs:
                        self.data[f'{which_end}_{key}'] = obs[key].item()
                self.data['q'] = compute_q(self.models['fe'].weights, self.models['se'].weights)
                self.data['dmin'] = self.distances['dmin']
                self.data['d'] = self.distances['d'].item()
                self.data['ratchet'] = self.distances['ratchet'].item()
                self.data['time'] = 0.
            self.check = {'move': self.data['move']}
            for which_end in ('fe', 'se'):
                for key in ('am', 'btb_d', 'step_dt', 'eff_v'):
                    self.check[f'{which_end}_{key}'] = 0
        else:
            self.t0 = ptime()-start['data']['time']
            self.data = start['data'].copy()
            self.check = start['check'].copy()
            for which_end in ('fe', 'se'):
                self.models[which_end].load(start['files']['models'][which_end])
            self.generator = load_torch_state(start['files']['generator'], self.generator)
            d = compute_d(self.models['fe'].weights, self.models['se'].weights)
            self.distances = {
                    'move': 0,
                    'dmin': self.data['dmin'],
                    'd': d,
                    'ratchet': torch.tensor([0.], device=self.models['fe'].device)
            }
        self._to_types()

        if self.data['move'] == 0:
            if save_step > 0:  self._save(self.data, f'{self.results_dir}/data.dat', header=True)
            if check_step > 0: self._save(self.check, f'{self.results_dir}/check.dat', header=True)
            if wsave_step > 0: self._wsave(0)
        if print_step > 0:
            self._prepare_printing()
            self._status()

        break_out = self.data['q'] >= self.pars["qmin"]
        if break_out:
            print(f'Threshold already met. Exit!')
            return

        dmin_i = self.distances['dmin']
        for _ in range(self.pars['max_moves']):
            for which_end in ('fe', 'se'):
                step_dt = ptime()

                wi = self.models[which_end].copy(grad=False)
                self._init_momenta()
                Ki = self._compute_K()
                Ui = self.data[f'{which_end}_loss'] + (self.pars['k']/2.)*self.data['ratchet']
                
                obf = self._integrate(which_end)
                Kf = self._compute_K()
                Uf = (obf['loss'] + (self.pars['k']/2.)*self.distances['ratchet']).item()
                dE = Uf-Ui + Kf-Ki
                p = torch.rand(1, device=self.models['fe'].device, generator=self.generator).item()
                if p <= np.exp(-dE/self.pars['T']):
                    for key in obf:
                        self.data[f'{which_end}_{key}'] = obf[key].item()
                    self.data['q'] = compute_q(self.models['fe'].weights, self.models['se'].weights)
                    self.data['dmin'] = self.distances['dmin']
                    self.data['d'] = self.distances['d'].item()
                    self.data['ratchet'] = self.distances['ratchet'].item()
                    self.check[f'{which_end}_am'] += 1
                    self.check[f'{which_end}_btb_d'] = compute_d(self.models[which_end].weights, wi).item()
                    dmin_i = self.distances['dmin']
                else:
                    self.models[which_end].set_weights(wi)
                    self.distances['dmin'] = dmin_i
                    self.distances['d'] = compute_d(self.models['fe'].weights, self.models['se'].weights)
                    self.distances['ratchet'] = torch.max(self.distances['d']-self.distances['dmin'], torch.tensor([0.], device=self.models['fe'].device))**2.
                    self.check[f'{which_end}_btb_d'] = 0.

                self.check[f'{which_end}_step_dt'] = ptime()-step_dt
                self.check[f'{which_end}_eff_v'] = self.check[f'{which_end}_btb_d']/self.check[f'{which_end}_step_dt']

            self.data['move'] += 1
            self.data['time'] = ptime()-self.t0
            self.check['move'] += 1

            break_out = self.data['q'] >= self.pars['qmin']
            if break_out:
                print(f'Threshold met. Exit at move {self.data["move"]}!')
                self._save(self.data, f'{self.results_dir}/data.dat')
                self._save(self.check, f'{self.results_dir}/check.dat')
                self._wsave(self.data['move'])
                self._status()
                break

            if save_step > 0 and self.data['move']%save_step == 0: self._save(self.data, f'{self.results_dir}/data.dat')
            if check_step > 0 and self.data['move']%check_step == 0: self._save(self.check, f'{self.results_dir}/check.dat')
            if wsave_step > 0 and self.data['move']%wsave_step == 0: self._wsave(self.data['move'])
            if print_step > 0 and self.data['move']%print_step == 0: self._status()


    ## Velocity verlet integration algorithm
    def _integrate(self, which_end):
        _ = self._compute_observables(which_end, backward=True, extra=False)
        old_grad = self.models[which_end].copy(grad=True)
        self.models[which_end].zero_grad()

        new_grad = {}
        for step in range(self.pars['isteps']):
            with torch.no_grad():
                for name in self.models[which_end].weights:
                    self.models[which_end].weights[name] += self.momenta[name]*self.pars['dt']/self.pars['m'] - old_grad[name]*self.pars['dt']**2./(2.*self.pars['m'])

            d = compute_d(self.models['fe'].weights, self.models['se'].weights)
            if d.item() < self.distances['dmin']: self.distances['dmin'] = d.item()
            self.distances['d'] = d
            self.distances['ratchet'] = torch.max(self.distances['d']-self.distances['dmin'], torch.tensor([0.], device=self.models['fe'].device))**2.
            last_step = step == self.pars['isteps']-1
            obf = self._compute_observables(which_end, backward=True, extra=last_step)

            for name in self.models[which_end].weights:
                new_grad[name] = self.models[which_end].weights[name].grad.detach().clone()
                self.momenta[name] -= (new_grad[name]+old_grad[name])*self.pars['dt']/2.

            if not last_step: old_grad = wcopy(new_grad)
            self.models[which_end].zero_grad()

        return obf


    ## Calculate observables (and gradient)
    def _compute_observables(self, which_end, backward=True, extra=False):
        fx = self.models[which_end](self.dataset.x)
        cost = self.Cost(fx, self.dataset.y)
        mod2 = compute_mod2(self.models[which_end].weights)
        loss = cost + (self.pars['lamda']/2.)*mod2

        if backward:
            U = loss + (self.pars['k']/2.)*self.distances['ratchet']
            U.backward(retain_graph=True)

        if extra:
            metric = self.Metric(fx, self.dataset.y)
            return {'loss':loss, 'cost':cost, 'mod2':mod2, 'metric':metric}
        else:
            return {'loss':loss, 'cost':cost, 'mod2':mod2}


    ## Initialize momenta
    def _init_momenta(self):
        self.momenta = {
                name: torch.randn(tuple(self.models['fe'].weights[name].shape), device=self.models['fe'].device, generator=self.generator) * np.sqrt(self.pars['T']*self.pars['m'])
                for name in self.models['fe'].weights
        }


    ## Calculate K
    def _compute_K(self):
        return 1./(2.*self.pars['m']) * compute_mod2(self.momenta).item()


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
        for which_end in ('fe', 'se'):
            self.models[which_end].save(f'{self.weights_dir}/{which_end}_{self.prefix}{move}.pt')
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
        for which_end in ('fe', 'se'):
            self.check[f'{which_end}_ar'] = self.check[f'{which_end}_am']/self.check['move'] if self.check['move']>0 else 0.

        line = ''
        for key, _, fp in self.printing_stuff['data']: line = f'{line}|{format(self.data[f"{key}"], f".{fp}f"):^15}'
        line = f'{line}|' + ''.join([' ']*5)
        for key, _, fp in self.printing_stuff['check']: line = f'{line}|{format(self.check[f"{key}"], f".{fp}f"):^9}'
        line = f'{line}|'

        self.data.pop('time_h')
        for which_end in ('fe', 'se'):
            self.check.pop(f'{which_end}_ar')
        print(f'{line}\n{self.separator}')


    ## Prepare printing of HMC data
    def _prepare_printing(self):
        self.printing_stuff = {
                'data':[
                    ['move', 'move', 0],
                    ['fe_loss', 'U (fe)', 5],
                    ['se_loss', 'U (se)', 5],
                    ['fe_metric', 'metric (fe)', 5],
                    ['se_metric', 'metric (se)', 5],
                    ['q', 'q', 5],
                    ['time_h', 'time', 2],
                ],
                'check':[
                    ['move', 'move', 0],
                    ['fe_ar', 'ar (fe)', 5],
                    ['se_ar', 'ar (se)', 5],
                ]
        }

        self.separator = ''.join(['-']*(16*len(self.printing_stuff['data'])+1)) + ''.join([' ']*5) + ''.join(['-']*(10*len(self.printing_stuff['check'])+1))
        header = ''
        for _, symbol, _ in self.printing_stuff['data']: header = f'{header}|{symbol:^15}'
        header = f'{header}|' + ''.join([' ']*5)
        for _, symbol, _ in self.printing_stuff['check']: header = f'{header}|{symbol:^9}'
        header = f'{header}|'

        print(f'// {self.name} status register:')
        print(f'{self.separator}\n{header}\n{self.separator}')


    ## Convert to correct types each dictionary values
    def _to_types(self):
        types_and_keys = {
                'pars': [
                    (int, ['max_moves', 'isteps']),
                    (bool, ['lob']),
                ],
                'data': [
                    (int, ['move']),
                ],
                'check': [
                    (int, ['move']+[f'{which_end}_am' for which_end in self.models]),
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
