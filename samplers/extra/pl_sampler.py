import os, shutil
import torch
import numpy as np
from time import process_time as ptime

from utils.rng_state import load_torch_state, save_torch_state
from utils.operations import compute_d, compute_mod2, wcopy, wsum, kpow

# Default precision
torch.set_default_dtype(torch.float64)


# ------------------------------------ #
# Double-Noise Pseudo-Langevin Sampler #
# ------------------------------------ #
class PLSampler():

    ## Initialize the sampler
    def __init__(self, model, Cost, Metric, dataset, generator, name:str='PLSampler'):
        self.model = model
        self.Cost = Cost
        self.Metric = Metric
        self.dataset = dataset
        self.generator = generator
        self.name = name


    ## Perform double-noise pseudo-Langevin sampling
    def sample(
            self, 
            pars, results_dir, weights_dir, prefixes,
            start=None, keep_going=False,
            save_step=10, check_step=10, wsave_step=1000, print_step=1000,
    ):
        self.pars = pars.copy()
        self.results_dir = results_dir
        self.weights_dir = weights_dir
        self.prefixes = prefixes
        if self.pars['lob']:
            self.pars['lamda'] *= self.pars['T']

        if start is None:
            if not keep_going:
                self.t0 = ptime()
                self._adjourn_data(0)
                self._init_check(0)
            self._adjourn_pars(self.data['move'])
            self.momenta = {
                    layer: torch.randn(self.model[layer].shape, device=self.model.device, generator=self.generator) * np.sqrt(self.pars['T']*self.pars['m'])
                    for layer in self.model.diff_layers
            }
        else:
            self.t0 = ptime()-start['data']['time']
            self.data = start['data'].copy()
            for key in start['vpars']: 
                self.pars[key] = start['vpars'][key]
            self.vpars = start['vpars'].copy()
            self.check = start['check'].copy()
            self.model.load(start['files']['model'])
            self.momenta = torch.load(start['files']['momenta'], map_location=self.model.device, weights_only=True)
            self.generator = load_torch_state(start['files']['generator'], self.generator)
        self._to_types()

        if self.data['move'] == 0:
            if save_step > 0:
                self._save(self.data, f'{self.results_dir}/data.dat', header=True)
                self._save(self.vpars, f'{self.results_dir}/vpars.dat', header=True)
            if check_step > 0: self._save(self.check, f'{self.results_dir}/check.dat', header=True)
            if wsave_step > 0: self._wsave(0)
        if print_step > 0:
            self._prepare_printing()
            self._status()

        for move in range(self.data['move'], self.data['move']+self.pars['moves']):
            if (move+1)%self.pars['adj_step'] == 0:
                self._adjourn_pars(move+1)
                self._save(self.vpars, f'{self.results_dir}/vpars.dat')
            
            if (move+1)%check_step == 0:
                step_dt = ptime()
                old_weights, old_K = self.model.copy(grad=False), self._compute_K()
                self._step()
                step_dt = ptime()-step_dt
                if (move+1)%save_step == 0:
                    self._adjourn_data(move+1)
                    self._save(self.data, f'{self.results_dir}/data.dat')
                self._check_detailed_balance(move+1, old_weights, old_K, step_dt)
                self._save(self.check, f'{self.results_dir}/check.dat')
                del old_weights, step_dt

            elif (move+1)%save_step == 0:
                self._step()
                self._adjourn_data(move+1)
                self._save(self.data, f'{self.results_dir}/data.dat')

            else:
                self._step()

            if wsave_step > 0 and (move+1)%wsave_step == 0: self._wsave(move+1)
            if print_step > 0 and (move+1)%print_step == 0: self._status()


    ## Integration step (velocity Verlet algorithm applied to Langevin equations)
    def _step(self):
        old_grad = self._compute_grad()
        old_noise = self._generate_noise()

        with torch.no_grad():
            for layer in self.model.diff_layers:
                self.model.weights[layer] += self.momenta[layer]*self.pars['c1']*self.pars['dt']/self.pars['m'] - old_grad[layer]*self.pars['dt']**2./(2.*self.pars['m']) + old_noise[layer]*self.pars['k_wn']*self.pars['dt']/self.pars['m']

        new_grad = self._compute_grad()
        new_noise = self._generate_noise()

        for layer in self.model.weights:
            self.momenta[layer] = self.momenta[layer]*self.pars['c1']**2. - (new_grad[layer]+old_grad[layer])*self.pars['c1']*self.pars['dt']/2. + old_noise[layer]*self.pars['c1']*self.pars['k_wn'] + new_noise[layer]*np.sqrt((1.-self.pars['c1']**2.)*self.pars['k_mb']**2. + self.pars['k_wn']**2.)


    ## Compute an estimate of the current standard deviation std deriving from the mini-batch extraction
    def _estimate_std(self):
        sum_grad, sum2_grad = {}, {}
        for iext in range(self.pars['extractions']):
            grad = self._compute_grad()
            if iext == 0:
                for layer in grad:
                    sum_grad[layer] = grad[layer].detach().clone()
                    sum2_grad[layer] = (grad[layer]**2.).detach().clone()
            else:
                for layer in grad:
                    sum_grad[layer] += grad[layer].detach().clone()
                    sum2_grad[layer] += (grad[layer]**2.).detach().clone()
        tot_extractions = self.pars['extractions']
        prev_std = torch.sqrt(( sum2_grad['committees']/(tot_extractions) - (sum_grad['committees']/(tot_extractions))**2. ).mean()).item()
        """
        This is done for a simple single-layer network. Actually, for real applications with multi-layer networks, 
        one could have different values of std along the network, especially in very deep ones.
        """

        while tot_extractions < self.pars['max_extractions']:
            for _ in range(self.pars['extractions']):
                grad = self._compute_grad()
                for name in grad:
                    sum_grad[name] += grad[name].detach().clone()
                    sum2_grad[name] += (grad[name]**2.).detach().clone()
            tot_extractions += self.pars['extractions']
            curr_std = torch.sqrt((sum2_grad['committees']/(tot_extractions) - (sum_grad['committees']/(tot_extractions))**2.).mean()).item()
            convergence = abs(curr_std-prev_std)/prev_std <= self.pars['threshold_est']
            prev_std = curr_std
            if convergence:
                break
        
        return prev_std


    ## Adjourn the current value of the mini-batch noise and check whether to increase the step
    def _adjourn_pars(self, move):
        curr_std = self._estimate_std()
        if 'std' not in self.pars.keys():
            self.pars['streak'] = 1
            self.pars['c1'] = np.sqrt(1.-self.pars['m1']**2.)
            self.pars['m'] = (self.pars['dt']*curr_std)**2./(4.*self.pars['T_mb']*self.pars['m1']**2.)
        else:
            keep_streak = abs(curr_std-self.pars['std'])/self.pars['std'] <= self.pars['threshold_adj']
            self.pars['streak'] = self.pars['streak']+1 if keep_streak else 1
        self.pars['adj_step'] = round( np.tanh(self.pars['streak']/self.pars['opt_streak']) * self.pars['max_adj_step']/self.pars['min_adj_step'] ) * self.pars['min_adj_step']
        self.pars['std'] = curr_std
            
        self.pars['T_mb'] = (self.pars['dt']*self.pars['std'])**2./(4.*self.pars['m']*self.pars['m1']**2.)
        self.pars['k_mb'] = self.pars['dt']*self.pars['std']/2.
        self.pars['k_wn'] = self.pars['k_mb'] * np.sqrt((self.pars['T']-self.pars['T_mb'])/self.pars['T_mb'])

        self.vpars = {'move': move}
        for key in ('std', 'T_mb', 'k_mb', 'k_wn', 'streak', 'adj_step'):
            self.vpars[key] = self.pars[key]

    
    ## Extract random mini-batch and return gradient computed on it
    def _compute_grad(self):
        mb_idxs = torch.zeros((len(self.dataset),), dtype=torch.bool)
        while mb_idxs.sum() < self.pars['mbs']:
            idx = torch.randint(low=0, high=len(self.dataset), size=(1,), device=self.generator.device, generator=self.generator)
            mb_idxs[idx] = True
        x, y, _ = self.dataset[mb_idxs]
        _ = self._compute_observables(x, y, backward=True, extra=False)
        grad = self.model.copy(grad=True)
        self.model.zero_grad()
        return grad
    
    
    ## Calculate observables (and eventually gradient)
    def _compute_observables(self, x, y, backward=True, extra=False):
        fx = self.model(x)
        cost = self.Cost(fx, y)
        mod2 = compute_mod2(self.model.weights)
        loss = cost + (self.pars['lamda']/2.)*mod2

        if backward:
            loss.backward(retain_graph=True)

        if extra: ### accuracy computed only on the full batch
            metric = self.Metric(fx, self.dataset.y)
            return {'loss':loss, 'cost':cost, 'mod2':mod2, 'metric':metric}
        else:
            return {'loss':loss, 'cost':cost, 'mod2':mod2}


    ## Generate true white noise
    def _generate_noise(self):
        noise = {
                layer: torch.randn(self.shapes[layer], device=self.model.device, generator=self.generator)
                for layer in self.model.diff_layers
        }
        return noise


    ## Adjourn data (computing observables on the whole dataset)
    def _adjourn_data(self, move):
        self.data = {'move':move}
        observables = self._compute_observables(self.dataset.x, self.dataset.y, backward=False, extra=True)
        for key in observables:
            self.data[key] = observables[key].item()
        self.data['time'] = ptime()-self.t0


    ## Compute the value of the kinetic energy
    def _compute_K(self):
        return 1./(2.*self.pars['m']) * compute_mod2(self.momenta).item()


    ## Check the validity of the detailed balance for gaussian-distributed noise
    def _check_detailed_balance(self, move, old_weights, old_K, step_dt):
        new_potential = self._compute_observables(self.dataset.x, self.dataset.y, backward=True, extra=False)['loss']
        new_grad = self.model.copy(grad=True)

        new_weights = self.model.copy(grad=False)
        self.model.set_weights(old_weights)
        old_potential = self._compute_observables(self.dataset.x, self.dataset.y, backward=True, extra=False)['loss']
        old_grad = self.model.copy(grad=True)
        self.model.set_weights(new_weights)

        dU = new_potential-old_potential
        logM = 0.
        for layer in self.model.diff_layers:
            logM_vector = -(new_weights[layer]-old_weights[layer])*(new_grad[layer]+old_grad[layer])/2.+(new_grad[layer]**2.-old_grad[layer]**2.)*self.pars['dt']**2./(8.*self.pars['m'])
            logM += logM_vector.detach().cpu().numpy().sum()

        new_K = self._compute_K()
        btb_d = compute_d(new_weights, old_weights).item()
        self.check = {
                'move': move,
                'logM': logM,
                'dU': dU,
                'dH': logM+dU,
                'wratio': logM/dU,
                'K': old_K,
                'dK': new_K-old_K,
                'btb_d': btb_d,
                'step_dt': step_dt,
                'eff_v': btb_d/step_dt,
        }


    ## Initialize check detailed balance dictionary
    def _init_check(self, move):
        self.check = {
                'move': move,
                'logM': 0.,
                'dU': 0.,
                'dH': 0.,
                'wratio': -1.,
                'K': 0.,
                'dK': 0.,
                'btb_d': 0.,
                'step_dt': 0.,
                'eff_v': 0.,
        }


    ## Save PL sampling data or detailed balace check data
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


    ## Save PL sampling model weights, and torch generator state
    def _wsave(self, move):
        self.model.save(f'{self.weights_dir}/{self.prefixes["w"]}{move}.pt')
        torch.save(self.momenta, f'{self.weights_dir}/{self.prefixes["m"]}{move}.pt')
        if move == 0:
            save_torch_state(f'{self.results_dir}/torch_state.npy', self.generator)
            shutil.copy(f'{self.results_dir}/torch_state.npy', f'{self.results_dir}/prev_torch_state.npy')
        else:
            os.remove(f'{self.results_dir}/prev_torch_state.npy')
            os.rename(f'{self.results_dir}/torch_state.npy', f'{self.results_dir}/prev_torch_state.npy')
            save_torch_state(f'{self.results_dir}/torch_state.npy', self.generator)


    ## Print PL sampling data
    def _status(self):
        self.data['time_h'] = self.data["time"] / 3600.

        line = ''
        for key, _, fp in self.printing_stuff['data']: line = f'{line}|{format(self.data[f"{key}"], f".{fp}f"):^12}'
        line = f'{line}|' + ''.join([' ']*5)
        for key, _, fp in self.printing_stuff['check']: line = f'{line}|{format(self.check[f"{key}"], f".{fp}f"):^12}'
        line = f'{line}|'

        self.data.pop('time_h')
        print(f'{line}\n{self.separator}')


    ## Prepare printing of PL data
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
                    ['dK', 'dK', 5],
                    ['ratio', 'logM/dU', 5],
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
                    (int, ['moves', 'mbs', 'max_extractions', 'extractions', 'max_adj_step', 'min_adj_step', 'opt_streak']),
                    (bool, ['lob']),
                ],
                'vpars': [
                    (int, ['move', 'streak', 'adj_step']),
                ],
                'data': [
                    (int, ['move']),
                ],
                'check': [
                    (int, ['move']),
                ]
        }
        self.pars = self._correct(self.pars, types_and_keys['pars'])
        self.vpars = self._correct(self.vpars, types_and_keys['vpars'])
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
