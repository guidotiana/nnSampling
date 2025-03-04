import torch
import math
import numpy as np
from time import process_time as ptime

from utils.operations import compute_mod2

# Default precision
torch.set_default_dtype(torch.float64)



# ------------------------ #
# Gradient Descent Trainer #
# ------------------------ #
class GDTrainer:

    ## Initialization
    def __init__(self, model, Cost, Metric, dataset, name:str='GDTrainer'):
        self.model = model
        self.Cost = Cost
        self.Metric = Metric
        self.dataset = dataset
        self.name = name


    ## Perform training through gradient descent
    def train(
            self, 
            pars, results_dir, weights_dir, prefix,
            start=None,
            save_step=10, print_step=10, wsave_step=1000, wsave_dmet=0.005
        ):
        self.pars = pars
        self.results_dir = results_dir
        self.weights_dir = weights_dir
        self.prefix = prefix

        if start is None:
            self.t0 = ptime()
            self.data = {'epoch': 0}
            obs = self._compute_observables(backward=False, extra=True)
            for key in obs: 
                self.data[key] = obs[key].item()
            self.data['time'] = 0.
            wsave_epoch = 0
        else:
            self.t0 = ptime()-start['data']['time']
            self.data = start['data'].copy()
            self.model.load(start['files']['model'])
        self._to_types()
        wsave_epoch = self.data['epoch']
        metbin = math.floor(self.data['metric']/wsave_dmet)
            
        if self.data['epoch'] == 0:
            if save_step > 0: self._save(header=True)
            if wsave_step > 0: self._wsave(0)
        if print_step > 0:
            self._prepare_printing()
            self._status()

        _ = self._compute_observables(backward=True, extra=False)
        grad = self.model.copy(grad=True)
        self.model.zero_grad()
        for _ in range(self.pars['epochs']):
            with torch.no_grad():
                for name in self.model.weights:
                    self.model.weights[name] -= self.pars['lr']*grad[name]

            self.data['epoch'] += 1
            obs = self._compute_observables(backward=True, extra=True)
            for key in obs:
                self.data[key] = obs[key].item()
            self.data['time'] = ptime()-self.t0
            grad = self.model.copy(grad=True)
            self.model.zero_grad()
                
            metbin_check = math.floor(self.data['metric']/wsave_dmet) > metbin
            wsave_epoch_check = (self.data['epoch'])-wsave_epoch >= wsave_step
            if metbin_check or wsave_epoch_check:
                self._wsave(self.data['epoch'])
                wsave_epoch = self.data['epoch']
                if metbin_check: 
                    metbin = math.floor(self.data['metric']/wsave_dmet)

            if self.data['epoch']%save_step == 0: self._save(header=False)
            if self.data['epoch']%print_step == 0: self._status()


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


    ## Save Trainer data
    def _save(self, header=False):
        if header:
            with open(f'{self.results_dir}/data.dat', 'w') as f:
                header, line = '', ''
                for key in self.data:
                    header = header + f'{key}\t'
                    line = line + f'{self.data[key]}\t'
                print(header[:-1], file=f)
                print(line[:-1], file=f)
        else:
            with open(f'{self.results_dir}/data.dat', 'a') as f:
                line = ''
                for key in self.data: line = line + f'{self.data[key]}\t'
                print(line[:-1], file=f)


    ### Save Trainer current weights vector and optimizer state
    def _wsave(self, epoch):
        self.model.save(f=f'{self.weights_dir}/{self.prefix}{epoch}.pt')


    ## Print Trainer status
    def _status(self):
        self.data['time_h'] = self.data["time"] / 3600.
        line = ''
        for key, _, fp in self.printing_stuff['data']: line = f'{line}|{format(self.data[f"{key}"], f".{fp}f"):^12}'
        line = f'{line}|'
        self.data.pop('time_h')
        print(f'{line}\n{self.separator}')


    ## Prepare printing of Trainer data
    def _prepare_printing(self):
        self.printing_stuff = {
                'data':[
                    ['epoch', 'epoch', 0],
                    ['loss', 'U', 5],
                    ['cost', 'loss', 5],
                    ['metric', 'metric', 5],
                    ['mod2', 'mod2', 1],
                    ['time_h', 'time', 2],
                ]
        }

        self.separator = ''.join(['-']*(13*len(self.printing_stuff['data'])+1))
        header = ''
        for _, symbol, _ in self.printing_stuff['data']: header = f'{header}|{symbol:^12}'
        header = f'{header}|'

        print(f'// {self.name} status register:')
        print(f'{self.separator}\n{header}\n{self.separator}')

    
    ## Convert to correct types each dictionary values
    def _to_types(self):
        types_and_keys = {
                'pars': [
                    (int, ['epochs']),
                ],
                'data': [
                    (int, ['epoch']),
                ],
        }
        self.pars = self._correct(self.pars, types_and_keys['pars'])
        self.data = self._correct(self.data, types_and_keys['data'])
        
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
