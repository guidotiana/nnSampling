import os
import torch
import argparse
import numpy as np
import pandas as pd
import math

import sys
sys.path.append('../..')

from models.commachine import CMModel
from datasets.rv_dataset import RVDataset
from samplers.gd_trainer import GDTrainer
from utils.general import load_stuff, load_inputs, create_path, find_path, choose_cost, choose_metric

# Default precision
torch.set_default_dtype(torch.float64)



# Check simulation inputs
def check_inputs(pars, extra, args):
    ## Define variables
    pars['N_k'] = math.floor(pars['N']/pars['K'])
    pars['P'] = math.floor(pars['alpha']*pars['N'])

    ## Set directories
    create_path(extra['results_dir'])
    extra['results_dir'] = find_path(raw_path=extra['results_dir'], dname='sim', pfile=args.pars_file, pname='pars.txt', lpfunc=load_inputs)
    extra['weights_dir'] = f"{extra['results_dir']}/weights"
    create_path(extra['weights_dir'])

    ## Device and threads
    if 'cuda' in extra['device']:
        extra['device'] = extra['device'] if torch.cuda.is_available() else 'cpu'
    assert extra['num_threads'] <= 5, f'main.check(): invalid value for "num_threads" variable: {extra["num_threads"]}. Allowed values: num_threads <= 5.'
    torch.set_num_threads(extra['num_threads'])

    return pars, extra


# Check if previous simulation data exists
def get_exists(extra):
    expected_files = ['data.dat', 'pars.txt']
    saved_files = [f for f in os.listdir(extra['results_dir']) if os.path.isfile(f'{extra["results_dir"]}/{f}')]
    check_files = [expected_file in saved_files for expected_file in expected_files]
    check_wfiles = f'{extra["prefix"]}0.pt' in os.listdir(extra['weights_dir'])
    return all(check_files + [check_wfiles])


# Restart previous simulation
def restart(extra):
    ## Get last trained epoch
    last_wepoch = max([int(f.strip(f'{extra["prefix"]}.pt')) for f in os.listdir(extra['weights_dir']) if os.path.isfile(f'{extra["weights_dir"]}/{f}') and (extra['prefix'] in f)])
    last = {}

    ## Load, clean and save files
    for fname in ['data']:
        stuff = load_stuff(f'{extra["results_dir"]}/{fname}.dat')
        stuff = stuff.loc[np.array(stuff['epoch']) <= last_wepoch, :].reset_index(drop=True)
        last[fname] = dict(stuff.loc[len(stuff)-1, :].copy())
        with open(f'{extra["results_dir"]}/{fname}.dat', 'w') as f:
            stuff.to_csv(f, sep='\t', index=False)
    
    ## Find last-saved model weights vector and torch generator state files
    last['files'] = {
            'model': f'{extra["weights_dir"]}/{extra["prefix"]}{last_wepoch}.pt',
    }
    
    return last


# Reset old simulation files
def reset(extra):
    try:
        os.remove(f'{extra["results_dir"]}/data.dat')
        for f in os.listdir(f'{extra["weights_dir"]}'):
            if os.path.isfile(f'{extra["weights_dir"]}/{f}'):
                os.remove(f'{extra["weights_dir"]}/{f}')
    except:
        pass

    return None


# Trainer parameters summary
def summary(tpars, extra, model, title):
    trainable = ''
    for name in model.weights: trainable = f'{trainable}, {name}'
    trainable = f'({trainable[2:]})'

    lines = []
    lines.append(f'# {title} parameters summary:')
    lines.append(f'# ')
    lines.append(f'# epochs:        {format(tpars["epochs"], ".1e")}')
    lines.append(f'# learning rate: {format(tpars["lr"], ".1e")}')
    lines.append(f'# lamda:         {format(tpars["lamda"], ".1e")}')
    lines.append(f'# ')
    lines.append(f'# trainable layers: {trainable}')
    lines.append(f'# ')
    lines.append(f'# results directory: {extra["results_dir"]}')
    lines.append(f'# weights directory: {extra["weights_dir"]}')
    lines.append(f'# restart:           {bool(extra["restart"])}')
    lines.append(f'# ')

    max_length = max([len(line) for line in lines])
    print(''.join(['#'] * (max_length+2)))
    for line in lines:
        line = line + ''.join([' '] * (max_length-len(line)+1)) + '#'
        print(line)
    print(''.join(['#'] * (max_length+2)))
    print()



# Main
def main(args):
    print(f'PID: {os.getpid()}\n')

    print('Loading inputs...')
    pars = load_inputs(args.pars_file)
    extra = load_inputs(args.extra_file)
    pars, extra = check_inputs(pars, extra, args)

    print('Initializing model...')
    model = CMModel(
            N_k=pars['N_k'],
            K=pars['K'],
            activ=pars['activ'],
            seed=pars['model_seed'],
            device=extra['device'],
            f=None,
    )

    print('Defining cost and metric functions...')
    Cost = choose_cost(
            name=pars['cost'],
            **pars
    )
    Metric = choose_metric(
            name=pars['metric'],
            **pars
    )

    print('Initializing dataset...')
    dataset = RVDataset(
            P=pars['P'], 
            shapex=(pars['K'], pars['N_k']), 
            shapey=(1,),
            valuex=pars['valuex'], 
            valuey='binary', 
            only_index=False,
            seed=pars['data_seed'], 
            device=extra['device'],
    )

    print('Initializing trainer...')
    trainer = GDTrainer(
            model=model, 
            Cost=Cost,
            Metric=Metric,
            dataset=dataset,
    )

    exists = get_exists(extra)
    if extra['restart'] and exists:
        print('Recovering previous training...')
        last = restart(extra)
    else:
        print('Resetting directories...')
        last = reset(extra)
    
    print(f'Starting the training!\n')
    if last is not None:
        pars['epochs'] -= last['data']['epoch']
    
    tpars = {
            'epochs': pars['epochs'],
            'lr': pars['lr'],
            'lamda': pars['lamda'],
    }
    summary(tpars, extra, model, trainer.name)
    trainer.train(
            pars=tpars,
            results_dir=extra['results_dir'],
            weights_dir=extra['weights_dir'],
            prefix=extra['prefix'],
            start=last,
            save_step=extra['save_step'],
            print_step=extra['print_step'],
            wsave_step=extra['wsave_step'],
            wsave_dmet=extra['wsave_dmet'],
    )

    print(f'\nTraining completed!')
    print(f'Total time: {format(trainer.data["time"]/3600., ".2f")} h')
    

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pars-file",
        type = str,
        default = "inputs/pars.txt",
        help = "str variable, (essential) parameters file used to load simulation parameters. Default: 'inputs/pars.txt'."
    )
    parser.add_argument(
        "--extra-file",
        type = str,
        default = "inputs/extra.txt",
        help = "str variable, (additive) input file used to load simulation parameters. Default: 'inputs/extra.txt'."
    )
    return parser
    
if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
