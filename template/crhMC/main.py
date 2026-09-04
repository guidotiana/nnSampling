import os, sys, argparse
import math
import torch
import torch.nn.functional as F

torch.cuda.empty_cache()

sys.path.append('../../code')

from models.commachine.commachine import ComMachine
from models.nnmodel import NNModel
from datasets.RandomVariable.rv_dataset import RVDataset
from samplers.crhmc_sampler import CRHMCSampler
from utils.general import load_inputs, create_path, find_path

# Load the input files and make the results directory
def prepare_directory(args):
	pars = {key: load_inputs(args.pars_file, start=f"## {key}", end="##") for key in ["model", "data", "cost", "sampler"]}
	settings = load_inputs(args.settings_file)

	create_path(settings['results_dir'])
	settings['results_dir'] = find_path(raw_path=settings['results_dir'], dname='sim', pfile=args.pars_file, pname='pars.txt', lpfunc=load_inputs)
	if 'weights_dir' not in settings.keys():
		settings['weights_dir'] = f"{settings['results_dir']}/weights"
	create_path(settings['weights_dir'])

	pars['model']['N_k'] = math.floor(pars['model']['N']/pars['model']['K'])
	pars['data']['shapex'] = (pars['model']['K'], pars['model']['N_k'])
	pars['data']['shapey'] = (1,)

	pars['model']['y'] = sum(['_from' in key for key in pars['model'].keys()])
	pars['model']['start_fns'] = {
		f'r{i}': pars['model'][f'r{i}_from'] for i in range(pars['model']['y'])
	}

	return pars, settings


# Main
def main(args):
	print(f'PID: {os.getpid()}\n')

	print('Loading inputs...')
	pars, settings = prepare_directory(args)

	print('Loading nn-models...')
	models = {
		f'r{i}': NNModel(
			ComMachine(
				N_k=pars['model']['N_k'],
				K=pars['model']['K'],
				activ=pars['model']['activ'],
				seed=pars['model']['model_seed']+i, #to avoid having a dictionary composed of the same models
			)
		) for i in range(pars['model']['y'])
	}

	print('Initializing datasets...')
	datasets = {
		"train": RVDataset(
			P=pars['data']['P_train'],
			shapex=pars['data']['shapex'],
			shapey=pars['data']['shapey'],
			rho=pars['data']['rho'],
			seed=pars['data']['data_seed'],
		)
	}
    
	print('Defining cost and metric functions...')
	Cost = lambda fx, y: torch.log( 1. + torch.exp(-fx*y*pars['cost']['zeta']) ).sum() / pars['cost']['zeta']
	Metric = lambda fx, y: (torch.sign(fx) == y).sum() / len(y)

	print('Initializing sampler...')
	sampler = CRHMCSampler(
			models=models,
			datasets=datasets,
			Cost=Cost,
			Metric=Metric,
	)

	print(f'Starting the simulation!')
	sampler.sample(
			pars=pars["sampler"],
			settings=settings,
			start_fns=pars["model"]["start_fns"],
	)
	print(f'\nSimulation completed!')


def create_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--pars-file",
		type = str,
		default = "pars.txt",
		help = "str variable, path to the parameters file used in the simulation. Default: 'pars.txt'."
	)
	parser.add_argument(
		"--settings-file",
		type = str,
		default = "settings.txt",
		help = "str variable, path to a secondary input file for specifics which do not alter the simulation. Default: 'settings.txt'."
	)
	return parser

if __name__ == '__main__':
	parser = create_parser()
	args = parser.parse_args()
	main(args)
