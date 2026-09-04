import os

from io import StringIO
from time import process_time as ptime
from collections.abc import Callable

import math
import torch
import numpy as np

from models.nnmodel import NNModel
from generator.custom_generator import CustomGenerator
from utils.general import create_path
from utils.operations import wcopy, compute_q, compute_d, compute_mod2, is_subset, roundup



### ----------------------------------------- ###
### Double-Ratchet Hybrid Monte Carlo Sampler ###
### ----------------------------------------- ###
class DRHMCSampler():

	def __init__(
			self,
			models: dict[NNModel],
			datasets: dict[torch.utils.data.Dataset],
			Cost: Callable,
			Metric: Callable,
			name: str = 'DRHMCSampler'
	):
		assert tuple(models.keys()) == ('fe', 'se'), f"{name}.__init__(): invalid models dictionary keys. Expected keys are: 'fe', 'se'."
		assert tuple(models['fe'].weights.keys()) == tuple(models['se'].weights.keys()), f"{name}.__init__(): mismatch between models['fe'] and models['se'] weights keys."
		self.models = models
		
		assert all([key in ["train", "val", "test"] for key in datasets]), f"{name}.__init__(): unexpected key in inputted datasets dictionary. Expected keys are: 'train', 'val', 'test'."
		assert "train" in datasets.keys(), f"{name}.__init__(): missing mandatory key 'train' in inputted datasets dictionary."
		self.datasets = datasets
		
		self.Cost = Cost
		self.Metric = Metric
		self.name = name

		self._init_attributes()



	def sample(
			self,
			pars: dict,
			settings: dict,
			start_fns: dict[str] | None = None,
	):
		pars_list, settings, data, distances = self._setup(pars, settings, start_fns)
		del pars

		for idx, pars in enumerate(pars_list):
			if idx > 0:
				data = self._start(distances, pars, settings, idx)

			data = self._correct_types(data, "data")
			pars = self._correct_types(pars, "pars")
            
			for move in range(data["move"]+1, pars["tot_moves"]+1):

				for end in ['fe', 'se']:
					# Save starting obs
					w_i = self.models[end].copy(grad=False)
					dmin_i = distances['dmin']
					U_i = data[f'{end}_loss'] + (pars['k']/2.)*distances['ratchet'].item()

					# Extract momenta and integrate the equations of motion
					obf, dK, distances = self._extract_and_integrate(end, pars, distances)
					U_f = obf['loss'] + (pars['k']/2.)*distances['ratchet'].item()

					# Compute energy difference and propose move
					dE = U_f-U_i + dK
					p = torch.rand(1, device=self.generator.device, generator=self.generator.get()).item()
					if p <= np.exp(-dE/pars['T']):
						for key,value in obf.items():
							self.data[f'{end}_{key}'] = value
						data[f'{end}_am'] += 1
					else:
						self.models[end].set_weights(w_i)
						d = compute_d(self.models['fe'].weights, self.models['se'].weights)
						distances['d'] = d.item()
						distances['dmin'] = dmin_i
						distances['ratchet'] = torch.max(d-distances['dmin'], torch.tensor([0.], device=self.generator.device))**2.

				# Complete update of the remaining observables
				data['move'] = move
				data['time'] = ptime() - self.t0
				data['d'] = distances['d']
				data['dmin'] = distances['dmin']
				data['ratchet'] = distances['ratchet'].item()
				data['q'] = compute_q(self.models['fe'].weights, self.models['se'].weights).item()
				self._extend_buffer(data)

				break_out = data['q'] >= pars['qmax']
				if break_out:
					print(f'Threshold met. Exit at move {move}!')
					self._save_log(data, settings)
					self._print_status(data)
					break

				# Save and/or print
				if move%settings['log_step'] == 0:
					self._save_log(data, settings)
				if move%settings['print_step'] == 0:
					self._print_status(data)

			if idx == len(pars_list):
				self._save_log(data, settings)
				self._print_status(data)

		del self.log, self.generator, self.t0



	def _setup(
			self,
			pars: dict,
			settings: dict,
			start_fns: dict[str] | None = None,
	):
		# 1. PARS
		# First the inputted pars dictionary is checked, verifying the all the necessary keys are present.
		# Then, the pars dictionary is completed, adding missing keys and checking the type for the inputted ones.
		# Then, the pars dictionary is splitted up in a list of dictionaries, where each instance of parameters must be executed 
		# when the previous one has been completed. Finally, the range of the values are checked for each parameters instance.
		assert is_subset(pars.keys(), self.defpars.keys()), f"{self.name}._setup(): unexpected key in inputted pars dictionary. Expected keys are: {list(self.defpars.keys())}."
		for key, (value, typ) in self.defpars.items():
			if value is None:
				assert key in pars.keys(), f"{self.name}._setup(): necessary key '{key}' missing from the inputted pars dictionary."
			else:
				if key not in pars:
					pars[key] = value
				else:
					if isinstance(pars[key], list):
						try:
							pars[key] = [typ(el) for el in pars[key]]
						except ValueError:
							raise ValueError(f"{self.name}.setup(): pars '{key}' type should be {typ}, but found {type(pars[key][0])}.")
					else:
						try:
							pars[key] = typ(pars[key])
						except ValueError:
							raise ValueError(f"{self.name}.setup(): pars '{key}' type should be {typ}, but found {type(pars[key])}.")

		list_lengths = [len(v) for k,v in pars.items() if isinstance(v, list)]
		if len(list_lengths) > 0:
			assert len(set(list_lengths)) == 1, f"{self.name}.setup(): list keys in the inputted pars dictionary with different lengths ({list_lengths}). All list keys must have the same length."
			for key, value in pars.items():
				if not isinstance(value, list):
					pars[key] = [value]*list_lengths[0]

			tot_moves = 0
			pars_list = []
			for idx, (stime, dt, isteps) in enumerate(zip(pars['stime'], pars['dt'], pars['isteps'])):
				moves = int(stime/(dt*isteps))
				tot_moves += moves
				pars_idx = {'moves': moves, 'tot_moves': tot_moves}
				for key, value in pars.items():
					pars_idx[key] = value[idx]
				pars_list.append(pars_idx)

		else:
			pars["moves"] = int(pars["stime"]/(pars["dt"]*pars["isteps"]))
			pars['tot_moves'] = pars['moves']
			pars_list = [pars]

		for idx in range(len(pars_list)):
			assert all([v>=0. for k,v in pars_list[idx].items() if k in ["lamda", "bss"]]), (
				f'{self.name}._setup(): invalid value for one of the following keys ("lamda", "bss") at index {idx}. Allowed values: v>=0.'
			)
			assert all([v>0. for k,v in pars_list[idx].items() if k in ["stime", "moves", "tot_moves", "T", "k", "dt", "M"]]), (
				f'{self.name}._setup(): invalid value for one of the following keys ',
				f'("stime", "moves", "tot_moves", "T", "k", "dt", "M") at index {idx}. Allowed values: v>0.'
			)
			assert -1. <= pars_list[idx]["qmax"] <= 1., (
				f'{self.name}._setup(): invalid value for one of the following keys ("qmax") at index {idx}. Allowed values: -1<=v<=1.'
			)
			if pars_list[idx]["bss"] == 0:
				pars_list[idx]["bss"] = max([len(dataset) for key, dataset in self.datasets.items()])


		# 2. SETTINGS
		# First inputted settings are controlled and completed. Then, data, log and print steps are rounded to be divisible.
		# Finally threads and devices are set. Once the device is defined, the models, the data and the generator are loaded on the correct device.
		assert is_subset(settings.keys(), self.defsettings.keys()), f"{self.name}._setup(): unexpected key in inputted settings dictionary. Expected keys are: {list(self.defsettings.keys())}."
		for key, (value, typ) in self.defsettings.items():
			if key not in settings:
				settings[key] = value
				if key in ["results_dir", "weights_dir"]:
					create_path(value)
			else:
				try:
					settings[key] = typ(settings[key])
				except ValueError:
					raise ValueError(f"{self.name}.setup(): settings '{key}' type should be {typ}, but found {type(settings[key])}.")

		assert settings["data_step"] > 0 and settings["log_step"] > 0 and settings["print_step"] > 0, (
			f"{self.name}._setup(): 'steps' keys in settings must all be positive, "
			f"but found {settings['data_step']} ('data'), {settings['log_step']} ('log') and {settings['print_step']} ('print')."
		)
		settings["data_step"] = roundup(multiple=settings["data_step"], divisor=settings["step_scale"])
		settings["log_step"] = roundup(multiple=settings["log_step"], divisor=settings["step_scale"])

		assert settings["num_threads"] <= 5, f"{self.name}.setup(): invalid value for 'num_threads' variable {settings['num_threads']}. Allowed values: num_threads <= 5."
		torch.set_num_threads(settings["num_threads"])

		settings["device"] = torch.device(settings["device"]) if isinstance(settings["device"], str) else settings["device"]
		if ("cuda" in settings["device"].type) and (not torch.cuda.is_available()):
			settings["device"] = torch.device("cpu")

		for end in ['fe', 'se']:
			self.models[end].to(settings["device"])
		for key in self.datasets:
			self.datasets[key].to(settings['device'])
		self.generator = CustomGenerator(
				seed=pars_list[0]["seed"],
				device=settings["device"],
		)


		# 3. CLEAN
		# If restart is True (and everything required to restart a previous simulation exists), load the log information.
		# Otherwise, reset everything and proceed to clean every file (except for pars.txt) in results and weights directories.
		# The last three temporary attributes are here initiated, (log and t0).
		if settings["restart"]:
			nec_rfiles = ['data.dat', 'pars.txt', 'generator.npy', 'log.pt']
			nec_wfiles = ['fe_weights_0.pt', 'se_weights_0.pt']
			settings["restart"] *= all([rf in os.listdir(settings['results_dir']) for rf in nec_rfiles] + [wf in os.listdir(settings['weights_dir']) for wf in nec_wfiles])

		if settings["restart"]:
			self.log = torch.load(f'{settings["results_dir"]}/log.pt')

			data = self.log["data"].copy()
			self.t0 = ptime()-data["time"]
			pars_list = [
					pars_idx for pars_idx in pars_list if pars_idx["tot_moves"]>data["move"]
			]

			for end in ['fe', 'se']:
				self.models[end].load(self.log['files'][f'{end}_weights'])
			self.generator.load(self.log['files']['generator'])

			d = compute_d(self.models['fe'].weights, self.models['se'].weights)
			distances = {
				'd': d.item(),
				'dmin': data['dmin'],
				'ratchet': torch.max(
					d-data['dmin'],
					torch.tensor([0.], device=settings["device"]),
				)**2.
			}

			self._print_pars(pars_list[0], settings, 0)
			self._print_status(data, header=True)

		else:
			for d in [settings['results_dir'], settings['weights_dir']]:
				for fn in os.listdir(d):
					if fn == 'pars.txt': continue
					if os.path.isfile(f'{d}/{fn}'):
						os.remove(f'{d}/{fn}')

			self.t0 = ptime()
			if start_fns is not None:
				for end, start_fn in start_fns.items():
					if start_fn is not None:
						self.models[end].load(start_fn)

			d = compute_d(self.models['fe'].weights, self.models['se'].weights)
			distances = {
				'd': d.item(),
				'dmin': d.item(),
				'ratchet': torch.tensor([0.], device=settings["device"]),
			}
			data = self._start(distances, pars_list[0], settings, 0)


		return (
			pars_list,
			settings,
			data,
			distances,
		)



	def _start(self, distances, pars, settings, idx):
		self._print_pars(pars, settings, idx)

		data = {
			'move': pars['tot_moves']-pars['moves'],
			'time': ptime()-self.t0,
			'q': compute_q(self.models['fe'].weights, self.models['se'].weights).item(),
			'd': distances['d'],
			'dmin': distances['dmin'],
			'ratchet': distances['ratchet'].item(),
		}
		for end in ['fe', 'se']:
			obs = self._compute_observables(end=end, lamda=pars['lamda'], bss=pars['bss'], k=pars['k'], ratchet=distances['ratchet'], backward=False, extra=True)
			for key, value in obs.items():
				data[f'{end}_{key}'] = value
			data[f'{end}_am'] = 0

		self._extend_buffer(data, header=data['move']==0)
		self._save_log(data, settings)
		self._print_status(data, header=True)

		return data



	def _extract_and_integrate(self, end, pars, distances):
		momenta = self._init_momenta(pars)
		Ki = self._compute_K(momenta, pars)
		old_grad, obs = self._compute_grad(end, distances['ratchet'], pars, extra=False)

		for step in range(1, pars['isteps']+1):
			with torch.no_grad():
				for layer in self.models[end].weights:
					self.models[end].weights[layer] += momenta[layer]*pars['dt']/pars['M'] - old_grad[layer]*pars['dt']**2./(2.*pars['M'])

			d = compute_d(self.models['fe'].weights, self.models['se'].weights)
			distances['d'] = d.item()
			if d.item() <= distances['dmin']: 
				distances['dmin'] = d.item()
				distances['ratchet'] = torch.tensor([0.], device=self.generator.device)
			else:
				distances['ratchet'] = (d-distances['dmin'])**2

			last_step = step==pars['isteps']
			new_grad, obs = self._compute_grad(end, distances['ratchet'], pars, extra=last_step)

			for layer in self.models[end].weights:
				momenta[layer] -= (new_grad[layer]+old_grad[layer])*pars['dt']/2.

			if not last_step: old_grad = wcopy(new_grad)

		Kf = self._compute_K(momenta, pars)
		torch.cuda.empty_cache()
		return obs, Kf-Ki, distances



	def _init_momenta(self, pars):
		momenta = {
				layer: torch.randn(values.shape, device=self.generator.device, generator=self.generator.get()) * math.sqrt(pars['T']*pars['M'])
				for layer, values in self.models['fe'].weights.items()
		}
		return momenta

	def _compute_K(self, momenta, pars):
		K = 0.
		for layer, momenta_l in momenta.items():
			K += (0.5*(momenta_l**2.)/pars["M"]).sum()
		return K.item()

	def _compute_observables(self, end, lamda, bss, k=None, ratchet=None, backward=True, extra=False):
		mod2 = compute_mod2(self.models[end].weights)
		if backward:
			reg = (lamda/2.)*mod2 + (k/2.)*ratchet
			reg.backward()

		obs = {}
		if not extra:
			dataset = self.datasets["train"]
			P = len(dataset)
			Nbs = P//bss if P%bss==0 else P//bss+1

			cost = 0.
			for ibs in range(Nbs):
				x_bs, y_bs = dataset.x[ibs*bss:(ibs+1)*bss], dataset.y[ibs*bss:(ibs+1)*bss]
				fx = self.models[end].NN(x_bs)
				cost_bs = self.Cost(fx, y_bs) * len(x_bs)/P
				cost += cost_bs.detach().item()

				if backward:
					cost_bs.backward()

			obs["loss"] = cost + (lamda/2.)*mod2.detach().item()
			obs["cost"] = cost
			obs["mod2"] = mod2.detach().item()
			for key in self.datasets:
				obs[f'{key}_metric'] = None

		else:
			for key, dataset in self.datasets.items():
				P = len(dataset)
				Nbs = P//bss if P%bss==0 else P//bss+1

				if key == "train":
					cost, metric = 0., 0.
					for ibs in range(Nbs):
						x_bs, y_bs = dataset.x[ibs*bss:(ibs+1)*bss], dataset.y[ibs*bss:(ibs+1)*bss]
						fx = self.models[end].NN(x_bs)
						cost_bs = self.Cost(fx, y_bs) * len(x_bs)/P
						cost += cost_bs.detach().item()
						metric_bs = self.Metric(fx, y_bs) * len(x_bs)/P
						metric += metric_bs.detach().item()

						if backward:
							cost_bs.backward()

					obs["loss"] = cost + (lamda/2.)*mod2.detach().item()
					obs["cost"] = cost
					obs["mod2"] = mod2.detach().item()
					obs["train_metric"] = metric

				else:
					metric = 0.
					for ibs in range(Nbs):
						x_bs, y_bs = dataset.x[ibs*bss:(ibs+1)*bss], dataset.y[ibs*bss:(ibs+1)*bss]
						fx = self.models[end].NN(x_bs)
						metric_bs = self.Metric(fx, y_bs) * len(x_bs)/P
						metric += metric_bs.detach().item()

					obs[f"{key}_metric"] = metric

		return obs

	def _compute_grad(self, end, ratchet, pars, extra):
		obs = self._compute_observables(end=end, lamda=pars['lamda'], bss=pars['bss'], k=pars['k'], ratchet=ratchet, backward=True, extra=extra)
		grad = self.models[end].copy(grad=True)
		self.models[end].zero_grad()
		return grad, obs



	def _extend_buffer(self, dikt, header=False):
		if header:
			header, line = '', ''
			for key in dikt:
				header = header + f'{key}\t'
				line = line + f'{dikt[key]}\t'
			self.buffer.write(f"{header[:-1]}\n{line[:-1]}\n")
		else:
			line = ''
			for key in dikt: line = line + f'{dikt[key]}\t'
			self.buffer.write(f"{line[:-1]}\n")

	def _flush_buffer(self, settings):
		with open(f'{settings["results_dir"]}/data.dat', 'a') as f:
			print(self.buffer.getvalue(), file=f, end="")
		self.buffer.seek(0)
		self.buffer.truncate(0)

	def _save_log(self, data, settings):
		self._flush_buffer(settings)

		files = {}
		for end in ['fe', 'se']:
			files[f'{end}_weights'] = f'{settings["weights_dir"]}/{end}_weights_{data["move"]}.pt'
			self.models[end].save(files[f'{end}_weights'])
		files['generator'] = f'{settings["results_dir"]}/generator.npy'
		self.generator.save(files['generator'])

		self.log = {
			"data": data.copy(),
			"files": files,
		}
		torch.save(self.log, f'{settings["results_dir"]}/log.pt')

	def _print_status(self, data, header=False):
		if header:
			print(f'// {self.name} status register:')
			print(f'{self.separator}\n{self.header}\n{self.separator}')

		if data['move'] > 0:
			data['fe_ar'] = data['fe_am'] / data['move']
			data['se_ar'] = data['se_am'] / data['move']
		else:
			data['fe_ar'] = 1.
			data['se_ar'] = 1.
		data['time_h'] = data["time"] / 3600.

		line = ''
		for key, _, fp in self.formatter['sampling']: line = f'{line}|{format(data[f"{key}"], f".{fp}f"):^17}'
		line = f'{line}|' + ''.join([' ']*5)
		for key, _, fp in self.formatter['efficiency']: line = f'{line}|{format(data[f"{key}"], f".{fp}f"):^12}'
		line = f'{line}|'

		data.pop('fe_ar')
		data.pop('se_ar')
		data.pop('time_h')
		print(f'{line}\n{self.separator}')

	def _print_pars(self, pars, settings, idx):
		fixed = ''
		for name, param in self.models['fe'].NN.named_parameters():
			if not param.requires_grad:
				fixed = f'{fixed}, {name}'
		fixed = f'({fixed[2:]})'

		lines = []
		lines.append(f'# {self.name} parameters summary:')
		lines.append(f'# ')
		lines.append(f'# moves:                      {pars["moves"]:.1e}')
		lines.append(f'# temperature:                {pars["T"]:.1e}')
		lines.append(f'# ratchet constant:           {pars["k"]:.1e}')
		lines.append(f'# integration time step:      {pars["dt"]:.1e}')
		lines.append(f'# per-move integration steps: {pars["isteps"]:.0f}')
		lines.append(f'# weights mass:               {pars["M"]:.2f}')
		lines.append(f'# threshold similarity:       {pars["qmax"]:.3f}')
		lines.append(f'# L2 regularization term:     {pars["lamda"]:.1e}')
		lines.append(f'# ')
		lines.append(f'# fixed layers: {fixed}')
		lines.append(f'# ')
		lines.append(f'# results directory: {settings["results_dir"]}')
		lines.append(f'# weights directory: {settings["weights_dir"]}')
		if idx == 0:
			lines.append(f'# restart:           {bool(settings["restart"])}')
		lines.append(f'# ')

		max_length = max([len(line) for line in lines])
		print('\n')
		print(''.join(['#'] * (max_length+2)))
		for line in lines:
			line = line + ''.join([' '] * (max_length-len(line)+1)) + '#'
			print(line)
		print(''.join(['#'] * (max_length+2)))
		print()



	def _correct_types(self, d, dname):
		# pars dictionary
		if dname == "pars":
			types_and_keys = [(int,  ['moves', 'tot_moves', 'isteps', 'bss'])]
		# data dictionary
		else:
			types_and_keys = [(int, ['move', 'am'])]

		for key in d:
			for _type, keys in types_and_keys:
				if key in keys:
					d[key] = _type(d[key])

		return d

	def _init_attributes(self):
		self.buffer = StringIO()

		self.defpars = {
			"stime":(None, float),
			"T": (None, float),
			"k": (None, float),
			"dt": (1.0, float),
			"isteps": (100, int),
			"M": (1.0, float),
			"qmax": (0.999, float),
			"lamda": (0.0, float),
			"bss": (0, int),
			"seed": (0, int),
		}

		self.defsettings = {
			"results_dir": ("./results", str),
			"weights_dir": ("./results/weights", str),
			"data_step": (1, int),
			"log_step": (1, int),
			"print_step": (1, int),
			"step_scale": (1, int),
			"restart": (False, bool),
			"device": ("cpu", str),
			"num_threads": (1, int),
		}
        
		self.formatter = {
			'sampling':[
				['move', 'move', 0],
				['fe_loss', 'U (fe)', 5],
			] + [
				[f'fe_{key}_metric', f'{key}_metric (fe)', 5] for key in self.datasets
			] + [
				['fe_loss', 'U (se)', 5],
			] + [
				[f'se_{key}_metric', f'{key}_metric (se)', 5] for key in self.datasets
			] + [
				['time_h', 'time', 2],
			],
			'efficiency':[
				['move', 'move', 0],
				['q', 'q', 5],
				['fe_ar', 'ar (fe)', 3],
				['se_ar', 'ar (se)', 3],
			],
		}

		self.separator = ''.join(['-']*(18*len(self.formatter['sampling'])+1)) + ''.join([' ']*5) + ''.join(['-']*(13*len(self.formatter['efficiency'])+1))
		self.header = ''
		for _, symbol, _ in self.formatter['sampling']: self.header = f'{self.header}|{symbol:^17}'
		self.header = f'{self.header}|' + ''.join([' ']*5)
		for _, symbol, _ in self.formatter['efficiency']: self.header = f'{self.header}|{symbol:^12}'
		self.header = f'{self.header}|'
