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
from utils.operations import wcopy, compute_q, compute_d, compute_mod, is_subset, roundup



### ------------------------------------------ ###
### Coupled-Replica Hybrid Monte Carlo Sampler ###
### ------------------------------------------ ###
class CRHMCSampler():

	def __init__(
			self,
			models: dict[NNModel],
			datasets: dict[torch.utils.data.Dataset],
			Cost: Callable,
			Metric: Callable,
			name: str = 'CRHMCSampler'
	):
		assert len(models)>1, f"{name}.__init__(): invalid length of models dictionary (found {len(models)}). The minimum number of models is 1."
		assert all([f'r{i}' in models.keys() for i in range(len(models))]), f"{name}.__init__(): invalid keys in models dictionary (found {models.keys()}) given its length (found {len(models)})."
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
		pars_list, settings, data, center, distances, mods = self._setup(pars, settings, start_fns)
		del pars

		for idx, pars in enumerate(pars_list):
			if idx > 0:
				data = self._start(distances, mods, pars, settings, idx)

			data = self._correct_types(data, "data")
			pars = self._correct_types(pars, "pars")
            
			for move in range(data["move"]+1, pars["tot_moves"]+1):

				for ri in self.models:
					# Save starting obs
					w_i = self.models[ri].copy(grad=False)
					U_i = data[f'{ri}_loss'] + (pars['gamma']/len(self.models))*sum(distances.values()).item()

					# Compute partial center
					pcenter = self._compute_partial_center(ri)

					# Extract momenta and integrate the equations of motion
					obf, dK, center, distances, mods = self._extract_and_integrate(ri, center, pcenter, distances, mods, pars)
					U_f = obf['loss'] + (pars['gamma']/len(self.models))*sum(distances.values()).item()

					# Compute energy difference and propose move
					dE = U_f-U_i + dK
					p = torch.rand(1, device=self.generator.device, generator=self.generator.get()).item()
					if p <= np.exp(-dE/pars['T']):
						for key,value in obf.items():
							self.data[f'{ri}_{key}'] = value
						data[f'{ri}_am'] += 1
					else:
						self.models[ri].set_weights(w_i)
						mods[ri] = compute_mod(self.models[ri].weights)
						center = self._adjourn_center(ri, pcenter, mods)
						distances = {rj: compute_d(self.models[rj].weights, center) for rj in self.models}
                        
				# Complete update of the remaining observables
				data['move'] = move
				data['time'] = ptime() - self.t0
				self._extend_buffer(data)

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
			assert all([v>=0. for k,v in pars_list[idx].items() if k in ["gamma", "lamda", "bss"]]), (
				f'{self.name}._setup(): invalid value for one of the following keys ("gamma", "lamda", "bss") at index {idx}. Allowed values: v>=0.'
			)
			assert all([v>0. for k,v in pars_list[idx].items() if k in ["stime", "moves", "tot_moves", "T", "k", "dt", "M"]]), (
				f'{self.name}._setup(): invalid value for one of the following keys ',
				f'("stime", "moves", "tot_moves", "T", "k", "dt", "M") at index {idx}. Allowed values: v>0.'
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

		for ri in self.models:
			self.models[ri].to(settings["device"])
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
			nec_wfiles = [f'{ri}_weights_0.pt' for ri in self.models]
			settings["restart"] *= all([rf in os.listdir(settings['results_dir']) for rf in nec_rfiles] + [wf in os.listdir(settings['weights_dir']) for wf in nec_wfiles])

		if settings["restart"]:
			self.log = torch.load(f'{settings["results_dir"]}/log.pt')

			data = self.log["data"].copy()
			self.t0 = ptime()-data["time"]
			pars_list = [
					pars_idx for pars_idx in pars_list if pars_idx["tot_moves"]>data["move"]
			]

			for ri in self.models:
				self.models[ri].load(self.log['files'][f'{ri}_weights'])
			self.generator.load(self.log['files']['generator'])

			center, distances, mods = self._init_center()

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
				for ri, start_fn in start_fns.items():
					if start_fn is not None:
						self.models[ri].load(start_fn)

			center, distances, mods = self._init_center()
			data = self._start(distances, mods, pars_list[0], settings, 0)


		return (
			pars_list,
			settings,
			data,
			center,
			distances,
			mods,
		)



	def _start(self, distances, mods, pars, settings, idx):
		self._print_pars(pars, settings, idx)

		data = {
			'move': pars['tot_moves']-pars['moves'],
			'time': ptime()-self.t0,
		}
		for ri in self.models:
			obs = self._compute_observables(ri=ri, distance=distances[ri], mod=mods[ri], lamda=pars['lamda'], bss=pars['bss'], gamma=pars['gamma'], backward=False, extra=True)
			for key, value in obs.items():
				data[f'{ri}_{key}'] = value
			data[f'{ri}_am'] = 0

		self._extend_buffer(data, header=data['move']==0)
		self._save_log(data, settings)
		self._print_status(data, header=True)

		return data



	def _init_center(self):
		mods = {ri: compute_mod(self.models[ri].weights) for ri in self.models}

		center = {}
		with torch.no_grad():
			for ri, model in self.models.items():
				if len(center)==0:
					center = model.copy(grad=False)
				else:
					for layer, value in model.weights.items():
						center[layer] += value.detach()

			scale = sum(mods.values()) / ( compute_mod(center) * len(self.models) )
			center = {layer: (scale*value).requires_grad_(False) for layer,value in center.items()}

		distances = {ri: compute_d(self.models[ri].weights, center) for ri in self.models}
		return center, distances, mods

	def _compute_partial_center(self, ri):
		pcenter = {}
		with torch.no_grad():
			for rj, model in self.models.items():
				if rj == ri: continue
				if len(pcenter)==0:
					pcenter = model.copy(grad=False)
				else:
					for layer, value in model.weights.items():
						pcenter[layer] += value.detach()
		return {layer: value.requires_grad_(False) for layer, value in pcenter.items()}

	def _adjourn_center(self, ri, pcenter, mods):
		with torch.no_grad():
			center = {layer: (pcenter[layer]+value).detach() for layer, value in self.models[ri].weights.items()}
			scale = sum(mods.values()) / ( compute_mod(center) * len(self.models) )
			center = {layer: (scale*value).requires_grad_(False) for layer,value in center.items()}
		return center



	def _extract_and_integrate(self, ri, center, pcenter, distances, mods, pars):
		momenta = self._init_momenta(pars)
		Ki = self._compute_K(momenta, pars)
		old_grad, obs = self._compute_grad(ri, distances[ri], mods[ri], pars, extra=False)

		for step in range(1, pars['isteps']+1):
			with torch.no_grad():
				for layer in self.models[ri].weights:
					self.models[ri].weights[layer] += momenta[layer]*pars['dt']/pars['M'] - old_grad[layer]*pars['dt']**2./(2.*pars['M'])

			mods[ri] = compute_mod(self.models[ri].weights)
			center = self._adjourn_center(ri, pcenter, mods)
			for rj in self.models:
				distances[rj] = compute_d(self.models[rj].weights, center)

			last_step = step==pars['isteps']
			new_grad, obs = self._compute_grad(ri, distances[ri], mods[ri], pars, extra=last_step)

			for layer in self.models[ri].weights:
				momenta[layer] -= (new_grad[layer]+old_grad[layer])*pars['dt']/2.

			if not last_step: old_grad = wcopy(new_grad)

		Kf = self._compute_K(momenta, pars)
		torch.cuda.empty_cache()
		return obs, Kf-Ki, center, distances, mods



	def _init_momenta(self, pars):
		momenta = {
				layer: torch.randn(values.shape, device=self.generator.device, generator=self.generator.get()) * math.sqrt(pars['T']*pars['M'])
				for layer, values in self.models['r0'].weights.items()
		}
		return momenta

	def _compute_K(self, momenta, pars):
		K = 0.
		for layer, momenta_l in momenta.items():
			K += (0.5*(momenta_l**2.)/pars["M"]).sum()
		return K.item()

	def _compute_observables(self, ri, distance, mod, lamda, bss, gamma=None, backward=True, extra=False):
		mod2 = mod**2
		if backward:
			reg = (lamda/2.)*mod2 + (gamma/len(self.models))*distance
			reg.backward()

		obs = {}
		if not extra:
			dataset = self.datasets["train"]
			P = len(dataset)
			Nbs = P//bss if P%bss==0 else P//bss+1

			cost = 0.
			for ibs in range(Nbs):
				x_bs, y_bs = dataset.x[ibs*bss:(ibs+1)*bss], dataset.y[ibs*bss:(ibs+1)*bss]
				fx = self.models[ri].NN(x_bs)
				cost_bs = self.Cost(fx, y_bs) * len(x_bs)/P
				cost += cost_bs.detach().item()

				if backward:
					cost_bs.backward()

			obs["loss"] = cost + (lamda/2.)*mod2.detach().item()
			obs["cost"] = cost
			obs["mod2"] = mod2.detach().item()
			obs["d"] = distance.detach().item()
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
						fx = self.models[ri].NN(x_bs)
						cost_bs = self.Cost(fx, y_bs) * len(x_bs)/P
						cost += cost_bs.detach().item()
						metric_bs = self.Metric(fx, y_bs) * len(x_bs)/P
						metric += metric_bs.detach().item()

						if backward:
							cost_bs.backward()

					obs["loss"] = cost + (lamda/2.)*mod2.detach().item()
					obs["cost"] = cost
					obs["mod2"] = mod2.detach().item()
					obs["d"] = distance.detach().item()
					obs["train_metric"] = metric

				else:
					metric = 0.
					for ibs in range(Nbs):
						x_bs, y_bs = dataset.x[ibs*bss:(ibs+1)*bss], dataset.y[ibs*bss:(ibs+1)*bss]
						fx = self.models[ri].NN(x_bs)
						metric_bs = self.Metric(fx, y_bs) * len(x_bs)/P
						metric += metric_bs.detach().item()

					obs[f"{key}_metric"] = metric

		return obs

	def _compute_grad(self, ri, distance, mod, pars, extra):
		obs = self._compute_observables(ri=ri, distance=distance, mod=mod, lamda=pars['lamda'], bss=pars['bss'], gamma=pars['gamma'], backward=True, extra=extra)
		grad = self.models[ri].copy(grad=True)
		self.models[ri].zero_grad()
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
		for ri in self.models:
			files[f'{ri}_weights'] = f'{settings["weights_dir"]}/{ri}_weights_{data["move"]}.pt'
			self.models[ri].save(files[f'{ri}_weights'])
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

		for ri in self.models:
			data[f'{ri}_ar'] = data[f'{ri}_am'] / data['move'] if data['move']>0 else 1.
		data['time_h'] = data["time"] / 3600.

		line = ''
		for key, _, fp in self.formatter['sampling']: line = f'{line}|{format(data[f"{key}"], f".{fp}f"):^12}'
		line = f'{line}|' + ''.join([' ']*5)
		for key, _, fp in self.formatter['efficiency']: line = f'{line}|{format(data[f"{key}"], f".{fp}f"):^12}'
		line = f'{line}|'

		for ri in self.models:
			data.pop(f'{ri}_ar')
		data.pop('time_h')
		print(f'{line}\n{self.separator}')

	def _print_pars(self, pars, settings, idx):
		fixed = ''
		for name, param in self.models['r0'].NN.named_parameters():
			if not param.requires_grad:
				fixed = f'{fixed}, {name}'
		fixed = f'({fixed[2:]})'

		lines = []
		lines.append(f'# {self.name} parameters summary:')
		lines.append(f'# ')
		lines.append(f'# moves:                      {pars["moves"]:.1e}')
		lines.append(f'# temperature:                {pars["T"]:.1e}')
		lines.append(f'# gamma:                      {pars["gamma"]:.1e}')
		lines.append(f'# integration time step:      {pars["dt"]:.1e}')
		lines.append(f'# per-move integration steps: {pars["isteps"]:.0f}')
		lines.append(f'# weights mass:               {pars["M"]:.2f}')
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
			"gamma": (None, float),
			"dt": (1.0, float),
			"isteps": (100, int),
			"M": (1.0, float),
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
			] + [
				[f'{ri}_loss', f'U ({ri})', 5] for ri in self.models
			] + [
				['time_h', 'time', 2],
			],
			'efficiency':[
				['move', 'move', 0],
			] + [
				[f'{ri}_ar', f'ar ({ri})', 3] for ri in self.models
			],
		}

		self.separator = ''.join(['-']*(13*len(self.formatter['sampling'])+1)) + ''.join([' ']*5) + ''.join(['-']*(13*len(self.formatter['efficiency'])+1))
		self.header = ''
		for _, symbol, _ in self.formatter['sampling']: self.header = f'{self.header}|{symbol:^12}'
		self.header = f'{self.header}|' + ''.join([' ']*5)
		for _, symbol, _ in self.formatter['efficiency']: self.header = f'{self.header}|{symbol:^12}'
		self.header = f'{self.header}|'
