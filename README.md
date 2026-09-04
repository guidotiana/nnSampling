# Artificial neural network sampling

This repository contains the data, code, and usage templates associated with the published article [PAPER](https://arxiv.org/pdf/2503.08266).


<details open><summary><b>Table of contents</b></summary>

- [Code](#Code)
  - [samplers/](#Code_samplers)
  - [models/](#Code_models)
  - [datasets/](#Code_datasets)
  - [generator/](#Code_generator)
  - [utils/](#Code_utils)
- [Article data](#Article-data)
</details>



---
## Code <a name="Code"></a>

The `code/` directory is the core of this repository, as it contains the implementation of the sampling algorithms, the neural network architectures and the datasets used in [PAPER](https://arxiv.org/pdf/2503.08266). Here follows a brief description of the content of each its sub-directories.

#### samplers/ <a name="Code_samplers"></a>
- `hmc_sampler.py`: the implementation of the hybrid Monte Carlo (hMC) algorithm as the python class `HMCSampler`. 
- `drhmc_sampler.py`: the implementation of the double-ratchet hybrid Monte Carlo (hMC) algorithm as the python class `DRHMCSampler`.
- `crhmc_sampler.py`: the implementation of the coupled-replicas hybrid Monte Carlo (hMC) algorithm as the python class `CRHMCSampler`.

#### models/ <a name="Code_models"></a>
- `nnmodel.py`: it contains the `NNModel` python class. It is derived from the `torch.nn.Module` class and it incapsulates the neural network model to be studied. It is used in the sampling classes `PLSampler`, `ConstrainedPLSampler` and `HMCSampler`.
- `commachine/`: implementation of the tree-like committee machine as the python class `ComMachine`, studied in [PAPER](https://arxiv.org/pdf/2503.08266).

#### datasets/ <a name="Code_datasets"></a>
- `RandomVariable/`: directory containing the code used to generate the random-label datasets, as described in [PAPER](https://arxiv.org/pdf/2503.08266).
- `SGCIFAR2/`: directory containing the scaled grayscale images of cats and dogs from the CIFAR10 dataset and the code used to generate datasets from them, as described in [PAPER](https://arxiv.org/pdf/2503.08266). 

#### generator/ <a name="Code_generator"></a>
- `custom_generator.py`: the implementation of a custom pytorch random-number generator named `CustomGenerator` used in the sampling classes `HMCSampler`, `DRHMCSampler` and `CRHMCSampler`.
- `rng_state.py`: code for the numpy pseudo-random number generator state file management.

#### utils/ <a name="Code_utils"></a>
- `general.py`: functions for input file reading and directory management.
- `geoline.py`: it contains the class GeoLine, which computes the geodesic between two weight configurations of the same ANN.
- `operations.py`: functions for operations on neural network parameters, dictionaries and float/integer variables.



---
## Article data <a name="Article-data"></a>
This directory, `PAPER_data/`, contains all the data necessary to reproduce the plots shown in [PAPER](https://arxiv.org/pdf/2503.08266).

---
