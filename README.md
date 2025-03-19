## Sampling the space of solutions of an artificial neural network

This repository contains the code to perform samplings of the space of parameters of a generic artificial neural network (ANN). Within $\textbf{article}$, one can find the results of the exploration of a tree-like committee machine (which are presented here https://arxiv.org/pdf/2503.08266). Inside, each folder "Fig" contains files which refer to the corresponding image of the cited article. To try and reproduce such results, general snippets of code are provided in the $\textbf{examples}$ directory for a tree-like committee machine network.

---

Brief description of each folder:
- in the directory $\textbf{models}$, the class NNModel (see "nn_model.py") takes as input a generic architecture and renders it readable by the classes in other directories (such as $\textbf{samplers}$ or $\textbf{plotting}$). In "commachine.py", a specific class (CMModel) for the tree-like committee machine architecture (ComMachine) is presented;
- the directory $\textbf{datasets}$ includes datasets classes for handling random-variable data (see "rv_dataset.py") or grayscale images (see "gi_dataset.py");
- the directory $\textbf{samplers}$ contains various classes to perform either Gradient Descent training (see "gd_trainer.py" file) or canonical ensemble samplings (see "samplers.py" files) for a given ANN. Each sampler is based on the Hybrid Monte Carlo algorithm for an efficient exploration of the space of parameters. A thorough description of their functioning is presented in the cited article;
- within $\textbf{utils}$ can be found different functions for file handling and the definition of cost functions and metrics (see "general.py"), to perform operations on the ANN parameters (see "operations.py") and for pseudo-random number generator state file management (see "rng_state.py");
- finally, in the $\textbf{plotting}$ directory, the class GeoLine (see "geoline.py") computes the geodesic between two weights configurations of the same ANN. Other files include functions for data plotting.
