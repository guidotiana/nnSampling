## Sampling the space of solutions of an artificial neural network

This repository contains the code to perform samplings of the space of parameters of a generic artificial neural network (ANN). In the $\textit{article_data}$ directory, one can find the results of the exploration of a tree-like committee machine (which are presented here https://arxiv.org/pdf/2503.08266). Inside, each folder "Fig" contains files which refer to the corresponding image of the cited article. To try and reproduce such results, general snippets of code are provided in the $\textit{examples}$ directory.

---

Brief description of each folder:
- in the directory $\textit{models}$, the class NNModel (see "nn_model.py") takes as input a generic architecture and renders it readable by the classes in other directories (such as $\textit{samplers}$ or $textit{plotting}$). In "commachine.py", a specific class (CMModel) for the tree-like committee machine architecture (ComMachine) is presented;
- the directory $\textit{datasets}$ includes datasets classes for handling random-variable data (see "rv_dataset.py") or grayscale images (see "gi_dataset.py");
- the directory $\textit{samplers}$ contains various classes to perform either Gradient Descent training (see "gd_trainer.py" file) or canonical ensemble samplings (see "samplers.py" files) for a given ANN. Each sampler is based on the Hybrid Monte Carlo algorithm for an efficient exploration of the space of parameters. A thorough description of their functioning is presented in the cited article;
- within $\textit{utils}$ can instead be found different functions for file handling, for the definition of cost functions and metrics (see "general.py"), to perform operations on the ANN parameters (see "operations.py") and for pseudo-random number generator state file management (see "rng_state.py");
- finally, in the $\textit{plotting}$ directory can be found the class GeoLine (see "geoline.py") to compute and study the geodesic between two weights configurations of the same ANN.
