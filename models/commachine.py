import torch
from torch import nn
from math import sqrt

# Default precision
torch.set_default_dtype(torch.float64)


# ----------------- #
# Committee Machine #
# ----------------- #
class ComMachine(nn.Module):

    def __init__(
            self,
            N_k:int,
            K:int,
            activ:str='ReLU',
            seed:int=0,
    ):
        super(ComMachine, self).__init__()
        assert K%2 == 0, f'ComMachine.__init__(): invalid value for K: {K}. Allowed values: K=2n, n=1,2,3...'
        assert activ in ('ReLU', 'Tanh'), f'ComMachine.__init__(): invalid value for activ: {activ}. Allowed values: "ReLU" (default), "Tanh".'
        self.N_k = N_k
        self.K = K
        self.activ = activ
        self.seed = seed
        torch.manual_seed(self.seed)

        ## committees and output layers init
        self.committees = 2./sqrt(self.N_k)*(torch.rand(size=(1, self.K, self.N_k))-0.5)
        self.committees = nn.Parameter(self.committees, requires_grad=True)
        if self.activ == 'ReLU':
            self.output = torch.cat(
                    (
                        torch.ones(size=(1, self.K//2)),
                        -torch.ones(size=(1, self.K//2))
                    ),
                    axis=1
            )
            self.output = nn.Parameter(self.output, requires_grad=False)
            self.committees_activation = nn.ReLU()
            self.output_activation = nn.Identity()
        else:
            self.output = torch.ones(size=(1, self.K))
            self.output = nn.Parameter(self.output, requires_grad=False)
            self.committees_activation = nn.Tanh()
            self.output_activation = nn.Tanh()

    def forward(self, x):
        x = (self.committees*x).sum(axis=-1)
        x = self.committees_activation(x/sqrt(self.N_k))
        x = (self.output*x).sum(axis=-1)
        x = self.output_activation(x/sqrt(self.K))
        return x.reshape(-1, 1)



# ----------------------- #
# Committee Machine Model #
# ----------------------- #
class CMModel(ComMachine):

    # Initialize neural network model
    def __init__(
            self,
            N_k:int,
            K:int,
            activ:str='ReLU',
            seed:int=0,
            device='cpu',
            f=None,
    ):
        super(CMModel, self).__init__(N_k, K, activ, seed)
        self.device = device if ('cuda' in device) and torch.cuda.is_available() else 'cpu'
        if f:
            self.load(f)
        else:
            self._to_device()
        self._init_weights()

    # Initialize weights dictionary
    def _init_weights(self):
        self.weights = {'committees': self.committees}

    # Returns a copy of the NN weights (or gradient)
    def copy(self, grad=False):
        if not grad:
            return {layer: self.weights[layer].detach().clone().requires_grad_(True) for layer in self.weights}
        else:
            return {layer: self.weights[layer].grad.detach().clone() for layer in self.weights}

    # Set NN weights
    def set_weights(self, weights):
        self.committees = nn.Parameter(weights['committees'].detach().clone(), requires_grad=True)
        self._init_weights()

    # Load NN weights from file
    def load(self, f):
        with open(f, 'rb') as ptf:
            self.load_state_dict(torch.load(ptf, map_location=torch.device(self.device)))
        self._to_device()

    # Save NN weights to file
    def save(self, f):
        with open(f, 'wb') as ptf:
            torch.save(self.state_dict(), ptf)

    # Transfer NN weights to device
    def _to_device(self):
        if 'cuda' in self.device:
            self.to(self.device)
