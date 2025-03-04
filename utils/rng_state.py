import numpy as np
import torch



""" Numpy random generator """

# Save numpy generator state
def save_np_state(f):
    state = np.random.get_state()
    state = np.array(state, dtype='object')
    with open(f, 'wb') as ff:
        np.save(ff, [state])

# Load numpy generator state
def load_np_state(f):
    with open(f, 'rb') as ff:
        state = np.load(ff, allow_pickle='True')
    state = tuple(state[0])
    np.random.set_state(state)



""" Torch random generator """

# Initiate torch random number generator
def init_torch_generator(seed=0, device='cpu'):
    generator = torch.Generator(device)
    generator.manual_seed(seed)
    return generator

# Save torch generator state
def save_torch_state(f, generator):
    state = generator.get_state().detach().numpy().copy()
    with open(f, 'wb') as ff:
        np.save(ff, state)

# Load torch generator state
def load_torch_state(f, generator):
    with open(f, 'rb') as ff:
        state = np.load(ff, allow_pickle=True)
    state = torch.tensor(state, dtype=torch.uint8)
    generator.set_state(state)
    return generator
