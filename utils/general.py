import torch
import numpy as np
import pandas as pd
import os


# Read desired inputs from <f> file
def load_inputs(f):
    with open(f, 'r') as ff:
        lines = ff.readlines()
    
    inputs = {}
    for line in lines:
        if (line == '\n' or line[0] == '#'): continue
        line = line.split('\t')[0]
        key_end_idx, value_start_idx = None, None
        idx = 0
        while value_start_idx is None:
            if key_end_idx is None:
                if line[idx] == ' ': key_end_idx = idx
            else:
                if (value_start_idx is None) and (line[idx] != ' '): value_start_idx = idx
            idx += 1
        key, value = line[0:key_end_idx], line[value_start_idx:]
        value = value[:-1] if value[-1]=='\n' else value
        if (value[0] == '[') and (value[-1] == ']'):
            values_list = value[1:-1].split(',')
            try: inputs[key] = [eval(el) for el in values_list]
            except: inputs[key] = [el.replace(' ', '') for el in values_list]
        else:
            try: inputs[key] = eval(value)
            except: inputs[key] = value
    return inputs


# Create path
def create_path(path):
    dirs = path.split('/')
    if dirs[0] == '':
        raise ValueError(f'Invalid starting directory: "/{dirs[1]}".')
    elif set(dirs[0]) != {'.'}:
        dirs = ['./'] + dirs

    actual_dir = dirs.pop(0)
    for idx, new_dir in enumerate(dirs):
        cwd_dirlist = [d for d in os.listdir(f'{actual_dir}') if os.path.isdir(f'{actual_dir}/{d}')]
        if not new_dir in cwd_dirlist:
            os.mkdir(f'{actual_dir}/{new_dir}')
        actual_dir = f'{actual_dir}/{new_dir}'


# Find correct path based on pars file
def find_path(raw_path, dname, pfile, pname, lpfunc):
    pars = lpfunc(pfile)

    dirlist = [d for d in os.listdir(raw_path) if os.path.isdir(f"{raw_path}/{d}") and (dname in d)]

    for d in dirlist:
        saved_pars = lpfunc(f"{raw_path}/{d}/{pname}")
        if pars == saved_pars:
            return f"{raw_path}/{d}"

    counter_check = any([f=='counter.txt' for f in os.listdir(raw_path) if os.path.isfile(f"{raw_path}/{f}")])
    if not counter_check:
        counter = 0
    else:
        with open(f"{raw_path}/counter.txt", 'r') as f:
            lines = f.readlines()
        assert len(lines)==1, f'find_path(): Too many lines found in "counter.txt" file in {raw_path} directory. Expected: 1, found: {len(lines)}.'
        counter = int(lines[0])
    path = f"{raw_path}/{dname}{counter}"
    create_path(path)
    with open(f"{raw_path}/counter.txt", 'w') as f:
        print(f'{counter+1}', file=f)

    with open(f"{pfile}", "r") as f:
        plines = f.readlines()
    with open(f"{path}/{pname}", "w") as f:
        for pline in plines:
            print(pline, file=f, end='')
    return path


# Load sampling data or check dictionary
def load_stuff(f, heavy=False, step=10):
    if heavy:
        stuff = load_heavy_csv(f, step)
    else:
        with open(f, 'r') as ff:
            stuff = pd.read_csv(ff, sep='\t', header=0)
    return stuff


# Load heavy csv file
def load_heavy_csv(f, step=10):
    assert step > 1, f'load_heavy_csv(): variable step must be greater than 1 (found {step}). Exit!'
    ff = open(f, 'r')

    header = ff.readline()
    header = header.split('\t')
    header[-1] = header[-1][:-1] #eliminate the "\n" in the last column name

    iline = 0
    lines = []
    while True:
        line = ff.readline()
        if not line: break
        elif iline%step == 0:
            line = [float(value) for value in line.split('\t') if (value != '\n')]
            lines.append(line)
        iline += 1

    data = pd.DataFrame(lines, columns=header)
    return data


# Evaluate if two numbers are close
def isclose(a, b, rel_tol=1e-08, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


# Return the corresponding move from a wfile
def get_wmove(wfile, prefix='w'):
    return int(wfile.split('/')[-1].replace(prefix, '').replace('.pt', ''))


# Load weights files (of training or monte carlo simulations) from weights directory
def get_wfiles(weights_d, prefix='w', files_number=None, add_start=True, min_move=None, max_move=None, return_wmoves=False):
    lp = len(prefix)

    try:
        wfiles = sorted(
            [f'{weights_d}/{wf}' for wf in os.listdir(weights_d) 
            if os.path.isfile(f'{weights_d}/{wf}') and (wf != f'{prefix}0.pt') and (wf[:lp] == prefix)],
            key=lambda wf: get_wmove(wf, prefix)
        )
    except:
        wfiles = sorted(
            [f'{weights_d}/{wf}' for wf in os.listdir(weights_d)
            if os.path.isfile(f'{weights_d}/{wf}') and (wf != f'{prefix}0.pt') and (wf != f'{prefix}f.pt') and (wf[:lp] == prefix)],
            key=lambda wf: get_wmove(wf, prefix)
        )

    if min_move is not None:
        wfiles = [wf for wf in wfiles if get_wmove(wf, prefix) >= min_move]
    if max_move is not None:
        wfiles = [wf for wf in wfiles if get_wmove(wf, prefix) <= max_move]
    if files_number is not None:
        indexes = np.unique(np.linspace(0, len(wfiles)-1, files_number).astype(int))
        wfiles = np.array(wfiles)[::-1]
        wfiles = wfiles[indexes]
        wfiles = list(wfiles)[::-1]
    if add_start and (f'{prefix}0.pt' in os.listdir(weights_d)): 
        wfiles = [f'{weights_d}/{prefix}0.pt'] + wfiles
    wfiles = np.array(wfiles)
        
    if return_wmoves:
        wmoves = np.array([get_wmove(wf, prefix) for wf in wfiles])
        return wfiles, wmoves
    else:
        return wfiles


# Choose a cost function
def choose_cost(name:str, **kwargs):
    assert name in ('MSE', 'BCE', 'Hinge'), f"choose_cost(): invalid value for 'name' variable ({name}). Allowed values: 'MSE', 'BCE'(default zeta=0.5), 'Hinge'(default k=0.0, g=1.0)."
    if not 'mean' in kwargs.keys(): kwargs['mean'] = True

    if name == 'MSE':
        if kwargs['mean']: Cost = lambda fx, y: ((fx - y)**2).mean()
        else: Cost = lambda fx, y: (fx - y)**2

    elif name == 'BCE':
        if not 'zeta' in kwargs.keys(): kwargs['zeta'] = 0.5
        k = 1./(2.*kwargs['zeta'])
        if kwargs['mean']: Cost = lambda fx, y: k*torch.log(1.+torch.exp(-fx*y/k)).mean()
        else: Cost = lambda fx, y: k*torch.log(1.+torch.exp(-fx*y/k))

    else:
        if not 'k' in kwargs.keys(): kwargs['k'] = 0.
        if not 'g' in kwargs.keys(): kwargs['g'] = 1.
        if kwargs['mean']: Cost = lambda fx, y: torch.pow(torch.max(kwargs['k']-torch.mul(fx, y), torch.zeros_like(y)), kwargs['g']).mean()
        else: Cost = lambda fx, y: torch.pow(torch.max(kwargs['k']-torch.mul(fx, y), torch.zeros_like(y)), kwargs['g'])

    return Cost


# Choose a metric function
def choose_metric(name:str, **kwargs):
    assert name in ('accuracy'), f"choose_metric(): invalid value for 'name' variable ({name}). Allowed values: 'accuracy'."
    if name == 'accuracy':
        Metric = lambda fx, y: (torch.sign(fx) == y).sum() / len(y)
    return Metric


# Get order of magnitude of input number
def get_ofm(x):
    return abs(np.log10(abs(x)).astype(int))+1
