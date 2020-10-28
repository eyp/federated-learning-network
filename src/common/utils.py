import sys
import torch
import numpy as np
from matplotlib.pyplot import plot as plt
from fastai.torch_core import to_np
import threading


def printf(format, *args):
    sys.stdout.write(format % args)


def plot_function(f, tx=None, ty=None, title=None, min=-2, max=2, figsize=(6, 4)):
    x = torch.linspace(min, max)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, f(x))
    if tx is not None: ax.set_xlabel(tx)
    if ty is not None: ax.set_ylabel(ty)
    if title is not None: ax.set_title(title)


def model_params_to_request_body(model_params):
    numpy_params = to_np(model_params)
    return {'weights': numpy_params[0].tolist(), 'bias': numpy_params[1].tolist()}


def request_params_to_model_params(request_data):
    weights = torch.tensor(np.array(request_data['weights']), dtype=torch.float, requires_grad=True)
    bias = torch.tensor(np.array(request_data['bias']), dtype=torch.float, requires_grad=True)
    return weights, bias


def synchronized(func):
    func.__lock__ = threading.Lock()

    def synced_func(*args, **kws):
        with func.__lock__:
            return func(*args, **kws)

    return synced_func
