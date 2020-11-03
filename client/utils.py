import sys
import torch
import numpy as np
from fastai.torch_core import to_np


def printf(format, *args):
    sys.stdout.write(format % args)


def model_params_to_request_params(model_params):
    numpy_params = to_np(model_params)
    return {'weights': numpy_params[0].tolist(), 'bias': numpy_params[1].tolist()}


def request_params_to_model_params(request_data):
    weights = torch.tensor(np.array(request_data['weights']), dtype=torch.float, requires_grad=True)
    bias = torch.tensor(np.array(request_data['bias']), dtype=torch.float, requires_grad=True)
    return weights, bias
