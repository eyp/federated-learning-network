import torch
import numpy as np
from fastai.torch_core import to_np

from .training_type import TrainingType


def model_params_to_request_params(training_type, model_params):
    if model_params is None:
        return {}
    if training_type == TrainingType.MNIST:
        numpy_params = to_np(model_params)
        return {'weights': numpy_params[0].tolist(), 'bias': numpy_params[1].tolist()}
    elif training_type == TrainingType.CHEST_X_RAY_PNEUMONIA:
        weights_array = []
        for i, weights in enumerate(model_params):
            print('model params SHAPE:', weights.shape)
            weights_array.append(np.array(weights).tolist())
        return {'weights': weights_array}
    else:
        raise ValueError('Unsupported training type', training_type)


def request_params_to_model_params(training_type, request_data):
    model_params = None
    if training_type == TrainingType.MNIST:
        weights = torch.tensor(np.array(request_data['weights']), dtype=torch.float, requires_grad=True)
        bias = torch.tensor(np.array(request_data['bias']), dtype=torch.float, requires_grad=True)
        model_params = weights, bias
    elif training_type == TrainingType.CHEST_X_RAY_PNEUMONIA:
        if 'weights' in request_data:
            weights_array = []
            weights_received = request_data['weights']
            print('Weights received length:', len(weights_received))
            for weights in weights_received:
                numpy_weights_array = np.array(weights)
                print('model weights SHAPE:', numpy_weights_array.shape)
                weights_array.append(np.array(weights))
            model_params = weights_array
        else:
            print('No weights found in the request')
            return None
    print('Model params received length:', len(model_params))
    return model_params

