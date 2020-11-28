import torch


def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets == 1, 1 - predictions, predictions).mean()


def linear_model(matrix, weights, bias): return matrix @ weights + bias
