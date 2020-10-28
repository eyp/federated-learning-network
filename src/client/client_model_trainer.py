import torch
from fastai.data.load import DataLoader


class ClientModelTrainer:
    def __init__(self, training_dataset, validation_dataset, model_params, client_config, model_function, loss_function):
        self.training_dataloader = DataLoader(training_dataset, batch_size=client_config.batch_size)
        self.validation_dataloader = DataLoader(validation_dataset, batch_size=client_config.batch_size)
        self.model_params = model_params
        self.model_function = model_function
        self.loss_function = loss_function
        self.learning_rate = client_config.learning_rate
        self.epochs = client_config.epochs

    def train_model(self):
        # print('Initial params:', self.model_params)
        for epoch in range(self.epochs):
            self.train_epoch()
            print('Accuracy of model trained at epoch', epoch + 1, ':', self.validate_epoch(), end='\n')
        return self.model_params

    def train_epoch(self):
        for train_data, train_labels in self.training_dataloader:
            self.calculate_gradients(train_data, train_labels)
            for model_param in self.model_params:
                model_param.data -= model_param.grad * self.learning_rate
                model_param.grad.zero_()

    def validate_epoch(self):
        accuracies = [self.accuracy(self.model_function(train_data, weights=self.model_params[0], bias=self.model_params[1]), train_labels) for
                      train_data, train_labels in
                      self.validation_dataloader]
        return round(torch.stack(accuracies).mean().item(), 4)

    def accuracy(self, train_data, train_labels):
        predictions = train_data.sigmoid()
        corrections = (predictions > 0.5) == train_labels
        return corrections.float().mean()

    def calculate_gradients(self, train_data, train_labels):
        predictions = self.model_function(train_data, self.model_params[0], self.model_params[1])
        loss = self.loss_function(predictions, train_labels)
        loss.backward()
