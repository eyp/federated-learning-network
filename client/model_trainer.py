from fastai.vision.all import DataLoader
from abc import ABC, abstractmethod


class ModelTrainer(ABC):
    """
    This is the base class of model trainers.
    If you want to implement a new training, your class must inherit from ModelTrainer and
    implement the abstract methods.
    See an implementation in mnist_model_trainer.py
    """

    def __init__(self, model_params, client_config):
        self.training_dataloader = None
        self.validation_dataloader = None

        self.client_config = client_config
        self.model_params = model_params

    def train_model(self):
        training_dataset, validation_dataset = self.__load_datasets()
        self.training_dataloader = DataLoader(training_dataset, batch_size=self.client_config.batch_size)
        self.validation_dataloader = DataLoader(validation_dataset, batch_size=self.client_config.batch_size)
        for epoch in range(self.client_config.epochs):
            self.__train_epoch()
            print('Accuracy of model trained at epoch', epoch + 1, ':', self.__validate_epoch(), end='\n', flush=True)
        return self.model_params

    @abstractmethod
    def __train_epoch(self):
        """
        Implements the actual model training. It will be called the times defined in 'client_config.epochs'
        """
        raise NotImplementedError()

    @abstractmethod
    def __validate_epoch(self):
        """
        Validates the training
        :returns the accuracy of the model as float
        """
        raise NotImplementedError()

    @abstractmethod
    def __load_datasets(self):
        """
        Load the dataset used for training the model in ModelTrainer.train_model().
        :returns training_dataset, validation_dataset
        """
        raise NotImplementedError()

