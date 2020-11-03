import sys
import requests
import random

from os import environ

from requests.exceptions import Timeout
from fastai.vision.all import *
from .utils import printf, model_params_to_request_params
from .client_model_trainer import ClientModelTrainer
from .client_status import ClientStatus
from .config import DEFAULT_SERVER_URL


class Client:
    def __init__(self, client_url):
        self.client_url = client_url
        self.status = ClientStatus.IDLE
        self.SERVER_URL = environ.get('SERVER_URL')
        if self.SERVER_URL is None:
            print('Warning: SERVER_URL environment variable is not defined, using DEFAUL_SERVER_URL:', DEFAULT_SERVER_URL)
            self.SERVER_URL = DEFAULT_SERVER_URL

        if self.client_url is None:
            print('Error: client_url is missing, cannot create a client')
            return
        self.register()

    def do_training(self, model_params, federated_learning_config):
        if self.can_do_training():
            self.model_params = model_params
            self.load_datasets()
            print(federated_learning_config)
            client_model_trainer = ClientModelTrainer(self.train_dataset, self.valid_dataset, self.model_params, federated_learning_config, Client.linear, Client.mnist_loss)
            self.status = ClientStatus.TRAINING
            print('Training started...')
            self.model_params = client_model_trainer.train_model()
            print('Training finished...')
            self.update_model_params_on_server()
            self.status = ClientStatus.IDLE
        else:
            print('Training requested but client status is', self.status)
        sys.stdout.flush()

    def update_model_params_on_server(self):
        request_url = self.SERVER_URL + '/model_params'
        request_body = model_params_to_request_params(self.model_params)
        request_body['client_url'] = self.client_url
        response = requests.put(request_url, json=request_body)
        print('Response received from updating server model params:', response)
        if response.status_code != 200:
            print('Error updating server model params. Error:', response.reason)
        else:
            print('Model params updated on server successfully')
        sys.stdout.flush()

    def can_do_training(self):
        return self.status == ClientStatus.IDLE

    def register(self):
        print('Registering in server:', self.SERVER_URL)
        request_url = self.SERVER_URL + '/client'
        try:
            print('Doing request', request_url)
            response = requests.post(request_url, data={'client_url': self.client_url}, timeout=5)
            print('Response received from registration:', response)
            if response.status_code != 201:
                print('Cannot register client in the system at', request_url, 'error:', response.reason)
            else:
                print('Client registered successfully')
        except Timeout:
            print('Cannot register client in the server, server is not responding')
        sys.stdout.flush()

    def load_datasets(self):
        print('Loading dataset MNIST_SAMPLE...')
        path = untar_data(URLs.MNIST_SAMPLE)
        print('Content of MNIST_SAMPLE:', path.ls())
        print("Content of 'train' directory of MNIST_SAMPLE", (path / 'train').ls())

        threes = random.sample((path / 'train' / '3').ls().sorted(), int(random.uniform(20, 30)))
        sevens = random.sample((path / 'train' / '7').ls().sorted(), int(random.uniform(20, 30)))

        three_tensors = [tensor(Image.open(image_path)) for image_path in threes]
        seven_tensors = [tensor(Image.open(image_path)) for image_path in sevens]

        printf("There are %d images of number 3 and %d of number 7\n", len(three_tensors), len(seven_tensors))

        stacked_threes = torch.stack(three_tensors).float() / 255
        stacked_sevens = torch.stack(seven_tensors).float() / 255

        valid_three_paths = random.sample((path / 'valid' / '3').ls(), int(random.uniform(10, 20)))
        valid_seven_paths = random.sample((path / 'valid' / '7').ls(), int(random.uniform(10, 20)))

        valid_three_tensors = torch.stack([tensor(Image.open(valid_three_path)) for valid_three_path in valid_three_paths])
        valid_three_tensors = valid_three_tensors.float() / 255
        valid_seven_tensors = torch.stack([tensor(Image.open(valid_seven_path)) for valid_seven_path in valid_seven_paths])
        valid_seven_tensors = valid_seven_tensors.float() / 255
        print('Shape of tensors of valid set of 3 images:', valid_three_tensors.shape, 'Shape of tensors of valid set of 7 images:',
              valid_seven_tensors.shape)

        train_images = torch.cat([stacked_threes, stacked_sevens]).view(-1, 28 * 28)
        train_labels = tensor([1] * len(threes) + [0] * len(sevens)).unsqueeze(1)
        self.train_dataset = list(zip(train_images, train_labels))
        print('Training images shape:', train_images.shape, ', training labels shape:', train_labels.shape)

        valid_images = torch.cat([valid_three_tensors, valid_seven_tensors]).view(-1, 28 * 28)
        valid_labels = tensor([1] * len(valid_three_tensors) + [0] * len(valid_seven_tensors)).unsqueeze(1)
        self.valid_dataset = list(zip(valid_images, valid_labels))
        print('Dataset ready to be used')
        sys.stdout.flush()

    @staticmethod
    def mnist_loss(predictions, targets):
        predictions = predictions.sigmoid()
        return torch.where(targets == 1, 1 - predictions, predictions).mean()

    @staticmethod
    def linear(matrix, weights, bias): return matrix @ weights + bias
