import sys
import requests

from os import environ

from requests.exceptions import Timeout
from .utils import model_params_to_request_params
from .mnist_model_trainer import MnistModelTrainer
from .client_status import ClientStatus
from .config import DEFAULT_SERVER_URL


class Client:
    def __init__(self, client_url):
        self.client_url = client_url
        self.status = ClientStatus.IDLE
        self.SERVER_URL = environ.get('SERVER_URL')
        if self.SERVER_URL is None:
            print('Warning: SERVER_URL environment variable is not defined, using DEFAULT_SERVER_URL:', DEFAULT_SERVER_URL)
            self.SERVER_URL = DEFAULT_SERVER_URL

        if self.client_url is None:
            print('Error: client_url is missing, cannot create a client')
            return
        self.register()

    def do_training(self, model_params, federated_learning_config):
        if self.can_do_training():
            self.model_params = model_params
            print(federated_learning_config)
            client_model_trainer = MnistModelTrainer(self.model_params, federated_learning_config)
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
