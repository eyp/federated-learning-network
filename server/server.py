import asyncio
import sys
import aiohttp
import torch

from .utils import model_params_to_request_params
from .federated_learning_config import FederatedLearningConfig
from .client_training_status import ClientTrainingStatus
from .server_status import ServerStatus
from .training_client import TrainingClient


class Server:
    def __init__(self, model_params=None):
        self.model_params = model_params
        self.init_params()
        self.training_clients = {}
        self.status = ServerStatus.IDLE
        self.federated_learning_config = FederatedLearningConfig(learning_rate=1., epochs=20, batch_size=256)

    def init_params(self):
        if self.model_params is None:
            weights = torch.randn((28 * 28, 1), dtype=torch.float, requires_grad=True)
            bias = torch.randn(1, dtype=torch.float, requires_grad=True)
            self.model_params = weights, bias

    async def start_training(self):
        if self.status != ServerStatus.IDLE:
            print('Server is not ready for training yet, status:', self.status)
            for training_client in self.training_clients.values():
                print(training_client)
        elif len(self.training_clients) == 0:
            print("There aren't any clients registered in the system, nothing to do yet")
        else:
            request_body = model_params_to_request_params(self.model_params)
            request_body['learning_rate'] = self.federated_learning_config.learning_rate
            request_body['epochs'] = self.federated_learning_config.epochs
            request_body['batch_size'] = self.federated_learning_config.batch_size
            print('There are', len(self.training_clients), 'clients registered')
            tasks = []
            for training_client in self.training_clients.values():
                tasks.append(asyncio.ensure_future(self.do_training_client_request(training_client, request_body)))
            print('Requesting training to clients...')
            self.status = ServerStatus.CLIENTS_TRAINING
            await asyncio.gather(*tasks)
        sys.stdout.flush()

    @staticmethod
    async def do_training_client_request(training_client, request_body):
        request_url = training_client.client_url + '/training'
        print('Requesting training to client', request_url)
        async with aiohttp.ClientSession() as session:
            training_client.status = ClientTrainingStatus.TRAINING_REQUESTED
            async with session.post(request_url, json=request_body) as response:
                if response.status != 200:
                    print('Error requesting training to client', training_client.client_url)
                    training_client.status = ClientTrainingStatus.TRAINING_REQUEST_ERROR
                else:
                    print('Client', training_client.client_url, 'started training')

    def update_client_model_params(self, training_client, client_model_params):
        print('New model params received from client', training_client.client_url)
        training_client.model_params = client_model_params
        training_client.status = ClientTrainingStatus.TRAINING_FINISHED
        if self.are_all_client_model_params_received():
            self.update_server_model_params()

    def update_server_model_params(self):
        print('Updating global model params')
        self.status = ServerStatus.UPDATING_MODEL_PARAMS
        received_weights = []
        received_biases = []
        for training_client in self.training_clients.values():
            if training_client.status == ClientTrainingStatus.TRAINING_FINISHED:
                received_weights.append(training_client.model_params[0])
                received_biases.append(training_client.model_params[1])
                training_client.status = ClientTrainingStatus.IDLE

        new_weights = torch.stack(received_weights).mean(0)
        new_bias = torch.stack(received_biases).mean(0)
        self.model_params = new_weights, new_bias
        self.status = ServerStatus.IDLE
        sys.stdout.flush()

    def are_all_client_model_params_received(self):
        for training_client in self.training_clients.values():
            if training_client.status != ClientTrainingStatus.TRAINING_FINISHED \
                    and training_client.status != ClientTrainingStatus.TRAINING_REQUEST_ERROR:
                return False
        return True

    def register_client(self, client_url):
        print('Registering new training client [', client_url, ']')
        if self.training_clients.get(client_url) is None:
            self.training_clients[client_url] = TrainingClient(client_url)
            print('Client [', client_url, '] registered successfully')
        else:
            print('Client [', client_url, '] was already registered in the system')
        sys.stdout.flush()

    def unregister_client(self, client_url):
        print('Unregistering client [', client_url, ']')
        try:
            self.training_clients.pop(client_url)
            print('Client [', client_url, '] unregistered successfully')
        except KeyError:
            print('Client [', client_url, '] is not registered yet')
        sys.stdout.flush()

    def are_clients_ready_for_training(self):
        for training_client in self.training_clients.values():
            if training_client.status == ClientTrainingStatus.IDLE:
                return True
        return False

