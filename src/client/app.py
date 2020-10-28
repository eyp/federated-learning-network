import os
import signal

from flask import Flask, request, Response
from os import environ

from .client import Client
from src.common.federated_learning_config import FederatedLearningConfig
from src.common.utils import request_params_to_model_params

CLIENT_URL = environ.get('CLIENT_URL')
if CLIENT_URL is None:
    print("Error, CLIENT_URL environment variable must be defined. "
          "Example: export CLIENT_URL='http://127.0.0.1:5003' if client is running on port 5003")
    os.kill(os.getpid(), signal.SIGINT)

app = Flask(__name__)
client = Client(CLIENT_URL)


@app.route('/')
def index():
    return 'Federated Learning client running'


@app.route('/training', methods=['POST'])
def training():
    federated_learning_config = FederatedLearningConfig(request.json['learning_rate'],
                                                        request.json['epochs'],
                                                        request.json['batch_size'])
    client.do_training(request_params_to_model_params(request.json), federated_learning_config)
    return Response(status=200)
