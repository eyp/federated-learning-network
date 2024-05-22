import os
import signal

from flask import Flask, request, Response, jsonify
from os import environ

from .client import Client
from .federated_learning_config import FederatedLearningConfig
from .utils import request_params_to_model_params, model_params_to_request_params
from .training_type import TrainingType

CLIENT_URL = environ.get('CLIENT_URL')
if CLIENT_URL is None:
    print("Error, CLIENT_URL environment variable must be defined. "
          "Example: export CLIENT_URL='http://127.0.0.1:5003' if client is running on port 5003")
    os.kill(os.getpid(), signal.SIGINT)

app = Flask(__name__)
client = Client(CLIENT_URL)


@app.route('/')
def index():
    return 'Federated Learning client running. Status: ' + client.status


@app.route('/training', methods=['POST'])
def training():
    training_type = request.json['training_type']
    print('Request POST /training for training type:', training_type)
    federated_learning_config = FederatedLearningConfig(request.json['learning_rate'],
                                                        request.json['epochs'],
                                                        request.json['batch_size'])
    model_params = request_params_to_model_params(training_type, request.json)
    client_id = request.json['client_id']
    round = request.json['round']
    round_size = request.json['round_size']
    clients = request.json.get('clients', None)
    client.do_training(training_type, model_params, federated_learning_config, client_id, round, round_size, clients)
    return Response(status=200)


@app.route('/model_params', methods=['GET'])
def get_model_params():
    model_params = model_params_to_request_params(TrainingType.GOSSIP_MNIST, client.model_params)
    response = jsonify({'model_params': model_params})
    response.status_code = 200
    return response


@app.errorhandler(404)
def page_not_found(error):
    return 'This page does not exist', 404
