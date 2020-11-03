import asyncio

from flask import Flask, request, Response

from .utils import request_params_to_model_params
from .server import Server

app = Flask(__name__)
server = Server()


@app.route('/')
def index():
    return 'Federated Learning server running. Status: ' + server.status


@app.route('/training', methods=['GET'])
def training():
    asyncio.run(server.start_training())
    return 'Training started'


@app.route('/client', methods=['POST'])
def register_client():
    print('Registering client:', request.form['client_url'])
    server.register_client(request.form['client_url'])
    return Response(status=201)


@app.route('/model_params', methods=['PUT'])
def update_weights():
    print('Updating model params from client:', request.json['client_url'])
    try:
        training_client = server.training_clients[request.json['client_url']]
        server.update_client_model_params(training_client, request_params_to_model_params(request.json))
        return Response(status=200)
    except KeyError:
        print('Client', request.json['client_url'], 'is not registered in the system')
        return Response(status=401)


@app.errorhandler(404)
def page_not_found(error):
    return 'This page does not exist', 404
