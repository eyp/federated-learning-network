import asyncio
import os

from flask import (
    Flask, Response, request, render_template
)

from .server import Server
from .utils import request_params_to_model_params


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    server = Server()

    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'fl-network.sqlite'),
    )
    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/')
    def index():
        clients_ready_for_training = server.can_do_training()
        return render_template("index.html",
                               server_status=server.status,
                               training_clients=server.training_clients,
                               clients_ready_for_training=clients_ready_for_training)

    @app.route('/training', methods=['POST'])
    def training():
        print('Request POST /training')
        training_type = request.json['training_type']
        asyncio.run(server.start_training(training_type))
        return Response(status=200)

    @app.route('/client', methods=['POST'])
    def register_client():
        print('Request POST /client for client_url [', request.form['client_url'], ']')
        server.register_client(request.form['client_url'])
        return Response(status=201)

    @app.route('/client', methods=['DELETE'])
    def unregister_client():
        print('Request DELETE /client for client_url [', request.form['client_url'], ']')
        server.unregister_client(request.form['client_url'])
        return Response(status=200)

    @app.route('/model_params', methods=['PUT'])
    def update_weights():
        client_url = request.json['client_url']
        training_type = request.json['training_type']
        print('Request PUT /model_params for client_url [', client_url, '] and training type:', training_type)
        try:
            training_client = server.training_clients[request.json['client_url']]
            server.update_client_model_params(training_type, training_client, request_params_to_model_params(training_type, request.json))
            return Response(status=200)
        except KeyError:
            print('Client', client_url, 'is not registered in the system')
            return Response(status=401)

    @app.errorhandler(404)
    def page_not_found(error):
        return 'This page does not exist', 404

    return app
