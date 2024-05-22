import random
import sys
import asyncio
import aiohttp
import torch

from .deterministic_mnist_model_trainer import DeterministicMnistModelTrainer
from .utils import request_params_to_model_params, model_params_to_request_params
from .training_type import TrainingType


class GossipMnistModelTrainer(DeterministicMnistModelTrainer):
    def __init__(self, model_params, client_config, client_id, round, round_size, peers):
        super().__init__(model_params, client_config, client_id, round, round_size)
        self.peers = peers

    def train_model(self):
        updated_model_params = super(DeterministicMnistModelTrainer, self).train_model()

        peer_model_params = asyncio.run(self.update_model_params_from_peers())
        if peer_model_params is None:
            return updated_model_params

        all_peer_weights = [params[0] for params in peer_model_params]
        all_peer_weights.append(updated_model_params[0])
        new_weights = torch.stack(all_peer_weights).mean(0)

        all_peer_biases = [params[1] for params in peer_model_params]
        all_peer_biases.append(updated_model_params[1])
        new_biases = torch.stack(all_peer_biases).mean(0)

        # Hack to turn weights and biases into leaf tensors
        new_params = request_params_to_model_params(
            TrainingType.GOSSIP_MNIST,
            model_params_to_request_params(
                TrainingType.GOSSIP_MNIST, (new_weights, new_biases)
            )
        )

        return new_params

    def __select_peers(self):
        peers_without_self = [peer for peer in self.peers if peer["client_id"] != self.client_id]

        if len(peers_without_self) < 4:
            return peers_without_self
        else:
            random_peers = random.sample(peers_without_self, int(random.uniform(4, len(peers_without_self))))
            return random_peers

    async def do_peer_model_params_request(self, peer):
        request_url = peer["client_url"] + '/model_params'

        async with aiohttp.ClientSession() as session:
            async with session.get(request_url) as response:
                if response.status != 200:
                    print('Error sending model params to client', peer["client_url"])
                    return None
                else:
                    print('Model params received from', peer["client_url"])
                    data = await response.json()
                    peer_model_params = data["model_params"]
                    peer_model_params = request_params_to_model_params(TrainingType.GOSSIP_MNIST, peer_model_params)
                    return peer_model_params

    async def update_model_params_from_peers(self):
        print('Requesting parameters from peers...')

        # Select random peers to request parameters from
        peers = self.__select_peers()
        if len(peers) == 0:
            print('No peers found')
        else:
            tasks = []
            for peer in peers:
                tasks.append(asyncio.ensure_future(self.do_peer_model_params_request(peer)))

            all_peer_model_params = await asyncio.gather(*tasks)
            peer_model_params = [params for params in all_peer_model_params if params is not None]
            return peer_model_params

        sys.stdout.flush()
