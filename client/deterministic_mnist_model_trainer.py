from fastai.vision.all import *

from .mnist_model_trainer import MnistModelTrainer


class DeterministicMnistModelTrainer(MnistModelTrainer):
    def __init__(self, model_params, client_config, client_id, round, round_size):
        super().__init__(model_params, client_config)
        self.client_id = client_id
        self.round = round
        self.round_size = round_size

    def __get_tensors(self, path, number):
        train_sample_size = 25
        start_index = train_sample_size * self.client_id + (self.round - 1) * train_sample_size * self.round_size
        end_index = start_index + train_sample_size
        numbers = (path / 'train' / number).ls().sorted()[start_index:end_index]

        number_tensors = [tensor(Image.open(image_path)) for image_path in numbers]

        valid_sample_size = 15
        valid_start_index = valid_sample_size * self.client_id + (self.round - 1) * valid_sample_size * self.round_size
        valid_end_index = start_index + valid_sample_size
        valid_number_paths = (path / 'valid' / number).ls().sorted()[valid_start_index:valid_end_index]

        valid_number_tensors = torch.stack(
            [tensor(Image.open(valid_number_path)) for valid_number_path in valid_number_paths])
        valid_three_tensors = valid_number_tensors.float() / 255

        return number_tensors, numbers, valid_number_tensors
