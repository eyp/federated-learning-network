from fastai.vision.all import *

from .training_utils import mnist_loss, linear_model


class MnistModelTrainer:
    def __init__(self, model_params, client_config):
        print('Initializing MnistModelTrainer...')
        self.ACCURACY_THRESHOLD = 0.5
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

    def __train_epoch(self):
        for train_data, train_labels in self.training_dataloader:
            self.__calculate_gradients(train_data, train_labels)
            for model_param in self.model_params:
                model_param.data -= model_param.grad * self.client_config.learning_rate
                model_param.grad.zero_()

    def __validate_epoch(self):
        accuracies = [self.__accuracy(linear_model(train_data, weights=self.model_params[0], bias=self.model_params[1]), train_labels) for
                      train_data, train_labels in
                      self.validation_dataloader]
        return round(torch.stack(accuracies).mean().item(), 4)

    def __calculate_gradients(self, train_data, train_labels):
        predictions = linear_model(train_data, self.model_params[0], self.model_params[1])
        loss = mnist_loss(predictions, train_labels)
        loss.backward()

    def __accuracy(self, train_data, train_labels):
        predictions = train_data.sigmoid()
        corrections = (predictions > self.ACCURACY_THRESHOLD) == train_labels
        return corrections.float().mean()

    def __load_datasets(self):
        print('Loading dataset MNIST_SAMPLE...')
        path = untar_data(URLs.MNIST_SAMPLE)
        print('Content of MNIST_SAMPLE:', path.ls())
        print("Content of 'train' directory of MNIST_SAMPLE", (path / 'train').ls())

        threes = random.sample((path / 'train' / '3').ls().sorted(), int(random.uniform(20, 30)))
        sevens = random.sample((path / 'train' / '7').ls().sorted(), int(random.uniform(20, 30)))

        three_tensors = [tensor(Image.open(image_path)) for image_path in threes]
        seven_tensors = [tensor(Image.open(image_path)) for image_path in sevens]

        print("There are', len(three_tensors), 'images of number 3 and', len(seven_tensors), 'of number 7\n")

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
        training_dataset = list(zip(train_images, train_labels))
        print('Training images shape:', train_images.shape, ', training labels shape:', train_labels.shape)

        valid_images = torch.cat([valid_three_tensors, valid_seven_tensors]).view(-1, 28 * 28)
        valid_labels = tensor([1] * len(valid_three_tensors) + [0] * len(valid_seven_tensors)).unsqueeze(1)
        validation_dataset = list(zip(valid_images, valid_labels))
        print('Dataset ready to be used')
        sys.stdout.flush()
        return training_dataset, validation_dataset

