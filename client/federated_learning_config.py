class FederatedLearningConfig:
    def __init__(self, learning_rate=1, epochs=10, batch_size=256):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    def __str__(self):
        return "Federated Learning config:\n--Learning Rate: {}\n--Epochs: {}\n--Batch size: {}\n".format(
            self.learning_rate,
            self.epochs,
            self.batch_size)
