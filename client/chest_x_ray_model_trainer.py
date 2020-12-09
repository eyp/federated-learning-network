import glob
import os
import random
import shutil
import tempfile

from .config import GLOBAL_TMP_PATH, GLOBAL_DATASETS
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class ChestXRayModelTrainer:
    def __init__(self, model_params, client_config):
        print('Initializing ChestXRayModelTrainer...')
        self.client_config = client_config
        self.model_params = model_params
        self.temp_folder = None
        current_directory = os.path.dirname(os.path.realpath(__file__))
        self.chest_x_ray_temp_folder = current_directory + GLOBAL_TMP_PATH + '/chest_xray/'

    def train_model(self):
        model = Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
            MaxPool2D(pool_size=(2, 2), strides=2),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPool2D(pool_size=(2, 2), strides=2),
            Flatten(),
            Dense(units=2, activation='softmax')
        ])
        model.summary()

        model.compile(optimizer=Adam(learning_rate=self.client_config.learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        if self.model_params is not None:
            print('Using model weights from central node')
            model.set_weights(self.model_params)
        else:
            print('Using default model weights')

        self.__create_temp_dataset_folder()
        train_batches, valid_batches = self.__load_datasets()

        model.fit(x=train_batches,
                  steps_per_epoch=10,
                  epochs=self.client_config.epochs,
                  validation_data=valid_batches,
                  validation_steps=5,
                  verbose=2)

        self.__clean_temp_dataset_folder()
        return model.get_weights()

    def __load_datasets(self):
        print('Loading CHEST X-RAY IMAGES dataset...')
        global_dataset_train_path = GLOBAL_DATASETS + '/chest_xray/train'
        global_dataset_valid_path = GLOBAL_DATASETS + '/chest_xray/test'

        training_dataset_train_path = self.temp_folder.name + "/train"
        training_dataset_valid_path = self.temp_folder.name + "/val"

        os.makedirs(training_dataset_train_path + '/NORMAL')
        os.makedirs(training_dataset_train_path + '/PNEUMONIA')
        os.makedirs(training_dataset_valid_path + '/NORMAL')
        os.makedirs(training_dataset_valid_path + '/PNEUMONIA')

        self.__build_training_dataset(global_dataset_train_path, training_dataset_train_path, ['NORMAL', 'PNEUMONIA'], 100)
        self.__build_training_dataset(global_dataset_valid_path, training_dataset_valid_path, ['NORMAL', 'PNEUMONIA'], 50)

        image_data_generator = ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input)

        train_batches = image_data_generator.flow_from_directory(
            directory=training_dataset_train_path, target_size=(224, 224), classes=['PNEUMONIA', 'NORMAL'],
            batch_size=self.client_config.batch_size)

        valid_batches = image_data_generator.flow_from_directory(
            directory=training_dataset_valid_path, target_size=(224, 224), classes=['PNEUMONIA', 'NORMAL'],
            batch_size=self.client_config.batch_size)

        return train_batches, valid_batches

    def __create_temp_dataset_folder(self):
        if os.path.isdir(self.chest_x_ray_temp_folder) is False:
            print('Temporary dataset folder', self.chest_x_ray_temp_folder, 'doesn\'t exist, creating it')
            os.mkdir(self.chest_x_ray_temp_folder)

        self.temp_folder = tempfile.TemporaryDirectory(dir=self.chest_x_ray_temp_folder)
        print('Temporary folder for training:', self.temp_folder.name)

    def __clean_temp_dataset_folder(self):
        print('Deleting content of temporary folder', self.temp_folder.name)
        self.temp_folder.cleanup()
        if os.path.isdir(self.chest_x_ray_temp_folder):
            path, dirs, files = next(os.walk(self.chest_x_ray_temp_folder))
            if len(dirs) == 0:
                print('Deleting temporary dataset folder', self.chest_x_ray_temp_folder)
                os.rmdir(self.chest_x_ray_temp_folder)

    def __build_training_dataset(self, global_dataset_path, training_dataset_path, classes, samples_size, pattern='*'):
        for a_class in classes:
            for random_file in random.sample(glob.glob(global_dataset_path + '/' + a_class + '/' + pattern), samples_size):
                shutil.copy(random_file, training_dataset_path + '/' + a_class)


chestXRayModelTrainer = ChestXRayModelTrainer(None, None)
