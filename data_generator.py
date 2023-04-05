import os

import cv2
import numpy as np
from tensorflow.keras.utils import Sequence


class ImageDataGenerator(Sequence):
    def __init__(
        self, input_dir, target_dir, batch_size, img_size, shuffle=True
    ):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.input_image_list = os.listdir(input_dir)
        self.target_image_list = os.listdir(target_dir)
        self.indexes = np.arange(len(self.input_image_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.input_image_list) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        input_images = [self.input_image_list[i] for i in indexes]
        target_images = [self.target_image_list[i] for i in indexes]
        X, y = self.__load_data(input_images, target_images)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __load_data(self, input_images, target_images):
        X = []
        y = []
        for input_img, target_img in zip(input_images, target_images):
            input_path = os.path.join(self.input_dir, input_img)
            target_path = os.path.join(self.target_dir, target_img)
            input_image = cv2.imread(input_path)[:, :, ::-1]
            target_image = cv2.imread(target_path)[:, :, ::-1]

            input_image = cv2.resize(input_image, self.img_size) / 255.0
            target_image = cv2.resize(target_image, self.img_size) / 255.0

            X.append(input_image)
            y.append(target_image)

        return np.array(X), np.array(y)
