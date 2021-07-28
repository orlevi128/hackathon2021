import os
import numpy as np
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.preprocessing import image

from config import EMBEDDING_LAYER_NAME, INPUT_SIZE, COLOR_CUBE_SIZE


class Embedder:
    base_model = VGG19(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer(EMBEDDING_LAYER_NAME).output)

    @classmethod
    def embed_naive(cls, image_path):
        img = image.load_img(image_path, target_size=INPUT_SIZE)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return cls.model.predict(x)

    @classmethod
    def embed_style(cls, input_embedding):
        num_of_kernels = input_embedding.shape[0]
        gram_matrix = np.zeros((num_of_kernels, num_of_kernels))
        for i, j in zip(range(num_of_kernels), range(num_of_kernels)):
            gram_matrix[i][j] = np.dot(input_embedding[i].flatten(), input_embedding[j].flatten())
        return gram_matrix

    @classmethod
    def embed_color(cls, image_path):
        histograms = np.zeros([int(256 / COLOR_CUBE_SIZE)] * 3)
        img = image.load_img(image_path, target_size=INPUT_SIZE)
        x = image.img_to_array(img)
        for i, j in zip(range(INPUT_SIZE[0]), range(INPUT_SIZE[1])):
            color = x[i][j]
            hist_i, hist_j, hist_k = list(map(lambda a: int(a / COLOR_CUBE_SIZE), color))
            histograms[hist_i][hist_j][hist_k] += 1
        return histograms

    @classmethod
    def embed(cls, image_path):
        return np.concatenate([cls.embed_style(np.swapaxes(cls.embed_naive(image_path), 0, 3)).flatten(),
                               cls.embed_color(image_path).flatten()])
