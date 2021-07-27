import os

import numpy as np
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.preprocessing import image

from config import EMBEDDING_LAYER_NAME


class Embedder:
    base_model = VGG19(weights='imagenet')
    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer(EMBEDDING_LAYER_NAME).output)

    @classmethod
    def run(cls, image_path: str):
        img = image.load_img(os.path.join(image_path), target_size=(224, 224))

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return cls.model.predict(x)

    @classmethod
    def with_style(cls, image_path: str):
        input_embedding = cls.run(image_path)
        num_of_kernels = input_embedding.shape[0]
        gram_matrix = np.zeros((num_of_kernels, num_of_kernels))
        for i, j in zip(range(num_of_kernels), range(num_of_kernels)):
            gram_matrix[i][j] = np.dot(input_embedding[i].flatten(), input_embedding[j].flatten())
        return gram_matrix
