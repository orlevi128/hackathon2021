from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
import os

input_dir = './images'
output_dir = './embeddings'
layer_name = 'block4_pool'

base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)

for filename in os.listdir(input_dir):
    try:
        img = image.load_img(os.path.join(input_dir, filename), target_size=(224, 224))
    except OSError as e:
        print(f'ERROR: {e}')
        continue

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    y = model.predict(x)

    np.save(os.path.join(output_dir, filename), y)
