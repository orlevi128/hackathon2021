from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
import os


def compute_distance(embedding_1, embedding_2):
    return np.linalg.norm(embedding_1 - embedding_2)


color_cube_size = 8

layer_name = 'block4_pool'
base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)

print('Input a path to an image:')
input_image_path = str(input())

print('Input the number of desired recommendations:')
k = int(input())

print('Should I use style-based recommendations? Input "yes" if so.')
should_use_style = str(input()) == 'yes'

should_use_colors = False
if should_use_style:
    print('Should I use color embeddings too? Input "yes" if so.')
    should_use_colors = str(input()) == 'yes'

img = image.load_img(input_image_path, target_size=(224, 224))
x = image.img_to_array(img)

histograms = np.zeros([int(256 / color_cube_size)] * 3)
for i, j in zip(range(224), range(224)):
    color = x[i][j]
    hist_i, hist_j, hist_k = list(map(lambda a: int(a / color_cube_size), color))
    histograms[hist_i][hist_j][hist_k] += 1

x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
input_embedding = model.predict(x)

if should_use_style:
    data_dir = './embeddings_with_style'
    input_embedding = np.swapaxes(input_embedding, 0, 3)
    num_of_kernels = input_embedding.shape[0]
    gram_matrix = np.zeros((num_of_kernels, num_of_kernels))
    for i, j in zip(range(num_of_kernels), range(num_of_kernels)):
        gram_matrix[i][j] = np.dot(input_embedding[i].flatten(), input_embedding[j].flatten())
    input_embedding = gram_matrix

    if should_use_colors:
        input_embedding = np.concatenate([input_embedding.flatten(), histograms.flatten()])
        data_dir = './embeddings_with_color'
else:
    data_dir = './embeddings'
data_file_names = [file_name for file_name in os.listdir(data_dir)]
data_embeddings = {file_name: np.load(os.path.join(data_dir, file_name), allow_pickle=True) for file_name in data_file_names}
distances = {file_name: compute_distance(input_embedding, data_embeddings[file_name]) for file_name in data_file_names}

recommended_file_names = sorted(data_file_names, key=lambda file_name: distances[file_name])[:k]
for file_name in recommended_file_names:
    print(os.path.splitext(file_name)[0])
