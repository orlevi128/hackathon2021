from keras.preprocessing import image
import numpy as np
import os

input_images_dir = './images'
input_embeddings_dir = './embeddings_with_style'
output_dir = './embeddings_with_color'

color_cube_size = 8
for embedding_file_name in os.listdir(input_embeddings_dir):
    image_file_name = os.path.splitext(embedding_file_name)[0]
    embedding = np.load(os.path.join(input_embeddings_dir, embedding_file_name))
    histograms = np.zeros([int(256 / color_cube_size)] * 3)
    img = image.load_img(os.path.join(input_images_dir, image_file_name), target_size=(224, 224))
    x = image.img_to_array(img)
    for i, j in zip(range(224), range(224)):
        color = x[i][j]
        hist_i, hist_j, hist_k = list(map(lambda a: int(a / color_cube_size), color))
        histograms[hist_i][hist_j][hist_k] += 1
    result = np.concatenate([embedding.flatten(), histograms.flatten()])
    np.save(os.path.join(output_dir, embedding_file_name), result)
