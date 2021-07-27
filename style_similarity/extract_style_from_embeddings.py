import numpy as np
import os

input_dir = './embeddings'
output_dir = './embeddings_with_style'
input_file_names = [file_name for file_name in os.listdir(input_dir)]
input_embeddings = {file_name:
                        np.swapaxes(np.load(os.path.join(input_dir, file_name)), 0, 3)
                    for file_name in input_file_names}

for file_name in input_embeddings:
    input_embedding = input_embeddings[file_name]
    num_of_kernels = input_embedding.shape[0]
    gram_matrix = np.zeros((num_of_kernels, num_of_kernels))
    for i, j in zip(range(num_of_kernels), range(num_of_kernels)):
        gram_matrix[i][j] = np.dot(input_embedding[i].flatten(), input_embedding[j].flatten())
    np.save(os.path.join(output_dir, file_name), gram_matrix)
