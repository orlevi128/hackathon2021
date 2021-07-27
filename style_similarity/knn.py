import os

import numpy as np

from config import K
from embedder import Embedder


def compute_distance(embedding_1, embedding_2):
    return np.linalg.norm(embedding_1 - embedding_2)


def main():
    embd_dir = 'embeddings_with_style'
    input_embedding = Embedder.with_style('input.png')
    data_file_names = [file_name for file_name in os.listdir(embd_dir)]
    data_embeddings = {file_name: np.load(os.path.join(embd_dir, file_name), allow_pickle=True) for file_name in
                       data_file_names}
    distances = {file_name: compute_distance(input_embedding, data_embeddings[file_name]) for file_name in
                 data_file_names}

    recommended_file_names = sorted(data_file_names, key=lambda file_name: distances[file_name])[:K]
    for file_name in recommended_file_names:
        print(os.path.splitext(file_name)[0])


if __name__ == '__main__':
    main()
