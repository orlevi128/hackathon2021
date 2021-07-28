import os
import numpy as np
from PIL import Image
from config import K, EMBEDDINGS_DIR
from embedder import Embedder


def compute_distance(v1, v2):
    return np.linalg.norm(v1 - v2)


def compute_cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def IOU(s1, s2):
    return len((s1.intersection(s2))) / len((s1.union(s2)))


def match_recommendations(image_path, k):
    input_embedding = Embedder.embed(image_path)
    null_features = np.load('null_features.npy', allow_pickle=True)
    input_embedding = np.delete(input_embedding, [null_features], 0)
    data_file_names = [vec_name for vec_name in os.listdir('encoded_' +EMBEDDINGS_DIR)]
    data_embeddings = {vec_name: np.load(os.path.join('encoded_' +EMBEDDINGS_DIR, vec_name), allow_pickle=True)
                       for vec_name in data_file_names}
    distances = {vec_name: compute_distance(input_embedding, data_embeddings[vec_name])
                 for vec_name in data_file_names}
    similarities = {vec_name: compute_cosine_similarity(input_embedding, data_embeddings[vec_name])
                    for vec_name in data_file_names}
    d_recommended_file_names = sorted(data_file_names, key=lambda file_name: distances[file_name])[:k]
    s_recommended_file_names = sorted(data_file_names, key=lambda file_name: -similarities[file_name])[:k]

    for file_name in d_recommended_file_names:
        print(os.path.splitext(file_name)[0])
        Image.open(os.path.join('images', os.path.splitext(file_name)[0])).show()

    for file_name in s_recommended_file_names:
        print(os.path.splitext(file_name)[0])
        Image.open(os.path.join('images', os.path.splitext(file_name)[0])).show()

    print('IOU {}'.format(IOU(set(d_recommended_file_names), set(s_recommended_file_names))))


def main():
    image_path = os.path.join('uploads', 'img1.png')
    match_recommendations(image_path, 10)


if __name__ == '__main__':
    main()
