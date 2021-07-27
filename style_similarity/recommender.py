import os
import numpy as np
from PIL import Image
from config import K, EMBEDDINGS_DIR
from embedder import Embedder


def compute_distance(embedding_1, embedding_2):
    return np.linalg.norm(embedding_1 - embedding_2)


def compute_cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def IOU(s1, s2):
    return len((s1.intersection(s2))) / len((s1.union(s2)))


def match_recommendations(image_path, k):
    input_embedding = Embedder.embed(image_path)
    data_file_names = [vec_name for vec_name in os.listdir(EMBEDDINGS_DIR)]
    data_embeddings = {vec_name: np.load(os.path.join(EMBEDDINGS_DIR, vec_name), allow_pickle=True)
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

    print('IOU {}'.format(IOU(set(d_recommended_file_names), set(d_recommended_file_names))))


def main():
    # embd_dir = 'embeddings_with_style'
    # input_embedding = Embedder.with_style('input.png')
    # data_file_names = [file_name for file_name in os.listdir(embd_dir)]
    # data_embeddings = {file_name: np.load(os.path.join(embd_dir, file_name), allow_pickle=True) for file_name in
    #                    data_file_names}
    # distances = {file_name: compute_distance(input_embedding, data_embeddings[file_name]) for file_name in
    #              data_file_names}
    #
    # recommended_file_names = sorted(data_file_names, key=lambda file_name: distances[file_name])[:K]
    # for file_name in recommended_file_names:
    #     print(os.path.splitext(file_name)[0])
    #     Image.open(os.path.join('images', os.path.splitext(file_name)[0])).show()
    image_path = 'img1.png'
    match_recommendations(image_path)


if __name__ == '__main__':
    main()
