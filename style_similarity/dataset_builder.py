import os
import pathlib
import numpy as np
from config import IMAGES_DIR, EMBEDDINGS_DIR
from embedder import Embedder

naive_embeddings_dir = pathlib.Path('naive_' + EMBEDDINGS_DIR)
style_embeddings_dir = pathlib.Path('style_' + EMBEDDINGS_DIR)
color_embeddings_dir = pathlib.Path('color_' + EMBEDDINGS_DIR)
embeddings_dir = pathlib.Path(EMBEDDINGS_DIR)
if (not naive_embeddings_dir.exists() or
    not style_embeddings_dir.exists() or
    not color_embeddings_dir.exists() or
    not embeddings_dir.exists()):
    naive_embeddings_dir.mkdir()
    style_embeddings_dir.mkdir()
    color_embeddings_dir.mkdir()
    embeddings_dir.mkdir()


def main():
    # TODO: add progress bar
    for filename in os.listdir(IMAGES_DIR):
        print(f'Embedding {filename}')
        image_path = os.path.join(IMAGES_DIR, filename)
        naive_e = Embedder.embed_naive(image_path)
        np.save(str(naive_embeddings_dir.joinpath(filename)), naive_e)
        styl_e = Embedder.embed_style(np.swapaxes(naive_e, 0, 3))
        np.save(str(style_embeddings_dir.joinpath(filename)), styl_e)
        color_e = Embedder.embed_color(image_path)
        np.save(str(color_embeddings_dir.joinpath(filename)), color_e)
        final_e = np.concatenate([styl_e.flatten(), color_e.flatten()])
        np.save(str(embeddings_dir.joinpath(filename)), final_e)


if __name__ == '__main__':
    main()
