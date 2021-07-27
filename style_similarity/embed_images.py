import os
import pathlib

import numpy as np

from config import IMAGES_DIR, EMBEDDINGS_DIR
from embedder import Embedder


def main():
    print('Starting')

    embeddings_dir = pathlib.Path(EMBEDDINGS_DIR)
    if not embeddings_dir.exists():
        embeddings_dir.mkdir()

    for filename in os.listdir(IMAGES_DIR):
        print(f'Embedding {filename}')
        vec = Embedder.run(os.path.join(IMAGES_DIR, filename))

        np.save(str(embeddings_dir.joinpath(filename)), vec)


if __name__ == '__main__':
    main()
