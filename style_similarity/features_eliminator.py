import os
import numpy as np
from config import EMBEDDINGS_DIR


X = np.array([np.load(os.path.join(EMBEDDINGS_DIR, fname), allow_pickle=True) for fname in os.listdir(EMBEDDINGS_DIR)])
X_t = X.transpose()
average_features = [x.sum()/x.shape[0] for x in X_t]
null_features = [i for i, af in enumerate(average_features) if af == 0]
print(X.shape[1]-len(null_features))
X = np.delete(X, [null_features], 1)
[np.save(os.path.join('encoded_' + EMBEDDINGS_DIR, fname), x) for x, fname in zip(X, os.listdir(EMBEDDINGS_DIR))]
np.save(os.path.join('null_features'), null_features)
print('done')
