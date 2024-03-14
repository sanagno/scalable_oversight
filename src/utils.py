import pickle
import random

import numpy as np
import torch
import transformers


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def set_seed(seed=0):
    # set the random seed
    random.seed(seed)
    transformers.set_seed(seed)
    # just to be safe
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
