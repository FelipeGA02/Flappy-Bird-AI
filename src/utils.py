import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def moving_average(x, w=50):
    if len(x) < w:
        return x
    out = []
    s = 0.0
    for i, v in enumerate(x):
        s += v
        if i >= w:
            s -= x[i - w]
        out.append(s / min(i + 1, w))
    return out
