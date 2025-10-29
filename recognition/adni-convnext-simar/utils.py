import random
import numpy as np
import torch

class AvgMeter:
    def __init__(self):
        self.sum = 0.0
        self.n = 0
    def update(self, val, k: int = 1):
        self.sum += float(val) * k
        self.n += k
    @property
    def avg(self):
        return self.sum / max(1, self.n)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
