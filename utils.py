import random
import numpy as np
import torch

# 1) Seed all RNGs
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# 2) Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 3) Clip gradients
def clip_grad(params, max_norm: float = 1.0):
    torch.nn.utils.clip_grad_norm_(params, max_norm)