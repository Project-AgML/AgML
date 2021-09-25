import inspect
import functools

def set_seed(seed = None):
    """Sets a new random seed. If None, uses a random seed."""
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
    torch.random.manual_seed(seed)

