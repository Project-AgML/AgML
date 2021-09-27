from agml.backend.tftorch import torch

def set_seed(seed = None):
    """Sets a new random seed. If None, uses a random seed."""
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.random.manual_seed(seed)

