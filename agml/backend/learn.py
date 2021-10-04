from agml.backend.tftorch import torch, tf, get_backend

def set_seed(seed = None):
    """Sets a new random seed. If None, uses a random seed."""
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    if get_backend() == 'torch':
        torch.random.manual_seed(seed)
    elif get_backend() == 'tensorflow':
        tf.random.set_seed(seed)

