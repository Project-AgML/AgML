import os
import re
import json
import functools

@functools.lru_cache(maxsize = None)
def load_public_sources():
    """Loads the public data sources JSON file."""
    with open(os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            '_assets/public_datasources.json')) as f:
        return json.load(f)

def to_camel_case(s):
    """Converts a given string `s` to camel case."""
    s = re.sub(r"(_|-)+", " ", s).title().replace(" ", "") # noqa
    return ''.join(s)

def resolve_list_value(l):
    """Determines whether a list contains one or multiple values."""
    if len(l) == 1:
        return l[0]
    return l

def resolve_tuple_values(*inputs, custom_error = None):
    """Determines whether `inputs[0]` contains two values or
    they are distributed amongst the values in `inputs`. """
    if isinstance(inputs[0], (list, tuple)) and all(c is None for c in inputs[1:]):
        if len(inputs[0]) != len(inputs):
            # special case for COCO JSON
            if len(inputs) == 3 and len(inputs[0]) == 2 and isinstance(inputs[0][1], dict):
                return inputs[0][0], inputs[0][1]['bboxes'], inputs[0][1]['labels']
            if custom_error is not None:
                raise ValueError(custom_error)
            else:
                raise ValueError(
                    f"Expected either a tuple with {len(inputs)} values "
                    f"or {len(inputs)} values across two arguments.")
        else:
            return inputs[0]
    return inputs

def as_scalar(inp):
    """Converts an input value to a scalar."""
    if isinstance(inp, (int, float)):
        return inp
    import numpy as np
    if isinstance(inp, np.ndarray):
        return inp.item()
    from agml.backend.tftorch import torch
    if isinstance(inp, torch.Tensor):
        return inp.item()
    from agml.backend.tftorch import tf
    if isinstance(inp, tf.Tensor):
        return inp.numpy()
    raise TypeError(f"Unsupported variable type {type(inp)}.")

def scalar_unpack(inp):
    """Unpacks a 1-d array into a list of scalars."""
    return [as_scalar(item) for item in inp]
