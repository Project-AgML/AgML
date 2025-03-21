# Copyright 2021 UC Davis Plant AI and Biophysics Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import math
import re
from math import floor, pi

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Represents an empty object, but allows passing `None`
# as an independent object in certain cases.
NoArgument = object()


def placeholder(obj):
    """Equivalent of lambda x: x, but enables pickling."""
    return obj


def to_camel_case(s):
    """Converts a given string `s` to camel case."""
    s = re.sub(r"(_|-)+", " ", s).title().replace(" ", "")  # noqa
    return "".join(s)


def resolve_list_value(val):
    """Determines whether a list contains one or multiple values."""
    if len(val) == 1:
        return val[0]
    return val


def resolve_tuple_values(*inputs, custom_error=None):
    """Determines whether values are distributed amongst the values in `inputs`."""
    if isinstance(inputs[0], (list, tuple)) and all(c is None for c in inputs[1:]):
        if len(inputs[0]) != len(inputs):
            # special case for COCO JSON
            if len(inputs) == 3 and len(inputs[0]) == 2 and isinstance(inputs[0][1], dict):
                try:
                    return (
                        inputs[0][0],
                        inputs[0][1]["bbox"],
                        inputs[0][1]["category_id"],
                    )
                except KeyError:
                    return inputs[0][0], inputs[0][1]["bboxes"], inputs[0][1]["labels"]
            if custom_error is not None:
                raise ValueError(custom_error)
            else:
                raise ValueError(
                    f"Expected either a tuple with {len(inputs)} values "
                    f"or {len(inputs)} values across two arguments."
                )
        else:
            return inputs[0]
    return inputs


def resolve_tuple(sequence):
    """Resolves a sequence to a tuple."""
    if isinstance(sequence, np.ndarray):
        sequence = sequence.tolist()
    return tuple(i for i in sequence)


def has_nested_dicts(obj):
    """Returns whether a dictionary contains nested dicts."""
    return any(isinstance(i, dict) for i in obj.values())


def as_scalar(inp):
    """Converts an input value to a scalar."""
    if isinstance(inp, (int, float)):
        return inp
    if np.isscalar(inp):
        return inp.item()
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


def is_array_like(inp):
    """Determines if an input is a np.ndarray, torch.Tensor, or tf.Tensor."""
    if isinstance(inp, np.ndarray):
        return True
    from agml.backend.tftorch import torch

    if isinstance(inp, torch.Tensor):
        return True
    from agml.backend.tftorch import tf

    if isinstance(inp, tf.Tensor):
        return True
    return False


def shapes(seq):
    """Returns the shapes (or lengths) of all of the objects in the sequence."""
    try:
        return [getattr(obj, "shape", len(obj)) for obj in seq]
    except:
        raise ValueError(f"One or more of the objects has no shape or length: {seq}.")


def weak_squeeze(arr, ndims=2):
    """Performs a 'weak squeeze', adding a dimension back if necessary."""
    if isinstance(arr, np.ndarray):
        arr = np.squeeze(arr)
        while arr.ndim < ndims:
            arr = np.expand_dims(arr, axis=0)
    if isinstance(arr, list):
        while len(arr) < ndims:
            arr = [arr]
    return arr


def is_float(num):
    """Determines if a number is a float."""
    is_ = isinstance(num, float) or isinstance(num, np.float32) or isinstance(num, np.float64)
    if is_:
        return True
    try:
        float(num)
    except ValueError:
        return False
    return True


def is_int(num):
    """Determines if a number is an int."""
    is_ = isinstance(num, int) or isinstance(num, np.int32) or isinstance(num, np.int64)
    if is_:
        return True
    try:
        int(num)
    except ValueError:
        return False
    return True


def has_func(module, func):
    """Determines if a module has a function."""
    try:
        return True
    except ImportError:
        return False


def flatten(xss):
    """Flatten a list of lists"""
    return [x for xs in xss for x in xs]
