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

"""
This part of the backend controls the AgML methods where either
TensorFlow or PyTorch methods can be used, and prevents unnecessary
importing of either library (which takes a significant amount of time).
"""
import types
import inspect
import logging
import importlib
import functools

from agml.utils.logging import log


# Suppress any irrelevant warnings which will pop up from either backend.
import warnings
warnings.filterwarnings(
    'ignore', category = UserWarning, message = '.*Named tensors.*Triggered internally.*')


# Custom errors.

class BackendError(ValueError):
    pass


class StrictBackendError(BackendError):
    def __init__(self, message = None, change = None, obj = None):
        if message is None:
            message = f"Backend was manually set to " \
                      f"'{get_backend()}', but got an object " \
                      f"from backend '{change}': {obj}."
        super(StrictBackendError, self).__init__(message)


# Check if TensorFlow and PyTorch exist in the environment.
_HAS_TENSORFLOW: bool
_HAS_TORCH: bool


@functools.lru_cache(maxsize = None)
def _check_tf_torch():
    global _HAS_TENSORFLOW, _HAS_TORCH
    try:
        import tensorflow
    except ImportError:
        _HAS_TENSORFlOW = False
    else:
        _HAS_TENSORFLOW = True
    try:
        import torch
    except ImportError:
        _HAS_TORCH = False
    else:
        _HAS_TORCH = True


# Default backend is PyTorch.
_BACKEND = 'torch'
_USER_SET_BACKEND = False


def get_backend():
    """Returns the current AgML backend."""
    return _BACKEND


def set_backend(backend):
    """Change the AgML backend for the current session.

    By default, AgML uses PyTorch as a backend, but it is
    compatible with both TensorFlow and PyTorch. AgML can
    also automatically inference the backend from the
    different parameters passed into `AgMLDataLoader` and
    other internal library methods.

    This method allows a user to automatically set the backend.
    """
    global _USER_SET_BACKEND, _BACKEND
    # Check whether the user has modified the backend.
    mod = inspect.getmodule(inspect.stack()[1][0])
    if 'agml.' not in mod.__name__:
        _USER_SET_BACKEND = True

    # If the backend is the same, don't do anything.
    if backend == _BACKEND:
        return

    _check_tf_torch()
    if backend not in ['tensorflow', 'tf', 'torch', 'pytorch']:
        raise ValueError(f"Invalid backend: {backend}.")
    if backend in ['tensorflow', 'tf'] and _BACKEND != 'tensorflow':
        if not _HAS_TENSORFLOW:
            raise ImportError(
                "TensorFlow not found on system, cannot be used as "
                "backend. Try running `pip install tensorflow`.")
        _BACKEND = 'tf'
        log("Switched backend to TensorFlow.", level = logging.INFO)
    elif backend in ['torch', 'pytorch'] and _BACKEND != 'torch':
        if not _HAS_TORCH:
            raise ImportError(
                "PyTorch not found on system, cannot be used as "
                "backend. Try running `pip install torch`.")
        _BACKEND = 'torch'
        log("Switched backend to PyTorch.", level = logging.INFO)


def user_changed_backend():
    """Returns whether the backend has been manually changed."""
    return _USER_SET_BACKEND


# Ported from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/util/lazy_loader.py
class LazyLoader(types.ModuleType):
  """Lazily import a module, mainly to avoid pulling in large dependencies.  """
  def __init__(self, local_name, parent_module_globals, name):
    self._local_name = local_name
    self._parent_module_globals = parent_module_globals
    super(LazyLoader, self).__init__(name)

  def _load(self):
    """Load the module and insert it into the parent's globals."""
    # Import the target module and insert it into the parent's namespace.
    module = importlib.import_module(self.__name__)
    self._parent_module_globals[self._local_name] = module

    # Update this object's dict so that if someone keeps a reference to the
    #   LazyLoader, lookups are efficient (__getattr__ is only called on lookups
    #   that fail).
    self.__dict__.update(module.__dict__)
    return module

  def __getattr__(self, item):
    module = self._load()
    return getattr(module, item)

  def __dir__(self):
    module = self._load()
    return dir(module)


# Load TensorFlow and PyTorch lazily to prevent pulling them in when unnecessary.
torch = LazyLoader('torch', globals(), 'torch')
torch_data = LazyLoader('torch_data', globals(), 'torch.utils.data')
torchvision = LazyLoader('torchvision', globals(), 'torchvision')
tf = LazyLoader('tensorflow', globals(), 'tensorflow')


######### GENERAL METHODS #########

def _convert_image_to_torch(image):
    """Converts an image (np.ndarray) to a torch Tensor."""
    if isinstance(image, (list, tuple)):
        return torch.tensor(image)
    if isinstance(image, torch.Tensor) or image.ndim == 4:
        return image
    if image.shape[0] > image.shape[-1]:
        return torch.from_numpy(image).permute(2, 0, 1).float()
    return torch.from_numpy(image)


def _postprocess_torch_annotation(image):
    """Post-processes a spatially augmented torch annotation."""
    try:
        if image.dtype.is_floating_point:
            image = (image * 255).int()
    except AttributeError:
        pass
    return image


######### AGMLDATALOADER METHODS #########

class AgMLObject(object):
    """Base class for the `AgMLDataLoader` to enable inheritance.

    This class solves a bug which arises when trying to dynamically
    inherit from `tf.keras.utils.Sequence` and/or `torch.utils.data.Dataset`.
    The fact that the `AgMLDataLoader` has this `AgMLObject` as a subclass
    enables it to be able to handle dynamic inheritance. This is the sole
    purpose of this subclass, it does not have any features.
    """


def _add_dataset_to_mro(inst, mode):
    """Adds the relevant backend class to the `AgMLDataLoader` MRO.

    This allows for the loader to dynamically inherent from the
    `tf.keras.utils.Sequence` and `torch.utils.data.Dataset`.
    """
    if mode == 'tf':
        if not get_backend() == 'tf':
            if user_changed_backend():
                raise StrictBackendError(change = 'tf', obj = inst)
            set_backend('tf')
        if tf.keras.utils.Sequence not in inst.__class__.__bases__:
            inst.__class__.__bases__ += (tf.keras.utils.Sequence, )
    if mode == 'torch':
        if not get_backend() == 'torch':
            if user_changed_backend():
                raise StrictBackendError(change = 'torch', obj = inst)
        if torch_data.Dataset not in inst.__class__.__bases__:
            inst.__class__.__bases__ += (torch_data.Dataset,)


