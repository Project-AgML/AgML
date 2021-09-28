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
    global _USER_SET_BACKEND
    # Check whether the user has modified the backend.
    mod = inspect.getmodule(inspect.stack()[1][0])
    if 'agml.' not in mod.__name__:
        _USER_SET_BACKEND = True

    # If the backend is the same, don't do anything.
    if backend == _USER_SET_BACKEND:
        return

    global _BACKEND
    _check_tf_torch()
    if backend not in ['tensorflow', 'tf', 'torch', 'pytorch']:
        raise ValueError(f"Invalid backend: {backend}.")
    if backend in ['tensorflow', 'tf'] and _BACKEND != 'tensorflow':
        if not _HAS_TENSORFLOW:
            raise ImportError(
                "TensorFlow not found on system, cannot be used as "
                "backend. Try running `pip install tensorflow`.")
        _BACKEND = 'tensorflow'
        log("Switched backend to TensorFlow.")
    elif backend in ['torch', 'pytorch'] and _BACKEND != 'torch':
        if not _HAS_TORCH:
            raise ImportError(
                "PyTorch not found on system, cannot be used as "
                "backend. Try running `pip install torch`.")
        _BACKEND = 'torch'
        log("Switched backend to PyTorch.", level = logging.INFO)
    _set_global_torch_tf_datasets()

def _was_backend_changed():
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
    return torch.from_numpy(image).long()

######### AGMLDATALOADER METHODS #########

class _DummyTorchDataset(object):
    pass

class _DummyTFSequenceDataset(object):
    pass

TorchDataset = _DummyTorchDataset
TFSequenceDataset = _DummyTFSequenceDataset

def _set_global_torch_tf_datasets():
    """Sets the global dataset classes for the data loader.

    If PyTorch (default) is chosen as the backend, then it sets
    the `TorchDataset` class to `torch.utils.data.Dataset`.
    Otherwise, if TensorFlow is chosen as the backend, it sets
    `TFSequenceDataset` to `tf.keras.utils.Sequence`.
    """
    global TorchDataset, TFSequenceDataset
    if get_backend() == 'tensorflow':
        TFSequenceDataset = tf.keras.utils.Sequence
        TorchDataset = _DummyTorchDataset
    elif get_backend() == 'torch':
        TorchDataset = torch_data.Dataset
        TFSequenceDataset = _DummyTFSequenceDataset

# Run upon loading the backend.
_set_global_torch_tf_datasets()

######### `AgMLImageClassificationDataLoader.transform()` #########

def _check_image_classification_transform(transform):
    """Check the image classification transform pipeline."""
    if transform is None:
        return None
    if isinstance(transform, types.FunctionType): # noqa
        return transform
    if get_backend() == 'tensorflow':
        _tf_check_sequential_preprocessing_pipeline(transform)
        return transform
    elif get_backend() == 'torch':
        return _torch_check_torchvision_preprocessing_pipeline(transform)
    else:
        raise TypeError(
            "Got unknown preprocessing transform (not a method, "
            f"or a TensorFlow/PyTorch preprocessing pipeline): {transform}.")

def _tf_check_sequential_preprocessing_pipeline(model):
    """Checks that a Sequential model passed to a dataset is valid."""
    if not isinstance(model, tf.keras.models.Sequential):
        if 'torchvision' in model.__module__:
            if _was_backend_changed():
                raise TypeError(
                    "Backend was manually set to `tensorflow`, "
                    "but got a PyTorch transformation.")
            else:
                log("Switching backend to PyTorch: got a torchvision "
                    "transform while backend was set to TensorFlow.")
                set_backend('torch')
                return _torch_check_torchvision_preprocessing_pipeline(model)
        else:
            raise TypeError(
                f"Unknown preprocessing object {model} of type {type(model)}. "
                f"Backend is currently `tensorflow`, but object is neither "
                f"a function or a torchvision transform.")
    for layer in model.layers:
        if 'keras' not in layer.__module__ and 'preprocessing' not in layer.__module__:
            raise TypeError(
                "Expected only preprocessing layers in "
                f"Sequential preprocessing model: got {type(layer)}.")
    return model

def _torch_check_torchvision_preprocessing_pipeline(transforms):
    """Checks that a torchvision transform pipeline is valid."""
    if 'torchvision' not in transforms.__module__:
        if isinstance(transforms, tf.keras.models.Sequential):
            if _was_backend_changed():
                raise TypeError(
                    "Backend was manually set to `torch`, but got"
                    "a TensorFlow preprocessing model.")
            else:
                log("Switching backend to TensorFlow: got a TensorFlow "
                    "preprocessing model while backend was set to PyTorch.")
    return transforms


######### `AgMLSemanticSegmentationDataLoader.transform()` #########

def _check_semantic_segmentation_transform(transform, target_transform, dual_transform):
    """Checks the semantic segmentation transform pipeline."""
    if transform is None and target_transform is None and dual_transform is None:
        return None, None, None
    if dual_transform is not None:
        _check_image_classification_transform(dual_transform)
        if isinstance(transform, types.FunctionType) and target_transform is None:  # noqa
            _check_function_type_transform_segmentation(transform)
    else:
        old_backend = get_backend()
        _check_image_classification_transform(transform)
        current_backend = get_backend()
        _check_image_classification_transform(target_transform)
        new_backend = get_backend()
        if old_backend == new_backend and old_backend != current_backend:
            raise ValueError(
                "Transform and target transform use methods from different backends.")
        return transform, target_transform

def _check_function_type_transform_segmentation(transform):
    if len(inspect.signature(transform)) != 2:  # noqa
        raise ValueError(
            "If passing a function for `transform`, it should accept "
            "two arguments, the input image and annotation.")
    return transform
