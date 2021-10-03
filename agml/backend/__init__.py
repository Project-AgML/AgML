"""
Controls the AgML backend system.

The first part of the backend is the backend deep learning library.
The backend, one of {TensorFlow, PyTorch}, primarily exists for internal
purposes, e.g. figuring out which methods to use in the data module or
the actual model configuring/training modules.

Secondly, the backend controls the loading/saving procedure for files within
AgML, specifically data loaded or generated from the data module.
"""
from .config import (
    default_data_save_path, clear_all_datasets, downloaded_datasets
)
from .tftorch import get_backend, set_backend
from .learn import set_seed
