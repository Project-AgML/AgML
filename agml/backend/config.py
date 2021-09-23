import os
import sys

# The base save directory for AgML.
BASE_SAVE_DIR = os.path.join(os.path.expanduser('~'), '.agml')

def default_data_save_path():
    """Builds the default dataset save path for AgML."""
    global BASE_SAVE_DIR
    base_dir = os.path.join(BASE_SAVE_DIR, 'datasets')
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    return base_dir

