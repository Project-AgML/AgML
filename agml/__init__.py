import os

from . import backend
from .data.loader import AgMLDataLoader
from .data.public import public_data_sources
from . import data, backend, viz

os.system('agml/_helios/helios_config.sh')