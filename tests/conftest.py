import os
from pathlib import Path
import shutil

import pytest

from pytest import TempPathFactory


def pytest_sessionstart(session):
    """
    Called before the whole run, right before
    returning the exit status to the system.
    """
    tmp_path_factory: TempPathFactory = session.config._tmp_path_factory
    tmp_config_dir: Path = tmp_path_factory.mktemp("agml")
    os.environ["AGML_CONFIG_DIR"] = str(tmp_config_dir)
