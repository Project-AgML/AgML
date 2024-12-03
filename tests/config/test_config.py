import os
from pathlib import Path

import pytest

import agml.backend.config as cfg


def test_config_setup():
    """Test that the config directory is set up correctly."""
    tmp_config_dir = Path(os.getenv("AGML_CONFIG_DIR"))
    # Check if the config directory exists
    assert tmp_config_dir.exists()

    # Check that the SUPER_BASE_DIR is set correctly
    assert cfg.SUPER_BASE_DIR == tmp_config_dir

    # Check that the default config file is created and valid parameters are accessible
    default_config_file = tmp_config_dir / "config.json"
    assert default_config_file.exists()

    assert cfg.data_save_path() == cfg._get_config("data_path")
    assert cfg.model_save_path() == cfg._get_config("model_path")
    assert cfg.synthetic_data_save_path() == cfg._get_config("synthetic_data_path")

    
