import os
import time
import shutil
import pytest

import agml.data as agdata
from agml.utils.general import load_public_sources

def test_location_filters():
    uganda_filter = agdata.public_data_sources(
        location = 'country:uganda')
    assert len(uganda_filter) == 1 and uganda_filter[0].name == 'bean_disease_uganda'
    for source in agdata.public_data_sources(location = 'country:australia'):
        assert source.location.continent == 'oceania'

def test_num_images_filters():
    for source in agdata.public_data_sources(n_images = '>1000'):
        assert int(float(load_public_sources()[source.name]['n_images'])) >= 1000

def test_source_download():
    local_path = os.path.join(
        os.getcwd(), 'download_test')
    agdata.download_public_dataset('bean_disease_uganda', local_path)
    time.sleep(2)
    assert os.path.exists(os.path.join(local_path, 'bean_disease_uganda'))
    assert not os.path.exists(os.path.join(local_path, 'bean_disease_uganda.zip'))
    shutil.rmtree(os.path.join(local_path))
