import os
import shutil
import pytest

import agml.data as agdata

@pytest.mark.order(1)
def test_loader_download_instantiation():
    loader = agdata.AgMLDataLoader('bean_disease_uganda')
    assert os.path.exists(loader.dataset_root)

@pytest.mark.order(2)
def test_did_you_mean():
    with pytest.raises(ValueError) as exec_info:
        agdata.AgMLDataLoader('applefr_flower_zegmentation') # noqa
    assert 'apple_flower_segmentation' in exec_info.value.args[0]
    with pytest.raises(ValueError) as exec_info:
        agdata.AgMLDataLoader('eban_disge_ugandfea') # noqa
    assert 'bean_disease_uganda' in exec_info.value.args[0]

@pytest.mark.order(3)
def test_loader_classification_split():
    loader = agdata.AgMLDataLoader('bean_disease_uganda')
    loader.split(train = 0.7, val = 0.2, test = 0.1)
    assert abs(len(loader.training_data) - int(0.7 * len(loader))) <= 1
    assert abs(len(loader.validation_data) - int(0.2 * len(loader))) <= 1
    assert abs(len(loader.test_data) - int(0.1 * len(loader))) <= 1

@pytest.mark.order(4)
def test_loader_segmentation_split():
    loader = agdata.AgMLDataLoader('apple_flower_segmentation')
    loader.split(train = 0.7, val = 0.2, test = 0.1)
    assert abs(len(loader.training_data) - int(0.7 * len(loader))) <= 1
    assert abs(len(loader.validation_data) - int(0.2 * len(loader))) <= 1
    assert abs(len(loader.test_data) - int(0.1 * len(loader))) <= 1

@pytest.mark.order(5)
def test_loader_detection_split():
    loader = agdata.AgMLDataLoader('apple_detection_usa')
    loader.split(train = 0.7, val = 0.2, test = 0.1)
    assert abs(len(loader.training_data) - int(0.7 * len(loader))) <= 1
    assert abs(len(loader.validation_data) - int(0.2 * len(loader))) <= 1
    assert abs(len(loader.test_data) - int(0.1 * len(loader))) <= 1

@pytest.mark.order(6)
def test_loader_classification_batch():
    loader = agdata.AgMLDataLoader('bean_disease_uganda')
    prev_length = len(loader)
    loader.batch(batch_size = 1)
    assert len(loader) == prev_length
    loader.batch(batch_size = 8)
    assert abs((len(loader) - prev_length // 8)) <= 1

@pytest.mark.order(7)
def test_loader_segmentation_batch():
    loader = agdata.AgMLDataLoader('apple_flower_segmentation')
    prev_length = len(loader)
    loader.batch(batch_size = 1)
    assert len(loader) == prev_length
    loader.batch(batch_size = 8)
    assert abs((len(loader) - prev_length // 8)) <= 1

@pytest.mark.order(8)
def test_loader_detection_batch():
    loader = agdata.AgMLDataLoader('apple_detection_usa')
    prev_length = len(loader)
    loader.batch(batch_size = 1)
    assert len(loader) == prev_length
    loader.batch(batch_size = 8)
    assert abs((len(loader) - prev_length // 8)) <= 1

@pytest.mark.order(9)
def test_loader_classification_shuffle():
    loader = agdata.AgMLDataLoader('bean_disease_uganda')
    contents = loader._image_paths
    loader.shuffle()
    assert contents != loader._image_paths

@pytest.mark.order(9)
def test_loader_segmentation_shuffle():
    loader = agdata.AgMLDataLoader('apple_flower_segmentation')
    contents = loader._image_paths
    loader.shuffle()
    assert contents != loader._image_paths

@pytest.mark.order(9)
def test_loader_detection_shuffle():
    loader = agdata.AgMLDataLoader('apple_detection_usa')
    contents = list(loader._coco_annotation_map.keys())
    loader.shuffle()
    assert contents != list(loader._coco_annotation_map.keys())
