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

import os
import pytest
import numpy as np

import agml


@pytest.mark.order(1)
def test_loader_download_instantiation():
    loader = agml.data.AgMLDataLoader('bean_disease_uganda')
    assert os.path.exists(loader.dataset_root)


@pytest.mark.order(2)
def test_did_you_mean():
    with pytest.raises(ValueError) as exec_info:
        agml.data.AgMLDataLoader('applefr_flower_zegmentation') # noqa
    assert 'apple_flower_segmentation' in exec_info.value.args[0]
    with pytest.raises(ValueError) as exec_info:
        agml.data.AgMLDataLoader('eban_disge_ugandfea') # noqa
    assert 'bean_disease_uganda' in exec_info.value.args[0]


@pytest.mark.order(3)
def test_loader_split():
    loader = agml.data.AgMLDataLoader('bean_disease_uganda')
    loader.split(train = 0.7, val = 0.2, test = 0.1)
    assert abs(len(loader.train_data) - int(0.7 * len(loader))) <= 1
    assert abs(len(loader.val_data) - int(0.2 * len(loader))) <= 1
    assert abs(len(loader.test_data) - int(0.1 * len(loader))) <= 1


@pytest.mark.order(4)
def test_loader_split_save_and_load():
    loader = agml.data.AgMLDataLoader('bean_disease_uganda')
    loader.split(train = 0.7, val = 0.2, test = 0.1)
    test_json = loader._test_content
    loader.save_split('test_beans_split')

    new_loader = agml.data.AgMLDataLoader('bean_disease_uganda')
    new_loader.load_split('test_beans_split')
    assert np.all(new_loader._test_content == test_json)


@pytest.mark.order(5)
def test_loader_batch():
    loader = agml.data.AgMLDataLoader('apple_flower_segmentation')
    prev_length = len(loader)
    loader.batch(batch_size = 1)
    assert len(loader) == prev_length
    loader.batch(batch_size = 8)
    assert abs((len(loader) - prev_length // 8)) <= 1


@pytest.mark.order(11)
def test_loader_detection_shuffle():
    loader = agml.data.AgMLDataLoader('apple_detection_usa')
    contents = loader._manager._accessors.copy()
    loader.shuffle()
    assert np.any(contents != loader._manager._accessors)



