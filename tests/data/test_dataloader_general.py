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

import numpy as np
import pytest

import agml


@pytest.mark.order(1)
def test_loader_download_instantiation():
    loader = agml.data.AgMLDataLoader("bean_disease_uganda")
    assert os.path.exists(loader.dataset_root)


@pytest.mark.order(2)
def test_did_you_mean():
    with pytest.raises(ValueError) as exec_info:
        agml.data.AgMLDataLoader("applefr_flower_zegmentation")  # noqa
    assert "apple_flower_segmentation" in exec_info.value.args[0]
    with pytest.raises(ValueError) as exec_info:
        agml.data.AgMLDataLoader("eban_disge_ugandfea")  # noqa
    assert "bean_disease_uganda" in exec_info.value.args[0]


@pytest.mark.order(3)
def test_loader_split():
    loader = agml.data.AgMLDataLoader("bean_disease_uganda")
    loader.split(train=0.7, val=0.2, test=0.1)
    assert abs(len(loader.train_data) - int(0.7 * len(loader))) <= 1
    assert abs(len(loader.val_data) - int(0.2 * len(loader))) <= 1
    assert abs(len(loader.test_data) - int(0.1 * len(loader))) <= 1


@pytest.mark.order(4)
def test_loader_split_save_and_load():
    loader = agml.data.AgMLDataLoader("bean_disease_uganda")
    loader.split(train=0.7, val=0.2, test=0.1)
    test_json = loader._test_content
    loader.save_split("test_beans_split")

    new_loader = agml.data.AgMLDataLoader("bean_disease_uganda")
    new_loader.load_split("test_beans_split")
    assert np.all(new_loader._test_content == test_json)


@pytest.mark.order(5)
def test_loader_batch():
    loader = agml.data.AgMLDataLoader("apple_flower_segmentation")
    prev_length = len(loader)
    loader.batch(batch_size=1)
    assert len(loader) == prev_length
    loader.batch(batch_size=8)
    assert abs((len(loader) - prev_length // 8)) <= 1


@pytest.mark.order(11)
def test_loader_detection_shuffle():
    loader = agml.data.AgMLDataLoader("apple_detection_usa")
    contents = loader._manager._accessors.copy()
    loader.shuffle()
    assert np.any(contents != loader._manager._accessors)


# ── Text & Multimodal Task Tests ──────────────────────────────────────────────

_LOCAL_DATASET_DIR = os.path.join(os.path.expanduser("~"), ".agml", "datasets")


def _require_dataset(name):
    path = os.path.join(_LOCAL_DATASET_DIR, name)
    if not os.path.isdir(path):
        pytest.skip(f"Local dataset '{name}' not found at {path}")


@pytest.mark.order(20)
def test_text_classification_folder_load_and_display():
    _require_dataset("agml_crop_disease_reports")
    loader = agml.data.AgMLDataLoader("agml_crop_disease_reports", dataset_path=_LOCAL_DATASET_DIR)
    text, label = loader[0]
    print(f"\n[agml_crop_disease_reports] task={loader.task} len={len(loader)}")
    print(f"  label={label}  text[:80]={text[:80]!r}")
    assert isinstance(text, str) and isinstance(label, int)


@pytest.mark.order(21)
def test_text_classification_csv_load_and_display():
    _require_dataset("agml_wheat_stress_survey")
    loader = agml.data.AgMLDataLoader("agml_wheat_stress_survey", dataset_path=_LOCAL_DATASET_DIR)
    text, label = loader[0]
    print(f"\n[agml_wheat_stress_survey] task={loader.task} len={len(loader)}")
    print(f"  label={label}  text[:80]={text[:80]!r}")
    assert isinstance(text, str) and isinstance(label, int)


@pytest.mark.order(22)
def test_multimodal_classification_load_and_display():
    _require_dataset("agml_field_crop_multimodal")
    loader = agml.data.AgMLDataLoader("agml_field_crop_multimodal", dataset_path=_LOCAL_DATASET_DIR)
    inputs, label = loader[0]
    print(f"\n[agml_field_crop_multimodal] task={loader.task} len={len(loader)}")
    print(f"  label={label}  image.shape={inputs['image'].shape}  text[:80]={inputs['text'][:80]!r}")
    assert isinstance(inputs, dict) and "image" in inputs and "text" in inputs and isinstance(label, int)
