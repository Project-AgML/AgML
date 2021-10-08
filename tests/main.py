import os
import shutil

import pytest

BASE_PATH = os.path.join(os.path.expanduser('~'), '.agml', 'datasets')
_USED_DATASETS = ['bean_disease_uganda', 'apple_flower_segmentation', 'apple_detection_usa']
_PREEXISTING_DATASETS = []

def _find_preexisting_datasets():
    global _USED_DATASETS, _PREEXISTING_DATASETS, BASE_PATH
    for dataset in _USED_DATASETS:
        if os.path.exists(os.path.join(BASE_PATH, dataset)) \
                and os.path.isdir(os.path.join(BASE_PATH, dataset)):
            _PREEXISTING_DATASETS.append(dataset)

def _remove_new_datasets():
    global _USED_DATASETS, _PREEXISTING_DATASETS, BASE_PATH
    for dataset in _USED_DATASETS:
        if dataset not in _PREEXISTING_DATASETS:
            shutil.rmtree(dataset)

def pre_test_configure():
    _find_preexisting_datasets()

def post_test_cleanup():
    _remove_new_datasets()

def execute_tests():
    pre_test_configure()
    pytest.main()
    post_test_cleanup()

if __name__ == '__main__':
    execute_tests()




