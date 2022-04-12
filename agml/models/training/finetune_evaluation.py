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
import pickle

import numpy as np

import torch
import pytorch_lightning as pl

from detection_lightning import (
    EfficientDetModel, EfficientDetDataModule, AgMLDatasetAdaptor
)
from mean_average_precision_torch import MeanAveragePrecision
from pytorch_lightning.loggers import TensorBoardLogger

import agml

from tools import gpus, checkpoint_dir
from tqdm import tqdm


FINETUNE_EPOCHS = 5
EVAL_CLASSES = ['orange', 'avocado', 'capsicum', 'mango']
EVAL_QUANTITIES = [6, 12, 18, 24, 30, 36]
PRETRAINED_PATH = '/data2/amnjoshi/amg/checkpoints/model_state.pth'
BASE = '/data2/amnjoshi/finetune'


def generate_splits():
    # Generate the base loader.
    pl.seed_everything(81923801)
    loader = agml.data.AgMLDataLoader('fruit_detection_worldwide')

    # Create the new loaders for each of the classes.
    cls_quant_loaders = {}
    for cls in EVAL_CLASSES:
        # We set aside a random pool of 36 from which the finetuning images
        # will be selected, and then another pool of 15 which will be used
        # purely for testing the mean average precision.
        cls_loader = loader.take_class(cls).take_random(36 + 15)
        cls_loader.split(train = 36, test = 15)
        pool_loader = cls_loader.train_data
        test_loader = cls_loader.test_data
        quant_loaders = {'test': test_loader}
        for quant in EVAL_QUANTITIES:
            quant_loaders[quant] = pool_loader.take_random(quant)
        cls_quant_loaders[cls] = quant_loaders
    return cls_quant_loaders


def train(cls, loader, save_dir, overwrite = False):
    """Constructs the training loop and trains a model."""
    dataset = loader.name
    save_dir = checkpoint_dir(save_dir, dataset)
    log_dir = os.path.join(save_dir, 'logs')

    # Check if the dataset already has benchmarks.
    if os.path.exists(save_dir) and os.path.isdir(save_dir):
        if not overwrite and len(os.listdir(save_dir)) >= 4:
            print(f"Checkpoints already exist for {dataset} "
                  f"at {save_dir}, skipping generation.")
            return

    # Create the loggers.
    loggers = [
        TensorBoardLogger(log_dir)
    ]

    # Construct the data.
    dm = EfficientDetDataModule(
        train_dataset_adaptor = AgMLDatasetAdaptor(loader.train_data),
        validation_dataset_adaptor = AgMLDatasetAdaptor(loader.val_data),
        num_workers = 12, batch_size = 1)

    # Construct the model.
    model = EfficientDetModel(
        num_classes = loader.num_classes,
        architecture = 'tf_efficientdet_d4',
        validation_dataset_adaptor = loader.val_data)
    model.load_state_dict(torch.load(PRETRAINED_PATH, map_location = 'cpu'))

    # Create the trainer and train the model.
    msg = f"Finetuning class {cls} of size {len(loader.train_data)}!"
    print("\n" + "=" * len(msg) + "\n" + msg + "\n" + "=" * len(msg) + "\n")
    trainer = pl.Trainer(
        max_epochs = FINETUNE_EPOCHS, gpus = None, logger = loggers)
    trainer.fit(model, dm)

    # Save the final state.
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))

    # Return the model state.
    return model


def run_evaluation(model, loader) -> dict:
    """Runs evaluation for mAP @ [0.5,0.95]."""
    iou_thresholds = np.linspace(
        0.5, 0.95, int(np.round((0.95 - .5) / .05) + 1))

    # Create the adaptor and load the test dataset.
    ds = AgMLDatasetAdaptor(loader)

    # Create the metric.
    ma = MeanAveragePrecision(num_classes = loader.num_classes)

    # Run inference for all of the images in the test dataset.
    for i in tqdm(range(len(ds)), leave = False, desc = "Running mAP Evaluation"):
        image, bboxes, labels, _ = ds.get_image_and_labels_by_idx(i)
        pred_boxes, pred_labels, pred_conf = model.predict([image])
        pred_boxes = np.squeeze(pred_boxes)
        ma.update([pred_boxes, pred_labels, pred_conf], [bboxes, labels])

    # Compute the mAP for all of the thresholds.
    map_values = {f'map@{round(float(thresh), 2)}':
                      ma.compute(thresh).detach().cpu().numpy()
                  for thresh in iou_thresholds}
    map_values['map@[0.5,0.95]'] = np.mean(list(map_values.values()))
    return map_values


def train_all():
    # Get all of the data splits.
    splits = generate_splits()

    # Create a dictionary with all of the results.
    results = {}

    # Iterate over each finetuning class and image quantity.
    for cls in EVAL_CLASSES:
        if cls not in results.keys():
            results[cls] = {}
        cls_path = os.path.join(BASE, cls)
        test_loader = splits[cls]['test']
        for quant in EVAL_QUANTITIES:
            quant_path = os.path.join(cls_path, f'n-{quant}')
            os.makedirs(quant_path, exist_ok = True)

            # Get the loader and split it accordingly.
            loader = splits[cls][quant]
            train_q, val_q = 5 * (quant // 6), quant // 6
            loader.split(train = train_q, val = val_q)

            # Train the model.
            try:
                model = train(cls, loader = loader, save_dir = quant_path)
            except KeyboardInterrupt:
                raise ValueError

            # Evaluate the model.
            model.eval()
            eval_dict = run_evaluation(model, test_loader)
            results[cls][quant] = eval_dict
            print("\n", eval_dict, "\n")

    # Save all of the results.
    from pprint import pprint
    print("\n\nRESULTS:\n")
    pprint(results)
    with open(os.path.join(BASE, 'results.pickle'), 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    train_all()

