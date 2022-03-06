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
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import pytorch_lightning as pl

import agml
from detection_lightning import AgMLDatasetAdaptor, EfficientDetModel
from mean_average_precision_torch import MeanAveragePrecision


def run_evaluation(model, name):
    """Runs evaluation for mAP @ [0.5,0.95]."""
    iou_thresholds = np.linspace(
        0.5, 0.95, int(np.round((0.95 - .5) / .05) + 1))

    # Create the adaptor and load the test dataset.
    pl.seed_everything(2499751)
    loader = agml.data.AgMLDataLoader(name)
    loader.shuffle()
    loader.split(0.8, 0.1, 0.1)
    ds = AgMLDatasetAdaptor(loader.test_data)

    # Create the metric.
    ma = MeanAveragePrecision(num_classes = loader.num_classes)

    # Run inference for all of the images in the test dataset.
    for i in tqdm(range(len(ds)), leave = False):
        image, bboxes, labels, _ = ds.get_image_and_labels_by_idx(i)
        pred_boxes, pred_labels, pred_conf = model.predict([image])
        pred_boxes = np.squeeze(pred_boxes)
        ma.update([pred_boxes, pred_labels, pred_conf], [bboxes, labels])

    # Compute the mAP for all of the thresholds.
    map_values = [ma.compute(thresh).detach().cpu().numpy()
                  for thresh in iou_thresholds]
    return map_values, np.mean(map_values)


def make_checkpoint(name):
    """Gets a checkpoint for the model name."""
    ckpt_path = os.path.join(
        "/data2/amnjoshi/final/detection_checkpoints", name, "final_model.pth")
    state = torch.load(ckpt_path, map_location = 'cpu')
    model = EfficientDetModel(
        num_classes = agml.data.source(name).num_classes,
        architecture = 'tf_efficientdet_d4')
    model.load_state_dict(state)
    model.eval().cuda()
    return model


def evaluate(names, log_file = None):
    """Runs the evaluation and saves results to a file."""
    print(f"Running mAP evaluation for {names}.")

    # Create the log file.
    if log_file is None:
        log_file = os.path.join(os.getcwd(), 'map_evaluation.csv')

    # Run the evaluation.
    log_contents = {}
    bar = tqdm(names)
    for name in bar:
        ckpt = make_checkpoint(name)
        bar.set_description(f"Evaluating {name}")
        if hasattr(name, 'name'):
            name = name.name
        log_contents[name] = run_evaluation(ckpt, name)

    # Save the results.
    df = pd.DataFrame(columns = ('name', *[f'map@{float(th)}' for th in np.linspace(
        0.5, 0.95, int(np.round((0.95 - .5) / .05) + 1))], 'map@[0.5,0.95]'))
    for name, values in log_contents.items():
        df.loc[len(df.index)] = [name, *values[0], values[1]]
    df.to_csv(log_file)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--dataset', type = str, nargs = '+', help = "The name of the dataset.")
    ap.add_argument(
        '--log_file', type = str, default = None,
        help = "The name of the output log file.")
    args = ap.parse_args()

    # Train the model.
    if args.dataset[0] in agml.data.public_data_sources(ml_task = 'object_detection'):
        datasets = args.dataset[0]
    else:
        if args.dataset[0] == 'all':
            datasets = [ds for ds in agml.data.public_data_sources(
                ml_task = 'object_detection')]
        elif args.dataset[0] == 'except':
            exclude_datasets = args.dataset[1:]
            datasets = [
                dataset for dataset in agml.data.public_data_sources(
                    ml_task = 'object_detection')
                if dataset.name not in exclude_datasets]
        else:
            datasets = args.dataset
    evaluate(datasets, args.log_file)




