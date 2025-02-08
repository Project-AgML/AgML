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

import argparse
import glob
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from detection_lightning import AgMLDatasetAdaptor, EfficientDetModel
from mean_average_precision_torch import MeanAveragePrecision
from tqdm import tqdm

import agml


def run_evaluation(model, name):
    """Runs evaluation for mAP @ [0.5,0.95]."""

    pl.seed_everything(2499751)
    loader = agml.data.AgMLDataLoader(name)
    loader.shuffle()
    loader.split(0.8, 0.1, 0.1)
    return run_evaluation_by_loader(model, loader)


def run_evaluation_by_loader(model, loader):
    """Runs evaluation for a class for mAP @ [0.5,0.95]."""
    iou_thresholds = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05) + 1))

    # Create the adaptor and load the test dataset.
    ds = AgMLDatasetAdaptor(loader.test_data)

    # Create the metric.
    ma = MeanAveragePrecision(num_classes=loader.num_classes)

    # Run inference for all of the images in the test dataset.
    for i in tqdm(range(len(ds)), leave=False):
        image, bboxes, labels, _ = ds.get_image_and_labels_by_idx(i)
        pred_boxes, pred_labels, pred_conf = model.predict([image])
        pred_boxes = np.squeeze(pred_boxes)
        ma.update([pred_boxes, pred_labels, pred_conf], [bboxes, labels])

    # Compute the mAP for all of the thresholds.
    map_values = [
        ma.compute(thresh).detach().cpu().numpy() for thresh in iou_thresholds
    ]
    return np.mean(map_values)


def make_checkpoint(name, path=None, num_classes=None):
    """Gets a checkpoint for the model name."""
    model = EfficientDetModel(
        num_classes=agml.data.source(name).num_classes
        if num_classes is None
        else num_classes,
        architecture="tf_efficientdet_d4",
    )
    if path is not None:
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
    model.eval().cuda()
    return model


def evaluate_different_benchmarks(paths, names, log_file=None):
    """Runs the evaluation for different pretrained weights."""
    print(f"Running mAP evaluation for {paths}.")

    # Create the log file.
    if log_file is None:
        log_file = os.path.join(os.getcwd(), "map_evaluation.csv")

    # Run the evaluation.
    df = pd.DataFrame(
        columns=names,
        index=[os.path.basename(os.path.dirname(os.path.dirname(p))) for p in paths],
    )
    for name in names:
        log_contents = {}
        bar = tqdm(paths)
        from collections import defaultdict

        ncs = defaultdict(lambda: 1)
        for path in bar:
            nc = ncs[os.path.basename(path).split("-")[0]]
            ckpt = make_checkpoint(name, path=path, num_classes=nc)
            bar.set_description(f"Evaluating {name} @ {path} @ nc = {nc}")
            if hasattr(name, "name"):
                name = name.name
            log_contents[os.path.basename(os.path.dirname(os.path.dirname(path)))] = (
                run_evaluation(ckpt, name)
            )

        # Save the results.
        df[name] = log_contents.values()
    df.to_csv(log_file)


def evaluate_per_class(dataset, path=None, log_file=None):
    """Runs the evaluation for each class in a dataset."""
    print(f"Running mAP evaluation for {dataset}.")

    # Create the log file.
    if log_file is None:
        log_file = os.path.join(os.getcwd(), "map_evaluation_per_class.csv")

    # Construct the super-loader.
    pl.seed_everything(2499751)
    loader = agml.data.AgMLDataLoader(dataset)
    loader.shuffle()

    # Create the loop for each class.
    # Run the evaluation.
    lit = {}
    for cl in range(agml.data.source(dataset).num_classes):
        cls = cl + 1
        new_loader = loader.take_class(cls)
        new_loader.split(train=0.8, val=0.1, test=0.1)
        ckpt = make_checkpoint(
            "grape_detection_californiaday", path=path, num_classes=1
        )
        print(
            f"Evaluating fruit_detection_worldwide @ class '{loader.num_to_class[cls]}'"
        )
        result = run_evaluation_by_loader(ckpt, new_loader)

        # Save the results.
        lit[loader.num_to_class[cls]] = result
    return lit


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, nargs="+", help="The name of the dataset.")
    ap.add_argument(
        "--log_file", type=str, default=None, help="The name of the output log file."
    )
    ap.add_argument(
        "--per-class-for-dataset",
        action="store_true",
        default=False,
        help="Whether to generate benchmarks per class.",
    )
    args = ap.parse_args()

    # Train the model.
    if args.per_class_for_dataset:
        results = []
        for path_type in [
            None,
            "/data2/amnjoshi/detection-models/grape.pth",
            "/data2/amnjoshi/detection-models/amg.pth",
        ]:
            results.append(
                evaluate_per_class(
                    args.dataset[0], path=path_type, log_file=args.log_file
                )
            )
            print(results)
        loader = agml.data.AgMLDataLoader("fruit_detection_worldwide")
        df = pd.DataFrame(columns=loader.classes, index=["COCO", "GRAPE", "AMG"])
        for result, typ in zip(results, ["COCO", "GRAPE", "AMG"]):
            df.loc[typ] = result
        df.to_csv(args.log_file)
    else:
        if args.dataset[0] in agml.data.public_data_sources(ml_task="object_detection"):
            datasets = args.dataset[0]
        else:
            if args.dataset[0] == "all":
                datasets = [
                    ds
                    for ds in agml.data.public_data_sources(ml_task="object_detection")
                ]
            elif args.dataset[0] == "except":
                exclude_datasets = args.dataset[1:]
                datasets = [
                    dataset
                    for dataset in agml.data.public_data_sources(
                        ml_task="object_detection"
                    )
                    if dataset.name not in exclude_datasets
                ]
            else:
                datasets = args.dataset
        evaluate_different_benchmarks(
            glob.glob("/data2/amnjoshi/flood-grape/ejb-cc/**/*.pth", recursive=True),
            datasets,
            args.log_file,
        )
