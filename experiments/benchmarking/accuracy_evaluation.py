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
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from classification_lightning import ClassificationBenchmark
from torchmetrics import Accuracy
from tqdm import tqdm

import agml

# Define device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_evaluation(model, name):
    """Runs evaluation for categorical accuracy."""
    # Create and load the test dataset.
    pl.seed_everything(2499751)
    loader = agml.data.AgMLDataLoader(name)
    loader.shuffle()
    loader.split(0.8, 0.1, 0.1)
    loader.batch(batch_size=16)
    loader.resize_images("imagenet")
    loader.normalize_images("imagenet")
    loader.labels_to_one_hot()
    ds = loader.test_data.as_torch_dataset()

    # Create the metric.
    acc = Accuracy(num_classes=loader.num_classes)

    # Run inference for all of the images in the test dataset.
    for i in tqdm(range(len(ds)), leave=False):
        image, y = ds[i]
        y_pred = model(image.to(device))
        acc(y_pred.detach().cpu(), torch.argmax(y, 1).cpu())

    # Compute the mAP for all of the thresholds.
    return acc.compute().detach().cpu().numpy()


def make_checkpoint(name):
    """Gets a checkpoint for the model name."""
    ckpt_path = os.path.join(
        "/data2/amnjoshi/final/classification_checkpoints", name, "final_model.pth"
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model = ClassificationBenchmark(dataset=name)
    model.load_state_dict(state)
    model.eval().to(device)
    return model


def evaluate(names, log_file=None):
    """Runs the evaluation and saves results to a file."""
    print(f"Running accuracy evaluation for {names}.")

    # Create the log file.
    if log_file is None:
        log_file = os.path.join(os.getcwd(), "accuracy_evaluation.csv")

    # Run the evaluation.
    log_contents = {}
    bar = tqdm(names)
    for name in bar:
        ckpt = make_checkpoint(name)
        bar.set_description(f"Evaluating {name}")
        if hasattr(name, "name"):
            name = name.name
        log_contents[name] = run_evaluation(ckpt, name)

    # Save the results.
    df = pd.DataFrame(columns=("name", "accuracy"))
    for name, value in log_contents.items():
        df.loc[len(df.index)] = [name, value]
    df.to_csv(log_file)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, nargs="+", help="The name of the dataset.")
    ap.add_argument(
        "--log_file", type=str, default=None, help="The name of the output log file."
    )
    args = ap.parse_args()

    # Train the model.
    if args.dataset[0] in agml.data.public_data_sources(ml_task="image_classification"):
        datasets = args.dataset[0]
    else:
        if args.dataset[0] == "all":
            datasets = [
                ds
                for ds in agml.data.public_data_sources(ml_task="image_classification")
            ]
        elif args.dataset[0] == "except":
            exclude_datasets = args.dataset[1:]
            datasets = [
                dataset
                for dataset in agml.data.public_data_sources(
                    ml_task="image_classification"
                )
                if dataset.name not in exclude_datasets
            ]
        else:
            datasets = args.dataset
    evaluate(datasets, args.log_file)
