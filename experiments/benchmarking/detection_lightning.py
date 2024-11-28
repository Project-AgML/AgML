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

"""
Some of the training code in this file is adapted from the following sources:

1. https://github.com/rwightman/efficientdet-pytorch
2. https://gist.github.com/Chris-hughes10/73628b1d8d6fc7d359b3dcbbbb8869d7
"""

import argparse
import os

import pytorch_lightning as pl
import torch
from detection_learning import AgMLDatasetAdaptor, EfficientDetDataModule, EfficientDetModel
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from tools import checkpoint_dir, gpus

import agml


def train(dataset, epochs, save_dir=None, overwrite=None, pretrained_path=None):
    """Constructs the training loop and trains a model."""
    save_dir = os.path.dirname(checkpoint_dir(save_dir, dataset))

    # Check if the dataset already has benchmarks.
    if os.path.exists(save_dir) and os.path.isdir(save_dir):
        if not overwrite and len(os.listdir(save_dir)) >= 4:
            print(
                f"Checkpoints already exist for {dataset} "
                f"at {save_dir}, skipping generation."
            )
            return

    # Set up the checkpoint saving callback.
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=save_dir,
            mode="min",
            filename=f"{dataset}" + "-epoch{epoch:02d}-valid_loss_{valid_loss:.2f}",
            monitor="valid_loss",
            save_top_k=3,
            auto_insert_metric_name=False,
        )
    ]

    # Create the loggers.
    loggers = [
        WandbLogger(project="detection-experiments", name=args.name, save_dir=save_dir)
    ]

    # Construct the data.
    pl.seed_everything(2499751)
    loader = agml.data.AgMLDataLoader(dataset)
    loader.shuffle()
    loader.split(train=0.8, val=0.1, test=0.1)
    dm = EfficientDetDataModule(
        train_dataset_adaptor=AgMLDatasetAdaptor(loader.train_data),
        validation_dataset_adaptor=AgMLDatasetAdaptor(loader.val_data),
        num_workers=12,
        batch_size=4,
    )

    # Construct the model.
    model = EfficientDetModel(
        num_classes=loader.num_classes,
        architecture="tf_efficientdet_d4",
        pretrained=pretrained_path,
        validation_dataset_adaptor=loader.val_data,
    )

    # Create the trainer and train the model.
    msg = f"Training dataset {dataset}!"
    print("\n" + "=" * len(msg) + "\n" + msg + "\n" + "=" * len(msg) + "\n")
    trainer = pl.Trainer(max_epochs=epochs, gpus=gpus(None), logger=loggers)
    trainer.fit(model, dm)

    # Save the final state.
    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pth"))


def train_per_class(dataset, epochs, save_dir=None, overwrite=None):
    """Constructs the training loop and trains a model."""
    save_dir = checkpoint_dir(save_dir, dataset)

    # Check if the dataset already has benchmarks.
    if os.path.exists(save_dir) and os.path.isdir(save_dir):
        if not overwrite and len(os.listdir(save_dir)) >= 4:
            print(
                f"Checkpoints already exist for {dataset} "
                f"at {save_dir}, skipping generation."
            )
            return

    # Construct the loader.
    pl.seed_everything(2499751)
    loader = agml.data.AgMLDataLoader(dataset)
    loader.shuffle()

    # Create the loop for each class.
    for cl in range(agml.data.source(dataset).num_classes):
        # Create the data module with the new, reduced class.
        cls = cl + 1
        new_loader = loader.take_class(cls)
        new_loader.split(train=0.8, val=0.1, test=0.1)
        dm = EfficientDetDataModule(
            train_dataset_adaptor=AgMLDatasetAdaptor(
                new_loader.train_data, adapt_class=True
            ),
            validation_dataset_adaptor=AgMLDatasetAdaptor(
                new_loader.val_data, adapt_class=True
            ),
            test_dataset_adaptor=AgMLDatasetAdaptor(
                new_loader.test_data, adapt_class=True
            ),
            num_workers=12,
            batch_size=4,
        )

        name = f"{loader.num_to_class[cls]}-{cls}" + f"-{args.name}"
        this_save_dir = os.path.join(save_dir, name)
        os.makedirs(this_save_dir, exist_ok=True)

        # Create the loggers.
        loggers = [
            WandbLogger(
                project="detection-experiments", save_dir=this_save_dir, name=name
            )
        ]

        # Construct the model.
        model = EfficientDetModel(
            num_classes=1,
            architecture="tf_efficientdet_d4",
            pretrained=True,
            validation_dataset_adaptor=new_loader.val_data,
            test_dataset_adaptor=new_loader.test_data,
        )
        # model.load_state_dict(
        #     torch.load('/data2/amnjoshi/detection-models/amg.pth',
        #                map_location = 'cpu'))

        # Create the trainer and train the model.
        msg = f"Training dataset {dataset} for class {cls}: {loader.num_to_class[cls]}!"
        print("\n" + "=" * len(msg) + "\n" + msg + "\n" + "=" * len(msg) + "\n")
        trainer = pl.Trainer(
            max_epochs=epochs,
            gpus=gpus(None),
            logger=loggers,
            callbacks=LearningRateMonitor("step"),
        )
        trainer.fit(model, dm)

        # Save the final state.
        torch.save(model.state_dict(), os.path.join(this_save_dir, "final_model.pth"))

        # Test the loader.
        trainer.test(datamodule=dm)


if __name__ == "__main__":
    # Parse input arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, help="The name of the run.", default=None)
    ap.add_argument("--dataset", type=str, nargs="+", help="The name of the dataset.")
    ap.add_argument(
        "--regenerate-existing",
        action="store_true",
        default=False,
        help="Whether to re-generate existing benchmarks.",
    )
    ap.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="The checkpoint directory to save to.",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="How many epochs to train for. Default is 50.",
    )
    ap.add_argument(
        "--per-class-for-dataset",
        action="store_true",
        default=False,
        help="Whether to generate benchmarks per class.",
    )
    ap.add_argument(
        "--pretrained-model-path",
        type=str,
        default=None,
        help="The path to a set of pretrained weights for the model.",
    )
    ap.add_argument(
        "--pretrained-num-classes",
        type=str,
        default=None,
        help="The number of classes in the pretrained model.",
    )
    args = ap.parse_args()
    if args.name is None:
        args.name = args.dataset

    # Train the model.
    if args.per_class_for_dataset:
        train_per_class(
            args.dataset[0], epochs=args.epochs, save_dir=args.checkpoint_dir
        )
    elif (
        args.dataset[0] in agml.data.public_data_sources(ml_task="object_detection")
        and len(args.dataset) > 1
    ):
        train(
            args.dataset,
            epochs=args.epochs,
            save_dir=args.checkpoint_dir,
            pretrained_path=(args.pretrained_model_path, args.pretrained_num_classes),
        )
    else:
        if args.dataset[0] == "all":
            datasets = [
                ds for ds in agml.data.public_data_sources(ml_task="object_detection")
            ]
        else:
            datasets = args.dataset
        for ds in datasets:
            train(
                ds,
                epochs=args.epochs,
                save_dir=args.checkpoint_dir,
                overwrite=args.regenerate_existing,
                pretrained_path=(
                    args.pretrained_model_path,
                    args.pretrained_num_classes,
                ),
            )
