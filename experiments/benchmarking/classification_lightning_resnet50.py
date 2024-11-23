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

import albumentations as A
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from tools import MetricLogger, checkpoint_dir, gpus
from torchmetrics.classification import Accuracy, Precision, Recall
from torchvision.models import resnet50

import agml


class ResNet50Transfer(nn.Module):
    """Represents a transfer learning ResNet50 model.

    This is the base benchmarking model for image classification, using
    the ResNet50 model with two added linear fully-connected layers.
    """

    def __init__(self, num_classes, pretrained=True):
        super(ResNet50Transfer, self).__init__()
        self.base = resnet50(pretrained=pretrained)
        self.l1 = nn.Linear(1000, 256)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(256, num_classes)

    def forward(self, x, **kwargs):  # noqa
        x = self.base(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.l1(x)))
        x = self.l2(x)
        return x


class ClassificationBenchmark(pl.LightningModule):
    """Represents an image classification benchmark model."""

    def __init__(self, dataset, pretrained=False, save_dir=None):
        # Initialize the module.
        super(ClassificationBenchmark, self).__init__()

        # Construct the network.
        self._source = agml.data.source(dataset)
        self._pretrained = pretrained
        self.net = ResNet50Transfer(self._source.num_classes, self._pretrained)

        # Construct the loss for training.
        self.loss = nn.CrossEntropyLoss()

        # Add a metric calculator.
        self.metric_logger = ClassificationMetricLogger(
            {
                "accuracy": Accuracy(num_classes=self._source.num_classes),
                "precision": Precision(num_classes=self._source.num_classes),
                "recall": Recall(num_classes=self._source.num_classes),
            },
            os.path.join(save_dir, f"logs-{self._version}.csv"),
        )
        self._sanity_check_passed = False

    def forward(self, x):
        return self.net.forward(x)

    def training_step(self, batch, *args, **kwargs):  # noqa
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        acc = accuracy(y_pred, torch.argmax(y, 1)).item()
        self.log("accuracy", acc, prog_bar=True, logger=True)
        self.log("loss", loss, logger=True)
        return {"loss": loss, "accuracy": acc}

    def validation_step(self, batch, *args, **kwargs):  # noqa
        x, y = batch
        y_pred = self(x)
        val_loss = self.loss(y_pred, y)
        val_acc = accuracy(y_pred, torch.argmax(y, 1))
        if self._sanity_check_passed:
            self.metric_logger.update(y_pred, torch.argmax(y, 1))
        self.log("val_loss", val_loss.item(), prog_bar=True, logger=True)
        self.log("val_accuracy", val_acc.item(), prog_bar=True, logger=True)
        return {"val_loss": val_loss, "val_accuracy": val_acc}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def get_progress_bar_dict(self):
        tqdm_dict = super(ClassificationBenchmark, self).get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict

    def on_validation_epoch_end(self) -> None:
        if not self._sanity_check_passed:
            self._sanity_check_passed = True
            return
        self.metric_logger.compile_epoch()

    def on_fit_end(self) -> None:
        self.metric_logger.save()


# Calculate and log the metrics.
class ClassificationMetricLogger(MetricLogger):
    def update_metrics(self, y_pred, y_true) -> None:
        for metric in self.metrics.values():
            metric.update(y_pred.cpu(), y_true.cpu())


def accuracy(output, target):
    """Computes the accuracy between `output` and `target`."""
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = torch.topk(output, 1, 1)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:1].reshape(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(100.0 / batch_size)


# Build the data loaders.
def build_loaders(name):
    pl.seed_everything(2499751)
    loader = agml.data.AgMLDataLoader(name)
    loader.split(train=0.8, val=0.1, test=0.1)
    loader.batch(batch_size=16)
    loader.resize_images("imagenet")
    loader.normalize_images("imagenet")
    loader.labels_to_one_hot()
    train_data = loader.train_data
    train_data.transform(transform=A.RandomRotate90())
    train_ds = train_data.copy().as_torch_dataset()
    val_ds = loader.val_data.as_torch_dataset()
    val_ds.shuffle_data = False
    test_ds = loader.test_data.as_torch_dataset()
    return train_ds, val_ds, test_ds


def train(dataset, pretrained, epochs, save_dir=None, overwrite=None):
    """Constructs the training loop and trains a model."""
    save_dir = "/data2/amnjoshi/resnet50_pretrained/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    log_dir = save_dir.replace("checkpoints", "logs")
    os.makedirs(log_dir, exist_ok=True)

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
            filename=f"{dataset}" + "-epoch{epoch:02d}-val_loss_{val_loss:.2f}",
            monitor="val_loss",
            save_top_k=3,
            auto_insert_metric_name=False,
        ),
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.0001,
            patience=10,
        ),
    ]

    # Construct the model.
    model = ClassificationBenchmark(
        dataset=dataset, pretrained=pretrained, save_dir=save_dir
    )

    # Construct the data loaders.
    train_ds, val_ds, test_ds = build_loaders(dataset)

    # Create the loggers.
    loggers = [CSVLogger(log_dir), TensorBoardLogger(log_dir)]

    # Create the trainer and train the model.
    msg = f"Training dataset {dataset}!"
    print("\n" + "=" * len(msg) + "\n" + msg + "\n" + "=" * len(msg) + "\n")
    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=gpus(None),
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=5,
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_ds,
        val_dataloaders=val_ds,
    )


if __name__ == "__main__":
    # Parse input arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, nargs="+", help="The name of the dataset.")
    ap.add_argument(
        "--regenerate-existing",
        action="store_true",
        default=False,
        help="Whether to re-generate existing benchmarks.",
    )
    ap.add_argument(
        "--not-pretrained",
        action="store_false",
        default=True,
        help="Whether to load a pretrained model.",
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
        default=20,
        help="How many epochs to train for. Default is 20.",
    )
    args = ap.parse_args()

    # Train the model.
    if args.dataset[0] in agml.data.public_data_sources(ml_task="image_classification"):
        train(
            args.dataset,
            args.not_pretrained,
            epochs=args.epochs,
            save_dir=args.checkpoint_dir,
        )
    else:
        if args.dataset[0] == "all":
            datasets = [
                ds
                for ds in agml.data.public_data_sources(ml_task="image_classification")
            ]
        else:
            datasets = args.dataset
        for dataset in datasets:
            train(
                dataset,
                args.not_pretrained,
                epochs=args.epochs,
                save_dir=args.checkpoint_dir,
                overwrite=args.regenerate_existing,
            )
