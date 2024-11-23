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
from torchvision.models import efficientnet_b4

import agml


class EfficientNetB4Transfer(nn.Module):
    """Represents a transfer learning EfficientNetB4 model.

    This is the base benchmarking model for image classification, using
    the EfficientNetB4 model with two added linear fully-connected layers.
    """

    def __init__(self, num_classes, pretrained=True):
        super(EfficientNetB4Transfer, self).__init__()
        self.base = efficientnet_b4(pretrained=pretrained)
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


def get_num_gpus(args):
    """Resolves the number of GPUs to use."""
    arg_gpus = getattr(args, "gpus", None)
    if isinstance(arg_gpus, int):
        return arg_gpus
    return torch.cuda.device_count()


class ClassificationBenchmark(pl.LightningModule):
    """Represents an image classification benchmark model."""

    def __init__(self, dataset, pretrained=False):
        # Initialize the module.
        super(ClassificationBenchmark, self).__init__()

        # Construct the network.
        self._source = agml.data.source(dataset)
        self._pretrained = pretrained
        self.net = EfficientNetB4Transfer(self._source.num_classes, self._pretrained)

        # Construct the loss for training.
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net.forward(x)

    def training_step(self, batch, *args, **kwargs):  # noqa
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        acc = accuracy(y_pred, torch.argmax(y, 1)).item()
        self.log("accuracy", acc, prog_bar=True, sync_dist=True)
        return {"loss": loss, "accuracy": acc}

    def validation_step(self, batch, *args, **kwargs):  # noqa
        x, y = batch
        y_pred = self(x)
        val_loss = self.loss(y_pred, y)
        val_acc = accuracy(y_pred, torch.argmax(y, 1))
        self.log("val_loss", val_loss.item(), prog_bar=True, sync_dist=True)
        self.log("val_accuracy", val_acc.item(), prog_bar=True, sync_dist=True)
        return {"val_loss": val_loss, "val_accuracy": val_acc}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def get_progress_bar_dict(self):
        tqdm_dict = super(ClassificationBenchmark, self).get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict


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


# Main training loop method.
def train(dataset, pretrained, epochs, gpus, save_dir=None):
    """Constructs the training loop and trains a model."""
    if save_dir is None:
        save_dir = os.path.join(f"/data2/amnjoshi/checkpoints/{dataset}")
        os.makedirs(save_dir, exist_ok=True)

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
            min_delta=0.001,
            patience=3,
        ),
    ]

    # Construct the model.
    model = ClassificationBenchmark(dataset=dataset, pretrained=pretrained)

    # Construct the data loaders.
    train_ds, val_ds, test_ds = build_loaders(dataset)

    # Create the trainer and train the model.
    trainer = pl.Trainer(
        max_epochs=epochs, gpus=get_num_gpus(gpus), callbacks=callbacks
    )
    trainer.fit(model=model, train_dataloaders=train_ds, val_dataloaders=val_ds)


if __name__ == "__main__":
    # Parse input arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, help="The name of the dataset.")
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
        default=50,
        help="How many epochs to train for. Default is 50.",
    )
    ap.add_argument(
        "--gpus", type=int, default=None, help="How many GPUs to use when training."
    )
    args = ap.parse_args()

    # Train the model.
    train(
        args.dataset,
        args.not_pretrained,
        epochs=args.epochs,
        gpus=args.gpus,
        save_dir=args.checkpoint_dir,
    )
