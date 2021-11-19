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

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.models as models
from torchvision.models import efficientnet_b4

import albumentations as A

import agml


class ClassificationBenchmark(pl.LightningModule):
    """Represents an image classification benchmark model."""
    def __init__(self, dataset, pretrained = False, model = None):
        super(ClassificationBenchmark, self).__init__()

        self._source = agml.data.source(dataset)
        self._pretrained = pretrained
        self._build_model(model)

    def _build_model(self, name):
        """Constructs the actual image classification model."""
        if name is None:
            base = efficientnet_b4
        else:
            if isinstance(name, str):
                try:
                    base = getattr(models, name)
                except AttributeError:
                    raise AttributeError(
                        f"Got invalid benchmark name '{name}'.")
            elif isinstance(name, nn.Module):
                base = name
            else:
                raise TypeError(
                    "Expected either an `nn.Module` or the name "
                    "of a pretrained model in torchvision.")

        # Build and store the architecture.
        self.net = nn.Sequential(
            base(pretrained = self._pretrained),
            nn.Linear(1000, 256),
            nn.Dropout(0.1),
            nn.Linear(256, self._source.num_classes),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net.forward(x)

    def training_step(self, batch, *args, **kwargs):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = 0.02)


# Build the data loaders.
def build_loaders(name):
    loader = agml.data.AgMLDataLoader(name)
    loader.split(train = 0.8, val = 0.1, test = 0.1)
    loader.batch(batch_size = 8)
    train_data = loader.train_data
    train_data.transform(
        transform = A.Compose([
            A.RandomRotate90(),
            A.Normalize
        ])
    )
    train_ds = loader.train_data.export_torch()
    val_ds = loader.val_data.export_torch()
    test_ds = loader.test_data.export_torch()
    return train_ds, val_ds, test_ds


def train(dataset, pretrained, model = None, save_dir = None):
    """Constructs the training loop and trains a model."""
    if save_dir is None:
        save_dir = os.path.join(f"/data2/amnjoshi/agml-benchmark/{dataset}")
        os.makedirs(save_dir, exist_ok = True)

    # Set up the checkpoint saving callback.
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath = save_dir, monitor = 'val_loss', mode = 'min')]

    # Construct the model.
    model = ClassificationBenchmark(
        dataset = dataset, pretrained = pretrained, model = model)

    # Construct the data loaders.
    train_ds, val_ds, test_ds = build_loaders(dataset)

    # Create the trainer and train the model.
    trainer = pl.Trainer(
        max_epochs = 100, gpus = 1, callbacks = callbacks)
    trainer.fit(
        model = model,
        train_dataloaders = train_ds,
        val_dataloaders = val_ds
    )


if __name__ == '__main__':
    # Parse input arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--dataset', type = str, help = "The name of the dataset.")
    ap.add_argument(
        '--not-pretrained', action = 'store_false',
        default = True, help = "Whether to load a pretrained model.")
    ap.add_argument(
        '--checkpoint_dir', type = str, default = None,
        help = "The checkpoint directory to save to.")
    args = ap.parse_args()

    # Train the model.
    train(args.dataset, args.not_pretrained, save_dir = args.checkpoint_dir)









