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

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models.segmentation import deeplabv3_resnet50

import agml
import albumentations as A


class DeepLabV3Transfer(nn.Module):
    """Represents a transfer learning DeepLabV3 model.

    This is the base benchmarking model for semantic segmentation,
    using the DeepLabV3 model with a ResNet50 backbone.
    """
    def __init__(self, num_classes, pretrained = True):
        super(DeepLabV3Transfer, self).__init__()
        self.base = deeplabv3_resnet50(
            pretrained = pretrained,
            num_classes = num_classes
        )

    def forward(self, x, **kwargs): # noqa
        return self.base(x)


class ClassificationBenchmark(pl.LightningModule):
    """Represents an image classification benchmark model."""
    def __init__(self, dataset, pretrained = False):
        # Initialize the module.
        super(ClassificationBenchmark, self).__init__()

        # Construct the network.
        self._source = agml.data.source(dataset)
        self._pretrained = pretrained
        self.net = DeepLabV3Transfer(
            self._source.num_classes,
            self._pretrained
        )

        # Construct the loss for training.
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.net.forward(x)

    def training_step(self, batch, *args, **kwargs): # noqa
        x, y = batch
        y_pred = self(x)['out']
        loss = self.loss(y_pred.float().squeeze(), y.float())
        return {
            'loss': loss,
        }

    def validation_step(self, batch, *args, **kwargs): # noqa
        x, y = batch
        y_pred = self(x)['out']
        val_loss = self.loss(y_pred.float().squeeze(), y.float())
        self.log('val_loss', val_loss.item(), prog_bar = True)
        return {
            'val_loss': val_loss,
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters())

    def get_progress_bar_dict(self):
        tqdm_dict = super(ClassificationBenchmark, self)\
            .get_progress_bar_dict()
        tqdm_dict.pop('v_num', None)
        return tqdm_dict


def accuracy(output, target):
    """Computes the accuracy between `output` and `target`."""
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = torch.topk(output, 1, 1)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:1].reshape(-1).float().sum(0, keepdim = True)
        return correct_k.mul_(100.0 / batch_size)


# Build the data loaders.
def build_loaders(name):
    loader = agml.data.AgMLDataLoader(name)
    loader.split(train = 0.8, val = 0.1, test = 0.1)
    loader.batch(batch_size = 16)
    loader.resize_images('imagenet')
    loader.normalize_images('imagenet')
    train_data = loader.train_data
    train_data.transform(transform = A.RandomRotate90())
    train_ds = train_data.copy().as_torch_dataset()
    val_ds = loader.val_data.as_torch_dataset()
    val_ds.shuffle_data = False
    test_ds = loader.test_data.as_torch_dataset()
    return train_ds, val_ds, test_ds


def train(dataset, pretrained, epochs, save_dir = None):
    """Constructs the training loop and trains a model."""
    if save_dir is None:
        save_dir = os.path.join(f"/data2/amnjoshi/checkpoints/{dataset}")
        os.makedirs(save_dir, exist_ok = True)

    # Set up the checkpoint saving callback.
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath = save_dir, mode = 'min',
            filename = f"{dataset}" + "-epoch{epoch:02d}-val_loss_{val_loss:.2f}",
            monitor = 'val_loss',
            save_top_k = 3,
            auto_insert_metric_name = False
        ),
        pl.callbacks.EarlyStopping(
            monitor = 'val_loss',
            min_delta = 0.001,
            patience = 3,
        )
    ]

    # Construct the model.
    model = ClassificationBenchmark(
        dataset = dataset, pretrained = pretrained)

    # Construct the data loaders.
    train_ds, val_ds, test_ds = build_loaders(dataset)

    # Create the trainer and train the model.
    trainer = pl.Trainer(
        max_epochs = epochs, gpus = 1, callbacks = callbacks)
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
    ap.add_argument(
        '--epochs', type = int, default = 50,
        help = "How many epochs to train for. Default is 50.")
    args = ap.parse_args()

    # Train the model.
    train(args.dataset,
          args.not_pretrained,
          epochs = args.epochs,
          save_dir = args.checkpoint_dir)










