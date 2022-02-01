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
from torchmetrics import IoU
from torchvision.models.segmentation import deeplabv3_resnet50

import agml
import albumentations as A

from tools import gpus, checkpoint_dir


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


def dice_loss(y_pred, y):
    y = y.float()
    try: # Multi-class segmentation
        c, h, w = y.shape[1:]
    except: # Binary segmentation
        h, w = y.shape[1:]; c = 1 # noqa
    pred_flat = torch.reshape(y_pred, [-1, c * h * w])
    y_flat = torch.reshape(y, [-1, c * h * w])
    intersection = 2.0 * torch.sum(pred_flat * y_flat, dim = 1) + 1e-6
    denominator = torch.sum(pred_flat, dim = 1) + torch.sum(y_flat, dim = 1) + 1e-6
    return 1. - torch.mean(intersection / denominator)


def dice_metric(y_pred, y):
    intersection = 2.0 * (y_pred * y).sum()
    union = y_pred.sum() + y.sum()
    if union == 0.0:
        return 1.0
    return intersection / union


class SegmentationBenchmark(pl.LightningModule):
    """Represents an image classification benchmark model."""
    def __init__(self, dataset, pretrained = False):
        # Initialize the module.
        super(SegmentationBenchmark, self).__init__()

        # Construct the network.
        self._source = agml.data.source(dataset)
        self._pretrained = pretrained
        self.net = DeepLabV3Transfer(
            self._source.num_classes,
            self._pretrained
        )

        # Construct the loss for training.
        self.loss = dice_loss

        # Construct the IoU metric.
        self.iou = IoU(self._source.num_classes + 1)

    def forward(self, x):
        return self.net.forward(x)

    def calculate_loss(self, y_pred, y):
        if self._source.num_classes != 1:
            return self.loss(y_pred, y.long())
        return self.loss(y_pred, y.float())

    def training_step(self, batch, *args, **kwargs): # noqa
        x, y = batch
        y_pred = self(x)['out'].float().squeeze()
        loss = self.calculate_loss(y_pred, y)
        iou = self.iou(y_pred, y.int())
        self.log('iou', iou.item(), prog_bar = True)
        self.log('dice', dice_metric(y_pred, y).item(), prog_bar = True)
        return {
            'loss': loss,
        }

    def validation_step(self, batch, *args, **kwargs): # noqa
        x, y = batch
        y_pred = self(x)['out'].float().squeeze()
        val_loss = self.calculate_loss(y_pred, y)
        self.log('val_loss', val_loss.item(), prog_bar = True)
        val_iou = self.iou(y_pred, y.int())
        self.log('val_iou', val_iou.item(), prog_bar = True)
        self.log('val_dice', dice_metric(y_pred, y).item(), prog_bar = True)
        return {
            'val_loss': val_loss,
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters())

    def get_progress_bar_dict(self):
        tqdm_dict = super(SegmentationBenchmark, self).get_progress_bar_dict()
        tqdm_dict.pop('v_num', None)
        return tqdm_dict


# Build the data loaders.
def build_loaders(name):
    loader = agml.data.AgMLDataLoader(name)
    loader.split(train = 0.8, val = 0.1, test = 0.1)
    loader.batch(batch_size = 16)
    loader.resize_images('imagenet')
    loader.normalize_images('imagenet')
    loader.mask_to_channel_basis()
    train_data = loader.train_data
    train_data.transform(transform = A.RandomRotate90())
    train_ds = train_data.copy().as_torch_dataset()
    val_ds = loader.val_data.as_torch_dataset()
    val_ds.shuffle_data = False
    test_ds = loader.test_data.as_torch_dataset()
    return train_ds, val_ds, test_ds


def train(dataset, pretrained, epochs, save_dir = None):
    """Constructs the training loop and trains a model."""
    save_dir = checkpoint_dir(save_dir, dataset)

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
    model = SegmentationBenchmark(
        dataset = dataset, pretrained = pretrained)

    # Construct the data loaders.
    train_ds, val_ds, test_ds = build_loaders(dataset)

    # Create the trainer and train the model.
    trainer = pl.Trainer(
        max_epochs = epochs, gpus = gpus(), callbacks = callbacks)
    trainer.fit(
        model = model,
        train_dataloaders = train_ds,
        val_dataloaders = val_ds)

    # Run the testing loop and log benchmarks.



if __name__ == '__main__':
    # Parse input arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--dataset', type = str, help = "The name of the dataset.")
    ap.add_argument(
        '--pretrained', action = 'store_true',
        default = False, help = "Whether to load a pretrained model.")
    ap.add_argument(
        '--checkpoint_dir', type = str, default = None,
        help = "The checkpoint directory to save to.")
    ap.add_argument(
        '--epochs', type = int, default = 50,
        help = "How many epochs to train for. Default is 50.")
    args = ap.parse_args()

    # Train the model.
    train(args.dataset,
          args.pretrained,
          epochs = args.epochs,
          save_dir = args.checkpoint_dir)










