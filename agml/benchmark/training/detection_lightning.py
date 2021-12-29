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
import torch
import torch.nn as nn
import pytorch_lightning as pl
from effdet.efficientdet import HeadNet
from effdet import (
    EfficientDet, get_efficientdet_config, DetBenchTrain
)

import agml
import albumentations as A


class EfficientDetTransfer(nn.Module):
    """Represents a transfer learning DeepLabV3 model.

    This is the base benchmarking model for semantic segmentation,
    using the DeepLabV3 model with a ResNet50 backbone.
    """
    def __init__(self, config):
        super(EfficientDetTransfer, self).__init__()
        self.base = EfficientDet(
            config, pretrained_backbone = config.pretrained)
        self.base.class_net = HeadNet(
            config, num_outputs = config.num_classes)
        self.config = config

    def forward(self, x, **kwargs): # noqa
        return self.base(x)


class DetectionBenchmark(pl.LightningModule):
    """Represents an image classification benchmark model."""
    def __init__(self, config):
        # Initialize the module.
        super(DetectionBenchmark, self).__init__()

        # Construct the network.
        self.net = DetBenchTrain(EfficientDetTransfer(config))

    def forward(self, x, target):
        x = self.net.forward(x, target)
        return x

    def training_step(self, batch, *args, **kwargs): # noqa
        x, y = process_data(*batch)
        loss_dict = self(x, y)
        return {
            'loss': loss_dict['loss'],
            'log': loss_dict
        }

    def validation_step(self, batch, *args, **kwargs): # noqa
        x, y = process_data(*batch)
        loss_dict = self(x, y)
        return {
            'loss': loss_dict['loss'],
            'log': loss_dict
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters())

    def get_progress_bar_dict(self):
        tqdm_dict = super(DetectionBenchmark, self)\
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


# Transform to swap bounding box order from `xyxy` to `yxyx` # noqa
def bbox_swap(coco):
    x1, y1, w, h = coco['bbox'].T
    coco['bbox'] = np.squeeze(np.dstack((y1, x1, y1 + h, x1 + w)))
    return coco


# Transform to process the image and COCO JSON dictionaries
# and make them compatible with the EfficientDet model. This
# is used directly in the training loop.
def process_data(image, coco):
    return image, {
        'bbox': [b['bbox'].float() for b in coco],
        'cls': [c['category_id'].float() for c in coco],
        'image_id': [i['image_id'] for i in coco],
        'area': [a['area'] for a in coco],
        'iscrowd': [i['iscrowd'] for i in coco],
        'img_size': torch.tensor([[256, 256]] * len(coco)).float(),
        'img_scale': torch.ones(size = (len(coco),)).float()
    }

# Build the data loaders.
def build_loaders(name):
    loader = agml.data.AgMLDataLoader(name)
    loader.split(train = 0.8, val = 0.1, test = 0.1)
    loader.batch(4)
    loader.resize_images((256, 256))
    loader.normalize_images('imagenet')
    train_data = loader.train_data
    train_data.transform(transform = A.Compose([
        A.RandomRotate90()
    ], bbox_params = A.BboxParams('coco')))
    train_data.transform(target_transform = bbox_swap)
    train_ds = train_data.copy().as_torch_dataset()
    val_ds = loader.val_data.as_torch_dataset()
    val_ds.transform(target_transform = bbox_swap)
    val_ds.shuffle_data = False
    test_ds = loader.test_data.as_torch_dataset()
    return train_ds, val_ds, test_ds


def train(dataset, pretrained, epochs, save_dir = None):
    """Constructs the training loop and trains a model."""
    if save_dir is None:
        save_dir = os.path.join(f"./data2/amnjoshi/checkpoints/{dataset}")
        os.makedirs(save_dir, exist_ok = True)

    # Construct the EfficientDet config.
    cfg = get_efficientdet_config('efficientdet_d4')
    cfg.update({
        'num_classes': agml.data.source(dataset).num_classes,
        'image_size': (256, 256),
        'pretrained': pretrained
    })

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
    model = DetectionBenchmark(cfg)

    # Construct the data loaders.
    train_ds, val_ds, test_ds = build_loaders(dataset)

    # Create the trainer and train the model.
    trainer = pl.Trainer(
        max_epochs = epochs, gpus = 0, callbacks = callbacks)
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
        '--pretrained', action = 'store_true',
        default = False, help = "Whether to load a pretrained model.")
    ap.add_argument(
        '--checkpoint_dir', type = str, default = None,
        help = "The checkpoint directory to save to.")
    ap.add_argument(
        '--epochs', type = int, default = 50,
        help = "How many epochs to train for. Default is 50.")
    args = ap.parse_args()
    args.dataset = "apple_detection_usa"

    # Train the model.
    train(args.dataset,
          args.pretrained,
          epochs = args.epochs,
          save_dir = args.checkpoint_dir)












