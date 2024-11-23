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
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision.models.detection import ssd300_vgg16

import agml


class EfficientDetTransfer(nn.Module):
    """Represents a transfer learning DeepLabV3 model.

    This is the base benchmarking model for semantic segmentation,
    using the DeepLabV3 model with a ResNet50 backbone.
    """

    def __init__(self):
        super(EfficientDetTransfer, self).__init__()
        self.base = ssd300_vgg16(pretrained=False)

    def forward(self, x, y, **kwargs):  # noqa
        return self.base(x, y)


class DetectionBenchmark(pl.LightningModule):
    """Represents an image classification benchmark model."""

    def __init__(self):
        # Initialize the module.
        super(DetectionBenchmark, self).__init__()

        # Construct the network.
        self.net = ssd300_vgg16(pretrained=False)

    def forward(self, x, target):
        return self.net(x, target)

    def training_step(self, batch, *args, **kwargs):  # noqa
        x, y = process_data(*batch)
        loss_dict = self(x, y)
        self.log("class_loss", loss_dict["classification"], prog_bar=True)
        self.log("reg_loss", loss_dict["bbox_regression"], prog_bar=True)
        return {"loss": sum(i for i in loss_dict.values()), "log": loss_dict}

    def validation_step(self, batch, *args, **kwargs):  # noqa
        x, y = process_data(*batch)
        self.net.train()
        loss_dict = self(x, y)
        self.log("class_loss", loss_dict["classification"], prog_bar=True)
        self.log("reg_loss", loss_dict["bbox_regression"], prog_bar=True)
        return {"loss": sum(i for i in loss_dict.values()), "log": loss_dict}

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters())

    def get_progress_bar_dict(self):
        tqdm_dict = super(DetectionBenchmark, self).get_progress_bar_dict()
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


# Transform to swap bounding box order from `xyxy` to `yxyx` # noqa
def bbox_swap(coco):
    x1, y1, w, h = coco["bbox"].T
    coco["bbox"] = np.squeeze(np.dstack((x1, y1, x1 + w, y1 + h)))
    return coco


# Transform to process the image and COCO JSON dictionaries
# and make them compatible with the EfficientDet model. This
# is used directly in the training loop.
def _make_boxes(b):
    if b.ndim == 1:
        return torch.unsqueeze(b, dim=0)
    return b


def process_data(image, coco):
    return torch.unbind(image, dim=0), [
        {"boxes": _make_boxes(d["bbox"].float()), "labels": d["category_id"].long()}
        for d in coco
    ]


# Build the data loaders.
def build_loaders(name):
    loader = agml.data.AgMLDataLoader(name)
    loader.split(train=0.8, val=0.1, test=0.1)
    loader.batch(4)
    loader.resize_images((256, 256))
    loader.normalize_images(method="scale")
    train_data = loader.train_data
    train_data.transform(
        transform=A.Compose([A.RandomRotate90()], bbox_params=A.BboxParams("coco"))
    )
    train_data.transform(target_transform=bbox_swap)
    train_ds = train_data.copy().as_torch_dataset()
    val_ds = loader.val_data.as_torch_dataset()
    val_ds.transform(target_transform=bbox_swap)
    val_ds.shuffle_data = False
    test_ds = loader.test_data.as_torch_dataset()
    return train_ds, val_ds, test_ds


def train(dataset, pretrained, epochs, save_dir=None):
    """Constructs the training loop and trains a model."""
    if save_dir is None:
        if os.path.isdir("/data2"):
            save_dir = os.path.join(f"/data2/amnjoshi/checkpoints/{dataset}")
        else:
            save_dir = os.path.join(os.path.dirname(__file__), "logs")
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
    model = DetectionBenchmark()

    # Construct the data loaders.
    train_ds, val_ds, test_ds = build_loaders(dataset)

    # Create the trainer and train the model.
    trainer = pl.Trainer(max_epochs=epochs, gpus=0, callbacks=callbacks)
    trainer.fit(model=model, train_dataloaders=train_ds, val_dataloaders=val_ds)


if __name__ == "__main__":
    from agml.backend import set_seed

    set_seed(0)

    # Parse input arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, help="The name of the dataset.")
    ap.add_argument(
        "--pretrained",
        action="store_true",
        default=False,
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
    args = ap.parse_args()
    args.dataset = "apple_detection_usa"

    # Train the model.
    train(
        args.dataset, args.pretrained, epochs=args.epochs, save_dir=args.checkpoint_dir
    )
