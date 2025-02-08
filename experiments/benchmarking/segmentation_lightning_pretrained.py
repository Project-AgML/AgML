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
from collections import OrderedDict

import albumentations as A
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.loggers import WandbLogger
from tools import MetricLogger, checkpoint_dir, gpus
from torchmetrics import IoU
from torchvision.models.segmentation import deeplabv3_resnet50

import agml


class DeepLabV3Transfer(nn.Module):
    """Represents a transfer learning DeepLabV3 model.

    This is the base benchmarking model for semantic segmentation,
    using the DeepLabV3 model with a ResNet50 backbone.
    """

    def __init__(self, num_classes, pretrained=False, unfreeze_backbone=False):
        super(DeepLabV3Transfer, self).__init__()
        self.base = deeplabv3_resnet50(pretrained=False, num_classes=num_classes)
        if pretrained:
            self.base.load_state_dict(load_checkpoint(), strict=False)

        if not unfreeze_backbone:
            for parameter in self.base.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, x, **kwargs):  # noqa
        return self.base(x)


def load_checkpoint():
    """Loads a ResNet50 pretrained checkpoint."""
    ckpt_path = (
        "/data2/amnjoshi/resnet50_pretrained/checkpoints/"
        "plant_village_classification-epoch42-val_loss_0.05.ckpt"
    )
    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    new_state = []
    for key in state:
        if key.startswith("net.base"):
            new_state.append((key.replace("net.base.", "backbone."), state[key]))
    new_state = OrderedDict(new_state)
    return new_state


def dice_loss(y_pred, y):
    y = y.float()
    try:  # Multi-class segmentation
        c, h, w = y.shape[1:]
    except:  # Binary segmentation
        h, w = y.shape[1:]
        c = 1  # noqa
    pred_flat = torch.reshape(y_pred, [-1, c * h * w])
    y_flat = torch.reshape(y, [-1, c * h * w])
    intersection = 2.0 * torch.sum(pred_flat * y_flat, dim=1) + 1e-6
    denominator = torch.sum(pred_flat, dim=1) + torch.sum(y_flat, dim=1) + 1e-6
    return 1.0 - torch.mean(intersection / denominator)


def dice_metric(y_pred, y):
    intersection = 2.0 * (y_pred * y).sum()
    union = y_pred.sum() + y.sum()
    if union == 0.0:
        return 1.0
    return intersection / union


class SegmentationBenchmark(pl.LightningModule):
    """Represents an image classification benchmark model."""

    def __init__(
        self, dataset, pretrained=False, unfreeze_backbone=False, save_dir=None
    ):
        # Initialize the module.
        super(SegmentationBenchmark, self).__init__()

        # Construct the network.
        self._source = agml.data.source(dataset)
        self._pretrained = pretrained
        self.net = DeepLabV3Transfer(
            self._source.num_classes,
            self._pretrained,
            unfreeze_backbone=unfreeze_backbone,
        )

        # Construct the loss for training.
        if self._source.num_classes == 1:
            self.loss = nn.BCEWithLogitsLoss()
        else:
            self.loss = dice_loss

        # Construct the IoU metric.
        self.iou = IoU(self._source.num_classes + 1)

        # Add a metric calculator.
        self.metric_logger = SegmentationMetricLogger(
            {"iou": IoU(self._source.num_classes + 1)},
            os.path.join(save_dir, f"logs-{self._version}.csv"),
        )
        self._sanity_check_passed = False

    def forward(self, x):
        return self.net.forward(x)

    def calculate_loss(self, y_pred, y):
        if self._source.num_classes != 1:
            return self.loss(y_pred, y.long())
        return self.loss(y_pred, y.float())

    def training_step(self, batch, *args, **kwargs):  # noqa
        x, y = batch
        y_pred = self(x)["out"].float().squeeze()
        loss = self.calculate_loss(y_pred, y)
        iou = self.iou(y_pred, y.int())
        self.log("iou", iou.item(), prog_bar=True)
        return {
            "loss": loss,
        }

    def validation_step(self, batch, *args, **kwargs):  # noqa
        x, y = batch
        y_pred = self(x)["out"].float().squeeze()
        val_loss = self.calculate_loss(y_pred, y)
        self.log("val_loss", val_loss.item(), prog_bar=True)
        val_iou = self.iou(y_pred, y.int())
        if self._sanity_check_passed:
            self.metric_logger.update_metrics(y_pred, y.int())
        self.log("val_iou", val_iou.item(), prog_bar=True)
        return {
            "val_loss": val_loss,
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters())

    def get_progress_bar_dict(self):
        tqdm_dict = super(SegmentationBenchmark, self).get_progress_bar_dict()
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
class SegmentationMetricLogger(MetricLogger):
    def update_metrics(self, y_pred, y_true) -> None:
        for metric in self.metrics.values():
            metric.update(y_pred.cpu(), y_true.cpu())


# Build the data loaders.
def build_loaders(name):
    pl.seed_everything(2499751)
    loader = agml.data.AgMLDataLoader(name)
    loader.split(train=0.8, val=0.1, test=0.1)
    loader.batch(batch_size=8)
    loader.resize_images("imagenet")
    # loader.normalize_images('imagenet')
    loader.mask_to_channel_basis()
    train_data = loader.train_data
    train_data.transform(transform=A.RandomRotate90())
    train_ds = train_data.copy().as_torch_dataset()
    val_ds = loader.val_data.as_torch_dataset()
    val_ds.shuffle_data = False
    test_ds = loader.test_data.as_torch_dataset()
    return train_ds, val_ds, test_ds


def train(
    dataset,
    epochs,
    save_dir=None,
    pretrained=False,
    unfreeze_backbone=False,
    overwrite=None,
):
    """Constructs the training loop and trains a model."""
    save_dir = checkpoint_dir(save_dir, dataset)
    log_dir = save_dir.replace("checkpoints", "logs")

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
            monitor="val_iou",
            save_top_k=3,
            auto_insert_metric_name=False,
        ),
    ]

    # Construct the model.
    model = SegmentationBenchmark(
        dataset=dataset,
        save_dir=save_dir,
        pretrained=pretrained,
        unfreeze_backbone=unfreeze_backbone,
    )

    # Construct the data loaders.
    train_ds, val_ds, test_ds = build_loaders(dataset)

    # Create the loggers.
    loggers = [
        WandbLogger(
            save_dir=os.path.dirname(log_dir),
            name=f"{dataset}-pretrained",
            project="segmentation-experiments",
        )
    ]

    # Create the trainer and train the model.
    msg = f"Training dataset {dataset}!"
    print("\n" + "=" * len(msg) + "\n" + msg + "\n" + "=" * len(msg) + "\n")
    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=gpus(),
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=5,
    )
    trainer.fit(model=model, train_dataloaders=train_ds, val_dataloaders=val_ds)


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
        default=20,
        help="How many epochs to train for. Default is 20.",
    )
    ap.add_argument(
        "--unfreeze-backbone",
        action="store_true",
        default=False,
        help="Whether to not freeze backbone weights.",
    )
    args = ap.parse_args()

    # Train the model.
    if args.dataset[0] in agml.data.public_data_sources(
        ml_task="semantic_segmentation"
    ):
        train(
            dataset=args.dataset[0],
            epochs=args.epochs,
            save_dir=args.checkpoint_dir,
            unfreeze_backbone=args.unfreeze_backbone,
        )
    else:
        if args.dataset[0] == "all":
            datasets = [
                ds
                for ds in agml.data.public_data_sources(ml_task="semantic_segmentation")
            ]
        else:
            datasets = args.dataset
        for ds in datasets:
            train(
                dataset=ds,
                epochs=args.epochs,
                pretrained=args.pretrained,
                save_dir=args.checkpoint_dir,
                unfreeze_backbone=args.unfreeze_backbone,
                overwrite=args.regenerate_existing,
            )
