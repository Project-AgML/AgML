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
import sys
from enum import Enum
import argparse

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4

import numpy as np
from tqdm import tqdm
import albumentations as A

import agml
from agml.utils.io import recursive_dirname


# Build the model with the correct image classification head.
# Build a wrapper class for the `EfficientNetB4` model.
class EfficientNetB4Transfer(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB4Transfer, self).__init__()
        self.base = efficientnet_b4(pretrained = True)
        self.l1 = nn.Linear(1000, 256)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(256, num_classes)

    def forward(self, x, **kwargs):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.l1(x)))
        x = self.l2(x)
        return x


def to_0_1_range(x):
    return (x / 255).astype(np.float32)


# Build the data loaders.
def build_loaders(name):
    loader = agml.data.AgMLDataLoader(name)
    loader.split(train = 0.8, val = 0.1, test = 0.1)
    loader.batch(batch_size = 2)
    loader.transform(
        transform = to_0_1_range
    )
    loader.resize_images('imagenet')
    loader.labels_to_one_hot()
    train_data = loader.train_data
    train_data.transform(
        transform = A.Compose([
            A.Normalize(max_pixel_value = 1.0),
            A.RandomRotate90(),
        ])
    )
    train_ds = train_data.copy()
    val_ds = loader.val_data
    test_ds = loader.test_data
    return train_ds, val_ds, test_ds


# Create the training loop.
class Trainer(object):
    """Trains a model and saves checkpoints to a save directory."""
    def __init__(self, checkpoint_dir = None):
        # Build the checkpoint directory.
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(
                recursive_dirname(__file__, 4), 'checkpoints')
        self._checkpoint_dir = checkpoint_dir
        self._saved_checkpoints = dict()

    def fit(self,
            model,
            train_ds,
            val_ds,
            epochs = 50,
            log = False,
            **kwargs):
        """Trains the model on the provided data loaders."""
        # Set up the checkpoint tracking.
        save_all = kwargs.pop('save_all', False)
        self._checkpoint_dir = os.path.join(
            self._checkpoint_dir, kwargs['dataset'])
        os.makedirs(self._checkpoint_dir, exist_ok = True)
        if log:
            log_file = os.path.join(self._checkpoint_dir, 'log.txt')
            open(log_file, 'w').close()

        # Determine if a GPU exists.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Create the optimizer.
        optimizer = torch.optim.Adam(model.parameters())

        # Initialize training state variables.
        print(f"Training EfficientNetB4 on '{kwargs['dataset']}': "
              f"{epochs} epochs, writing checkpoints to {self._checkpoint_dir}.")
        model.train()

        # Start the training loop.
        for epoch in range(epochs):
            # Create epoch-state variables.
            train_loss, val_loss, acc = [], [], []

            # Iterate through the training data loader.
            model.train()
            for (images, labels) in tqdm(
                    iter(train_ds), desc = f"Epoch {epoch + 1}/{epochs}", file = sys.stdout):
                # Move the data to the correct device.
                images = images.to(device)
                labels = labels.to(device)

                # Train the model.
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    out = model(images)
                    loss = nn.CrossEntropyLoss()(out, labels)
                    train_loss.append(loss.item())

                    # Backprop and update weights.
                    loss.backward()
                    optimizer.step()

                # Compute accuracy.
                label_logits = torch.argmax(labels, 1)
                acc.append(accuracy(out, label_logits))

            # Iterate through the validation data loader.
            model.eval()
            for (images, labels) in tqdm(
                    iter(val_ds), "Validating"):
                # Move the data to the correct device.
                images = images.to(device)
                labels = labels.to(device)

                # Calculate the validation metrics.
                with torch.no_grad():
                    out = model(images)
                    loss = nn.CrossEntropyLoss()(out, labels)
                    val_loss.append(loss)

            # Print out metrics.
            final_loss = train_loss[-1]
            train_loss = torch.mean(torch.tensor(train_loss))
            final_val_loss = val_loss[-1]
            final_acc = sum(acc) / len(acc)
            print(f"Average Loss: {train_loss:.4f}, "
                  f"Average Accuracy: {final_acc:.4f}, ",
                  f"Epoch Loss: {final_loss:.4f}, "
                  f"Validation Loss: {final_val_loss:.4f}")

            # Save info to log file.
            if log:
                with open(log_file, 'a') as f: # noqa
                    f.write(f"Epoch {epoch}, "
                            f"Average Loss: {train_loss:.4f}, "
                            f"Epoch Loss: {final_loss:.4f}, "
                            f"Validation Loss: {final_val_loss:.4f}\n")

            # Save the checkpoint.
            save_path = os.path.join(
                           self._checkpoint_dir,
                           f"epoch_{epoch}_loss_{final_val_loss:.3f}.pth")
            torch.save(model.state_dict(), save_path)
            if not save_all:
                for path, loss in self._saved_checkpoints.items():
                    if loss > final_val_loss:
                        if os.path.exists(path):
                            os.remove(path)
            self._saved_checkpoints[save_path] = final_val_loss


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target):
    """Computes the accuracy between `output` and `target`."""
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = torch.topk(output, 1, 1)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:1].reshape(-1).float().sum(0, keepdim = True)
        return correct_k.mul_(100.0 / batch_size)


def execute():
    # Parse command line arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--dataset', type = str, help = "The name of the dataset.")
    ap.add_argument(
        '--save_all', action = 'store_true', default = False, help = 'Save all checkpoints.')
    ap.add_argument(
        '--checkpoint_dir', type = str, default = '/data2/amnjoshi/checkpoints',
        help = "The checkpoint directory to save to.")
    args = ap.parse_args()
    args.dataset = 'bean_disease_uganda'

    # Execute the program.
    train, val, test = build_loaders(args.dataset)
    net = EfficientNetB4Transfer(agml.data.source(args.dataset).num_classes)
    Trainer(checkpoint_dir = args.checkpoint_dir).fit(
        net, train_ds = train, val_ds = val,
        dataset = args.dataset, save_all = args.save_all
    )

if __name__ == '__main__':
    execute()

