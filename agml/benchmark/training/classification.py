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
import argparse

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4
from torchmetrics.functional import accuracy

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
        self.out = nn.Softmax()

    def forward(self, x, **kwargs):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.l1(x)))
        x = self.out(self.l2(x))
        return x

# Helper method (convenience).
def export_torch(obj):
    return obj.export_torch(num_workers = 8)

# Build the data loaders.
def build_loaders(name):
    loader = agml.data.AgMLDataLoader(name)
    loader.split(train = 0.8, val = 0.1, test = 0.1)
    loader.batch(batch_size = 16)
    loader.resize_images('imagenet')
    loader.labels_to_one_hot()
    loader.transform(
        lambda x: x / 255
    )
    train_data = loader.train_data
    train_data.transform(
        transform = A.Compose([
            A.RandomRotate90(),
        ])
    )
    train_ds = train_data.export_torch(
        num_workers = os.cpu_count())
    val_ds = loader.val_data.export_torch(
        num_workers = os.cpu_count())
    test_ds = loader.test_data.export_torch(
        num_workers = os.cpu_count())
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
                            f"Average Loss: {avg_epoch_loss:.4f}, "
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

    # Execute the program.
    train, val, test = build_loaders(args.dataset)
    net = EfficientNetB4Transfer(agml.data.source(args.dataset).num_classes)
    Trainer(checkpoint_dir = args.checkpoint_dir).fit(
        net, train_ds = train, val_ds = val,
        dataset = args.dataset, save_all = args.save_all
    )

if __name__ == '__main__':
    execute()

