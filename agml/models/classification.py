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

from tqdm.auto import tqdm

import torch
import torch.nn as nn


from tqdm.auto import tqdm

try:
    from transformers import ResNetForImageClassification, AdamW, get_scheduler
except ImportError:
    raise ImportError("To use image classification models in `agml.models`, you "
                      "need to install Huggingface Transformers first. You can "
                      "do this by running `pip install torchvision`.")

from agml.models.base import AgMLModelBase
from agml.models.tools import auto_move_data, imagenet_style_process
from agml.models.metrics.accuracy import Accuracy
from agml.utils.general import has_func


from transformers import ResNetForImageClassification
import torch
from torch import nn
from torchmetrics import Accuracy
from tqdm.auto import tqdm


class ClassificationModel(AgMLModelBase):
    """Wraps a `ResNetForImageClassification` model for agricultural image classification.

    When using the model for inference, you should use the `predict()` method
    on any set of input images. This method wraps the `forward()` call with
    additional steps which perform the necessary preprocessing on the inputs,
    including resizing, normalization, and batching, as well as additional
    post-processing on the outputs (such as converting one-hot labels to
    integer labels), which will allow for a streamlined inference pipeline.

    If you want to use the `forward()` method directly, so that you can just
    call `model(inputs)`, then make sure your inputs are properly processed.
    This can be done by calling `model.preprocess_input(inputs)` on the
    input list of images/single image and then passing that result to the model.
    This will also return a one-hot label feature vector instead of integer
    labels, in the case that you want further customization of the outputs.
    """
    serializable = frozenset(("model", "regression"))
    state_override = frozenset(("model",))

    _ml_task = 'image_classification'

    def __init__(self, model='resnet', num_classes=None, regression=False, **kwargs):
        # Construct the network and load in pretrained weights.
        super(ClassificationModel, self).__init__()
        self._regression = regression

        if model == 'resnet':
            self.net = ResNetForImageClassification.from_pretrained(
                'microsoft/resnet-50', num_labels=num_classes
            )

    @auto_move_data
    def forward(self, batch):
        return self.net(batch)

    @staticmethod
    def _preprocess_image(image, **kwargs):
        """Preprocesses a single input image to ResNet standards.

        Resizing, normalization (mean and std of ImageNet), and conversion to PyTorch tensor.
        """
        return imagenet_style_process(image, **kwargs)

    @staticmethod
    def preprocess_input(images=None, **kwargs) -> "torch.Tensor":
        """Preprocesses the input image to the specification of the ResNet model.
        """
        images = ClassificationModel._expand_input_images(images)
        return torch.stack(
            [ClassificationModel._preprocess_image(
                image, **kwargs) for image in images], dim=0)

    @torch.no_grad()
    def predict(self, images, **kwargs):
        """Runs `ResNetForImageClassification` inference on the input image(s).
        """
        images = self.preprocess_input(images, **kwargs)
        out = self.forward(images)
        if not self._regression:  # standard classification
            out = torch.argmax(out.logits, 1)  # ResNet logits need to be argmaxed
        if not kwargs.get('return_tensor_output', False):
            return self._to_out(torch.squeeze(out))
        return out

    def evaluate(self, loader, **kwargs):
        """Runs an accuracy evaluation on the given loader."""
        acc = Accuracy()
        bar = tqdm(loader, desc="Calculating Accuracy")
        for sample in bar:
            image, truth = sample
            pred_label = self.predict(image, return_tensor_output=True, **kwargs)
            acc.update(pred_label, truth)
            bar.set_postfix({'accuracy': acc.compute().item()})

        return acc.compute().item()

    def run_training(self,
                     dataset=None,
                     *,
                     epochs=50,
                     loss=None,
                     metrics=None,
                     optimizer=None,
                     lr_scheduler=None,
                     lr=None,
                     batch_size=8,
                     loggers=None,
                     train_dataloader=None,
                     val_dataloader=None,
                     test_dataloader=None,
                     use_cpu=False,
                     save_dir=None,
                     experiment_name=None,
                     **kwargs):
        """Trains a ResNet image classification model."""
        from agml.models.training.basic_trainers import train_classification

        return train_classification(
            self,
            dataset=dataset,
            epochs=epochs,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            lr=lr,
            batch_size=batch_size,
            loggers=loggers,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            use_cpu=use_cpu,
            save_dir=save_dir,
            experiment_name=experiment_name,
            **kwargs
        )

    def _prepare_for_training(self, loss='ce', metrics=(), optimizer=None, **kwargs):
        """Prepares this classification model for training."""

        # Initialize the loss
        if loss == 'ce':
            if self._num_classes == 1:
                self.loss = nn.BCEWithLogitsLoss()
            else:
                self.loss = nn.CrossEntropyLoss()
        else:
            if not isinstance(loss, nn.Module) or not callable(loss):
                raise TypeError(f"Expected a callable loss function, but got '{type(loss)}'.")

        # Initialize the metrics.
        metric_collection = []
        if len(metrics) > 0:
            for metric in metrics:
                # Check if it is a valid torchmetrics metric.
                if isinstance(metric, str):
                    from torchmetrics import classification as class_metrics

                    if metric == 'accuracy':
                        from agml.models.metrics.accuracy import Accuracy
                        metric_collection.append(['accuracy', Accuracy()])
                    else:
                        metric_collection.append([metric, getattr(class_metrics, metric)()])
                elif isinstance(metric, nn.Module):
                    metric_collection.append(metric)
                else:
                    raise TypeError(f"Expected a metric name or a metric class, but got '{type(metric)}'.")

        self._metrics = metric_collection

        # Initialize the optimizer/learning rate scheduler.
        if isinstance(optimizer, str):
            optimizer_class = optimizer.capitalize()
            optimizer = getattr(torch.optim, optimizer_class)(self.parameters(), lr=kwargs.get('lr', 1e-3))
        elif isinstance(optimizer, torch.optim.Optimizer):
            pass
        else:
            raise TypeError(f"Expected an optimizer name or a torch optimizer, but got '{type(optimizer)}'.")

        scheduler = kwargs.get('lr_scheduler', None)
        if scheduler is not None:
            if not isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
                raise TypeError(f"Expected a torch LR scheduler, but got '{type(scheduler)}'.")

        self._optimization_parameters = {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }

    def configure_optimizers(self):
        opt = self._optimization_parameters['optimizer']
        scheduler = self._optimization_parameters['lr_scheduler']
        if scheduler is None:
            return opt
        return [opt], [scheduler]

    def training_step(self, batch, *args, **kwargs):
        x, y = batch
        y_pred = self(x)

        loss = self.loss(y_pred.logits, y)  # ResNetForImageClassification returns logits
        for metric_name, metric in self._metrics:
            metric.update(y_pred.logits, y)
            self.log(metric_name, metric.compute().item(), prog_bar=True)

        return {'loss': loss}

    def validation_step(self, batch, *args, **kwargs):
        x, y = batch
        y_pred = self(x)

        val_loss = self.loss(y_pred.logits, y)
        self.log('val_loss', val_loss.item(), prog_bar=True)
        for metric_name, metric in self._metrics:
            metric.update(y_pred.logits, y)
            self.log('val_' + metric_name, metric.compute().item(), prog_bar=True)

        return {'val_loss': val_loss}

    def test_step(self, batch, *args, **kwargs):
        x, y = batch
        y_pred = self(x)

        test_loss = self.loss(y_pred.logits, y)
        self.log('test_loss', test_loss.item(), prog_bar=True)
        for metric_name, metric in self._metrics:
            metric.update(y_pred.logits, y)
            self.log('test_' + metric_name, metric.compute().item(), prog_bar=True)

        return {'test_loss': test_loss}






