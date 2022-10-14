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

from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4

from agml.models.base import AgMLModelBase
from agml.models.tools import auto_move_data, imagenet_style_process
from agml.models.metrics.accuracy import Accuracy
from agml.data.public import source


class EfficientNetB4Transfer(nn.Module):
    """Wraps an EfficientDetB4 model with a classification head."""
    def __init__(self, num_classes):
        super(EfficientNetB4Transfer, self).__init__()
        self.base = efficientnet_b4(pretrained = False)
        self.l1 = nn.Linear(1000, 256)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(256, num_classes)

    def forward(self, x, **kwargs): # noqa
        x = self.base(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.l1(x)))
        return self.l2(x)


class ClassificationModel(AgMLModelBase):
    """Wraps an `EfficientNetB4` model for agricultural image classification.

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

    This model can be subclassed in order to run a full training job; the
    actual transfer `EfficientNetB4` model can be accessed through the
    parameter `net`, and you'll need to implement methods like `training_step`,
    `configure_optimizers`, etc. See PyTorch Lightning for more information.
    """
    serializable = frozenset(("model", ))
    state_override = serializable

    def __init__(self, num_classes):
        # Construct the network and load in pretrained weights.
        super(ClassificationModel, self).__init__()
        self._num_classes = num_classes
        self.net = self._construct_sub_net(num_classes)

    @auto_move_data
    def forward(self, batch):
        return self.net(batch)

    @staticmethod
    def _construct_sub_net(num_classes):
        return EfficientNetB4Transfer(num_classes)

    @staticmethod
    def _preprocess_image(image):
        """Preprocesses a single input image to EfficientNet standards.

        The preprocessing steps are applied logically; if the images
        are passed with preprocessing already having been applied, for
        instance, the images are already resized or they are already been
        normalized, the operation is not applied again, for efficiency.

        Preprocessing includes the following steps:

        1. Resizing the image to size (224, 224).
        2. Performing normalization with ImageNet parameters.
        3. Converting the image into a PyTorch tensor format.

        as well as other intermediate steps such as adding a channel
        dimension for two-channel inputs, for example.
        """
        return imagenet_style_process(image)

    @staticmethod
    def preprocess_input(images = None) -> "torch.Tensor":
        """Preprocesses the input image to the specification of the model.

        This method takes in a set of inputs and preprocesses them into the
        expected format for the `EfficientNetB4` image classification model.
        There are a variety of inputs which are accepted, including images,
        image paths, as well as fully-processed image tensors. The inputs
        are expanded and standardized, then run through a preprocessing
        pipeline which formats them into a single tensor ready for the model.

        Preprocessing steps include normalization, resizing, and converting
        to the channels-first format used by PyTorch models. The output
        of this method will be a single `torch.Tensor`, which has shape
        [N, C, H, W], where `N` is the batch dimension. If only a single
        image is passed, this will have a value of 1.

        This method is largely beneficial when you just want to preprocess
        images into the specification of the model, without getting the output.
        Namely, `predict()` is essentially just a wrapper around this method
        and `forward()`, so you can run this externally and then run `forward()`
        to get the original model outputs, without any extra post-processing.

        Parameters
        ----------
        images : Any
            One of the following formats (and types):
                1. A single image path (str)
                2. A list of image paths (List[str])
                3. A single image (np.ndarray, torch.Tensor)
                4. A list of images (List[np.ndarray, torch.Tensor])
                5. A batched tensor of images (np.ndarray, torch.Tensor)

        Returns
        -------
        A 4-dimensional, preprocessed `torch.Tensor`.
        """
        images = ClassificationModel._expand_input_images(images)
        return torch.stack(
            [ClassificationModel._preprocess_image(
                image) for image in images], dim = 0)

    @torch.no_grad()
    def predict(self, images):
        """Runs `EfficientNetB4` inference on the input image(s).

        This method is the primary inference method for the model; it
        accepts a set of input images (see `preprocess_input()` for a
        detailed specification on the allowed input parameters), then
        preprocesses them to the model specifications, forward passes
        them through the model, and finally returns the predictions.

        In essence, this method is a wrapper for `forward()` that allows
        for passing a variety of inputs. If, on the other hand, you
        have pre-processed inputs and simply want to forward pass through
        the model without having to spend computational time on what
        is now unnecessary preprocessing, you can simply call `forward()`
        and then run `torch.argmax()` on the outputs to get predictions.

        Parameters
        ----------
        images : Any
            See `preprocess_input()` for the allowed input images.

        Returns
        -------
        A `np.ndarray` with integer labels for each image.
        """
        images = self.preprocess_input(images)
        return self._to_out(torch.squeeze(torch.argmax(self.forward(images), 1)))

    def evaluate(self, loader, **kwargs):
        """Runs an accuracy evaluation on the given loader.

        This method will loop over the `AgMLDataLoader` and compute accuracy.

        Parameters
        ----------
        loader : AgMLDataLoader
            A semantic segmentation loader with the dataset you want to evaluate.

        Returns
        -------
        The final calculated accuracy.
        """
        # Construct the metric and run the calculations.
        acc = Accuracy()
        bar = tqdm(loader, desc = "Calculating Accuracy")
        for sample in bar:
            image, truth = sample
            pred_label = self.predict(image)
            acc.update([pred_label], [truth])
            bar.set_postfix({'accuracy': acc.compute().numpy().item()})

        # Compute the final accuracy.
        return acc.compute().numpy().item()




