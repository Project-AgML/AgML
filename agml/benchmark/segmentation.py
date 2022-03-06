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

from typing import final

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50

from agml.benchmark.model import AgMLModelBase
from agml.benchmark.tools import auto_move_data, imagenet_style_process
from agml.data.public import source
from agml.utils.general import resolve_list_value


class DeepLabV3Transfer(nn.Module):
    """Wraps a DeepLabV3 model with the right number of classes."""
    def __init__(self, num_classes):
        super(DeepLabV3Transfer, self).__init__()
        self.base = deeplabv3_resnet50(
            pretrained = False,
            num_classes = num_classes)
        
    def forward(self, x, **kwargs): # noqa
        return self.base(x)['out']


class SegmentationModel(AgMLModelBase):
    """Wraps a `DeepLabV3` model for agricultural semantic segmentation.

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

    def __init__(self, dataset):
        # Construct the network and load in pretrained weights.
        super(SegmentationModel, self).__init__()
        self.net = self._construct_sub_net(dataset)

    @auto_move_data
    def forward(self, batch):
        return self.net(batch)

    @staticmethod
    def _construct_sub_net(dataset):
        return DeepLabV3Transfer(source(dataset).num_classes)

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
        return imagenet_style_process(image, size = (512, 512))

    @final
    def preprocess_input(self, images, return_shapes = False):
        """Preprocesses the input image to the specification of the model.

        This method takes in a set of inputs and preprocesses them into the
        expected format for the `DeepLabV3` semantic segmentation model.
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
        return_shapes : bool
            Whether to return the original shapes of the input images.

        Returns
        -------
        A 4-dimensional, preprocessed `torch.Tensor`. If `return_shapes`
        is set to True, it also returns the original shapes of the images.
        """
        images = self._expand_input_images(images)
        shapes = self._get_shapes(images)
        images = torch.stack(
            [self._preprocess_image(
                image) for image in images], dim = 0)
        if return_shapes:
            return images, shapes
        return images

    @torch.no_grad()
    def predict(self, images):
        """Runs `DeepLabV3` inference on the input image(s).

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
        A list of `np.ndarray`s with resized output masks.
        """
        # Process the images and run inference.
        images, shapes = self.preprocess_input(images, return_shapes = True)
        out = torch.sigmoid(self.forward(images))

        # Post-process the output masks to a valid format.
        if out.shape[1] == 1: # binary class predictions
            out[out >= 0.2] = 1
            out[out != 1] = 0
            out = torch.squeeze(out, dim = 1)
        else: # multi-class predictions to integer labels
            out = torch.argmax(out, 1)

        # Resize the masks to their original shapes.
        masks = []
        for mask, shape in zip(
                torch.index_select(out, 0, torch.arange(len(out))), shapes):
            masks.append(self._to_out(torch.squeeze(F.interpolate(
                torch.unsqueeze(torch.unsqueeze(mask, 0), 0).float(),
                size = shape).int())))
        return resolve_list_value(masks)




