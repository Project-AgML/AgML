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

from typing import Callable
from functools import wraps

import torch
import numpy as np
import albumentations as A

from agml.backend.tftorch import is_array_like


def imagenet_style_process(image, size = None):
    """Preprocesses a single input image to ImageNet standards.

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
    # Convert the image to a NumPy array.
    if is_array_like(image) and hasattr(image, 'numpy'):
        image = image.numpy()

    # Add a channel dimension for grayscale imagery.
    if image.ndim == 2:
        image = np.expand_dims(image, axis = -1)

    # If the image is already in channels-first format, convert
    # it back temporarily until preprocessing has concluded.
    if image.shape[0] <= 3:
        image = np.transpose(image, (1, 2, 0))

    # Resize the image to ImageNet standards.
    h = w = 224
    if size is not None:
        h, w = size
    rz = A.Resize(height = h, width = w)
    if image.shape[0] != h or image.shape[1] != w:
        image = rz(image = image)['image']

    # Normalize the image to ImageNet standards.
    if 1 <= image.max() <= 255:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image = image.astype(np.float32) / 255.
        mean = np.array(mean, dtype = np.float32)
        std = np.array(std, dtype = np.float32)
        denominator = np.reciprocal(std, dtype = np.float32)
        image = (image - mean) * denominator

    # Convert the image into a PyTorch tensor.
    image = torch.from_numpy(image).permute(2, 0, 1)

    # Return the processed image.
    return image


# Ported from PyTorch Lightning v1.3.0.
def auto_move_data(fn: Callable) -> Callable:
    """
    Decorator for :class:`~pytorch_lightning.core.lightning.LightningModule` methods for which
    input arguments should be moved automatically to the correct device.
    It has no effect if applied to a method of an object that is not an instance of
    :class:`~pytorch_lightning.core.lightning.LightningModule` and is typically applied to ``__call__``
    or ``forward``.

    Args:
        fn: A LightningModule method for which the arguments should be moved to the device
            the parameters are on.

    Example::

        # directly in the source code
        class LitModel(LightningModule):

            @auto_move_data
            def forward(self, x):
                return x

        # or outside
        LitModel.forward = auto_move_data(LitModel.forward)

        model = LitModel()
        model = model.to('cuda')
        model(torch.zeros(1, 3))

        # input gets moved to device
        # tensor([[0., 0., 0.]], device='cuda:0')

    """

    @wraps(fn)
    def auto_transfer_args(self, *args, **kwargs):
        from pytorch_lightning import LightningModule

        if not isinstance(self, LightningModule):
            return fn(self, *args, **kwargs)

        args, kwargs = self.transfer_batch_to_device(
            (args, kwargs), device = self.device, dataloader_idx = None) # noqa
        return fn(self, *args, **kwargs)

    return auto_transfer_args


