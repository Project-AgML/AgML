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

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import JaccardIndex as IoU
from torchvision.models.segmentation import deeplabv3_resnet50

from agml.models.base import AgMLModelBase
from agml.models.benchmarks import BenchmarkMetadata
from agml.models.tools import auto_move_data, imagenet_style_process
from agml.data.public import source
from agml.utils.general import resolve_list_value
from agml.utils.image import resolve_image_size
from agml.viz.masks import visualize_overlaid_masks, visualize_image_and_mask


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
    serializable = frozenset((
        "net", "num_classes", "conf_thresh", "image_size"))
    state_override = frozenset(("net",))

    def __init__(self,
                 num_classes = 1,
                 image_size = 512,
                 **kwargs):
        # Construct the network and load in pretrained weights.
        super(SegmentationModel, self).__init__()

        # If being initialized by a subclass, then don't do any of
        # model construction logic (since that's already been done).
        if not kwargs.get('model_initialized', False):
            self._num_classes = num_classes
            self._image_size = resolve_image_size(image_size)
            self.net = self._construct_sub_net(num_classes)
            if self._num_classes == 1:
                conf_threshold = kwargs.get('conf_threshold', 0.2)
                if not 0 < conf_threshold < 1:
                    raise ValueError(
                        "The given confidence threshold "
                        "must be between 0 and 1.")
                self._conf_thresh = conf_threshold

        # By default, the model starts in inference mode.
        self.eval()

    @auto_move_data
    def forward(self, batch):
        return self.net(batch)

    @staticmethod
    def _construct_sub_net(num_classes):
        return DeepLabV3Transfer(num_classes)

    @staticmethod
    def _preprocess_image(image, image_size):
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
        return imagenet_style_process(image, size = image_size)

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
                image, self._image_size) for image in images], dim = 0)
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
            out[out >= self._conf_thresh] = 1
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

    def show_prediction(self, image, overlay = False, **kwargs):
        """Shows the output predictions for one input image.

        This method is useful for instantly visualizing the predictions
        for a single input image. It accepts a single input image (or
        any type of valid 'image' input, as described in the method
        `preprocess_input()`), and then runs inference on that input
        image and displays its predictions in a matplotlib window.

        Parameters
        ----------
        image : Any
            See `preprocess_input()` for allowed types of inputs.
        method : str
            Either `True` for overlaid masks, or `False` for
            visualizing the mask separately from the image.
        kwargs
            Visualization keyword arguments.

        Returns
        -------
        The matplotlib figure containing the image.
        """
        image = self._expand_input_images(image)[0]
        mask = self.predict(image)
        if overlay:
            return visualize_overlaid_masks(image, mask, **kwargs)
        return visualize_image_and_mask(image, mask, **kwargs)

    def load_benchmark(self, dataset):
        """Loads a benchmark for the given semantic segmentation dataset.

        This method is used to load pretrained weights for a specific AgML dataset.
        In essence, it serves as a wrapper for `load_state_dict`, directly getting
        the model from its save path in the AWS storage bucket. You can then use the
        `benchmark` property to access the metric value of the benchmark, as well as
        additional training parameters which you can use to train your own models.

        Parameters
        ----------
        dataset : str
            The name of the semantic segmentation benchmark to load.

        Notes
        -----
        If the benchmark has a different number of classes than this input model, then
        this method will raise an error. This issue may be adapted in the future.
        """
        if source(dataset).tasks.ml != 'semantic_segmentation':
            raise ValueError(
                f"You are trying to load a benchmark for a "
                f"{source(dataset).tasks.ml} task ({dataset}) "
                f"in a semantic segmentation model.")

        # Number of classes must be the same for semantic segmentation.
        if source(dataset).num_classes != self._num_classes:
            raise ValueError(
                f"You cannot load a benchmark for a dataset '{dataset}' "
                f"with {source(dataset).num_classes} classes, while your "
                f"model has {self._num_classes} classes.")

        # Load the benchmark.
        state = self._get_benchmark(dataset)
        self.load_state_dict(state)
        self._benchmark = BenchmarkMetadata(dataset)

    def evaluate(self, loader, **kwargs):
        """Runs a mean intersection over union evaluation on the given loader.

        This method will loop over the provided `AgMLDataLoader` and compute
        the mean intersection over union (mIOU).

        Parameters
        ----------
        loader : AgMLDataLoader
            A semantic segmentation loader with the dataset you want to evaluate.

        Returns
        -------
        The final calculated mIoU.
        """
        # Construct the metric and run the calculations.
        iou = IoU(num_classes = self._num_classes + 1)
        bar = tqdm(loader, desc = "Calculating Mean Intersection Over Union")
        for sample in bar:
            image, truth = sample
            pred_mask = self.predict(image)
            if pred_mask.ndim == 3:
                pred_mask = np.transpose(pred_mask, (2, 0, 1))
                truth = np.transpose(truth, (2, 0, 1))
            iou(torch.from_numpy(pred_mask).int().unsqueeze(0),
                torch.from_numpy(truth).unsqueeze(0))
            bar.set_postfix({'miou': iou.compute().numpy().item()})

        # Compute the final mIoU.
        return iou.compute().numpy().item()

