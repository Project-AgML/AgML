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
import abc
from typing import List, Union, overload

import cv2
import numpy as np

import torch
from pytorch_lightning import LightningModule

from agml.framework import AgMLSerializable
from agml.backend.config import model_save_path
from agml.backend.tftorch import is_array_like
from agml.utils.image import imread_context
from agml.utils.downloads import download_model
from agml.models.benchmarks import BenchmarkMetadata


class AgMLModelBase(AgMLSerializable, LightningModule):
    """Base class for all AgML pretrained models.

    All pretrained model variants in AgML inherit from this base class,
    which provides common methods which each use, such as weight loading
    and image input preprocessing, as well as other stubs for common methods.
    """

    def __init__(self):
        self._benchmark = BenchmarkMetadata(None)
        super(AgMLModelBase, self).__init__()

    @property
    def original(self):
        """Returns the original model architecture (without weights)."""
        return self.net.base

    @overload
    def preprocess_input(self, images: str) -> "torch.Tensor":
        ...

    @overload
    def preprocess_input(self, images: List[str]) -> "torch.Tensor":
        ...

    @overload
    def preprocess_input(self, images: Union[np.ndarray, torch.Tensor]) -> "torch.Tensor":
        ...

    @overload
    def preprocess_input(self, images: List[Union[np.ndarray, torch.Tensor]]) -> "torch.Tensor":
        ...

    @abc.abstractmethod
    def preprocess_input(self, *args, **kwargs):
        """Preprocesses input images to model specifications."""
        raise NotImplementedError

    @overload
    def predict(self, images: str) -> "torch.Tensor":
        ...

    @overload
    def predict(self, images: List[str]) -> "torch.Tensor":
        ...

    @overload
    def predict(self, images: Union[np.ndarray, torch.Tensor]) -> "torch.Tensor":
        ...

    @overload
    def predict(self, images: List[Union[np.ndarray, torch.Tensor]]) -> "torch.Tensor":
        ...

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        """Runs model inference on input image(s)."""
        raise NotImplementedError

    @staticmethod
    def _expand_input_images(images):
        """Expands the input list of images to a specification.

        This is particularly useful because the model accepts numerous
        inputs, ranging from a list of image paths all the way to an
        already pre-processed image batch. This method standardizes all
        inputs to a common format, in particular, a list of all of the
        images that are going to be then passed to the input preprocessing
        method, before being passed through the model for inference.
        """
        # First check for a path or a list of paths, for speed.
        if isinstance(images, str):
            with imread_context(images) as image:
                return [cv2.cvtColor(image, cv2.COLOR_BGR2RGB), ]
        elif isinstance(images, list) and isinstance(images[0], str):
            parsed_images = []
            for path in images:
                with imread_context(path) as image:
                    parsed_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            return parsed_images

        # Then check if we already have read-in images, either one
        # single image or a batch of images.
        if is_array_like(images):
            # Check if there is one single input image, or a batch of input
            # images. If there is a single input image, just return this.
            if images.ndim == 3:
                return [images, ]

            # Check if we have a batch of images first. This check is
            # done by seeing if the input is 4-dimensional.
            if images.ndim == 4:
                # If so, unstack the images along the first dimension.
                return [i for i in images]

        # Finally, the only remaining viable input type is a list of images.
        if isinstance(images, list) and is_array_like(images[0]):
            return images

        # Otherwise, we need to raise an error.
        raise TypeError(
            "Expected an input of a list of paths or images, a "
            "single path or image, or a batched image tensor for "
            f"preprocessing inputs, instead got {type(images)}.")

    @staticmethod
    def _to_out(tensor: "torch.Tensor") -> "torch.Tensor":
        return tensor.detach().cpu().numpy()

    @staticmethod
    def _get_shapes(images: list) -> list:
        """Gets the height and width of each of the input images."""
        shapes = []
        for image in images:
            if image.ndim == 2:
                shapes.append(image.shape)
                continue
            if image.shape[0] <= 3: # channels first
                shapes.append(image.shape[1:])
            else: # channels last
                shapes.append(image.shape[:2])
        return shapes

    @property
    def benchmark(self):
        """Information about the loaded benchmark."""
        return self._benchmark

    @benchmark.setter
    def benchmark(self, value):
        self._benchmark = value

    @staticmethod
    def _get_benchmark(name):
        """Returns the `state_dict` for a pretrained model benchmark."""
        # Check if the benchmark exists; if not, download it.
        benchmark_path = os.path.join(model_save_path(), name + '.pth')
        if not os.path.exists(benchmark_path):
            download_model(name, os.path.dirname(benchmark_path))

        # Load the benchmark.
        return torch.load(benchmark_path, map_location = 'cpu')

    @abc.abstractmethod
    def load_benchmark(self, dataset: str):
        """Loads a benchmark for the given AgML dataset."""
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, loader, **kwargs):
        """Evaluates the model on the given loader."""
        raise NotImplementedError



