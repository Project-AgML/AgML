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
from typing import List
from numbers import Number
from dataclasses import dataclass

import numpy as np

from agml.framework import AgMLSerializable
from agml.synthetic.config import load_default_helios_configuration


class Parameters:
    """Base class for parameters, to enable runtime type checks."""
    def __post_init__(self):
        self._block_new_attributes = True

    def __setattr__(self, key, value):
        # Don't allow the assignment of new attributes.
        if key not in self.__dict__.keys():
            if not hasattr(self, '_block_new_attributes'):
                super().__setattr__(key, value)
                return
            raise AttributeError(f"Cannot assign new attributes '{key}' "
                                 f"to class {self.__class__.__name__}.")

        # Check if the type of the value matches that of the key.
        annotation = self.__annotations__[key]
        if not isinstance(value, annotation):
            raise TypeError(
                f"Expected a value of type ({annotation}) for attribute "
                f"'{key}', instead got '{value}' of type ({type(value)}).")
        super().__setattr__(key, value)


@dataclass
class CanopyParameters(Parameters):
    """Stores canopy-specific parameters for Helios."""
    leaf_length: Number
    leaf_subdivisions: List[Number]
    leaf_texture_file: str
    stem_color: List[Number]
    stem_subdivisions: Number
    stems_per_plant: Number
    stem_radius: Number
    plant_height: Number
    fruit_radius: Number
    fruit_texture_file: str
    fruit_subdivisions: Number
    clusters_per_stem: Number
    plant_spacing: Number
    row_spacing: Number
    plant_count: List[Number]
    canopy_origin: List[Number]
    canopy_rotation: Number


@dataclass
class CameraParameters(Parameters):
    """Stores camera parameters for Helios."""
    image_resolution: List[Number]
    camera_position: List[Number]
    camera_lookat: List[Number]


@dataclass
class LiDARParameters(Parameters):
    """Stores LiDAR parameters for Helios."""
    origin: List[Number]
    size: List[Number]
    thetaMin: Number
    thetaMax: Number
    phiMin: Number
    phiMax: Number
    exitDiameter: Number
    beamDivergence: Number
    ASCII_format: str


class HeliosOptions(AgMLSerializable):
    """Stores a set of parameter options for a `HeliosDataGenerator`.

    The primary exposed options are the `canopy_parameters` (as well as similar
    options for camera position and LiDAR, `camera/lidar_parameters' specifically),
    as well as the `canopy_parameter_ranges` (as well as a similar set of options
    for camera and LiDAR, once again).

    The `HeliosOptions` is instantiated with the name of the canopy type that you
    want to generate; from there, the parameters and ranges can be accessed through
    properties, which are set to their default values as loaded from the Helios
    configuration upon instantiating the class,

    This `HeliosOptions` object, once configured, can then be passed directly to
    a `HeliosDataGenerator` upon instantiation, and be used to generate synthetic
    data according to its specification. The options can be edited and then the
    generation process run again to obtain a new set of data.

    Parameters
    ----------
    canopy : str
        The type of plant canopy to be used in generation.
    """

    # The default configuration parameters are loaded directly from
    # the `helios_config.json` file which is constructed each time
    # Helios is installed or updated.
    _default_config = load_default_helios_configuration()

    def __init__(self, canopy = None):
        # Check that the provided canopy is valid.
        self._initialize_canopy(canopy)

    def _initialize_canopy(self, canopy):
        """Initializes Helios options from the provided canopy."""
        if canopy not in self._default_config['canopy']['types']:
            raise ValueError(
                f"Received invalid canopy type '{canopy}', expected "
                f"one of: {self._default_config['canopy']['types']}.")

        # Get the parameters and ranges corresponding to the canopy type.
        self._canopy_parameters = \
            CanopyParameters(**self._default_config['canopy']['parameters'][canopy])
        self._camera_parameters = \
            CameraParameters(**self._default_config['camera']['parameters'])
        self._lidar_parameters = \
            LiDARParameters(**self._default_config['lidar']['parameters'])

    @property
    def canopy(self) -> CanopyParameters:
        return self._canopy_parameters

    @property
    def camera(self) -> CameraParameters:
        return self._camera_parameters

    @property
    def lidar(self) -> LiDARParameters:
        return self._lidar_parameters

