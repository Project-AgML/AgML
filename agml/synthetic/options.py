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
from enum import Enum
from numbers import Number
from dataclasses import dataclass, fields, asdict
from typing import List, Union, Sequence, TypeVar

from agml.framework import AgMLSerializable
from agml.synthetic.config import load_default_helios_configuration, verify_helios
from agml.synthetic.tools import generate_camera_positions


class AnnotationType(Enum):
    """The type of annotation to use in generation."""
    object_detection: str = "object_detection"
    semantic_segmentation: str = "semantic_segmentation"
    instance_segmentation: str = "instance_segmentation"
    none: str = "none"


class SimulationType(Enum):
    """The simulation render (RGB vs. LiDAR) that is generated."""
    RGB: str = "rgb"
    LiDAR: str = "lidar"


NumberOrMaybeList = TypeVar('NumberOrMaybeList', Number, List[Number])


@dataclass(repr = False)
class Parameters:
    """Base class for parameters, to enable runtime type checks."""
    def __post_init__(self):
        # Remove all parameters which don't belong to this class. We know
        # they don't belong if they are still `None` after initialization.
        self_dict = self.__dict__.copy()
        for attr in self_dict:
            if self.__dict__[attr] is None:
                delattr(self, attr)
        self._block_new_attributes = True

    def __repr__(self):
        # This is custom-defined to exclude optional unused attributes.
        defined_values = (
            (f.name, getattr(self, f.name))
            for f in fields(self)
            if getattr(self, f.name) != f.default)
        value_repr = ", ".join(f"{name}={value}" for name, value in defined_values)
        return f"{self.__class__.__qualname__}({value_repr})"

    def __setattr__(self, key, value):
        # Don't allow the assignment of new attributes.
        if key not in self.__dict__.keys():
            if not hasattr(self, '_block_new_attributes'):
                super().__setattr__(key, value)
                return
            raise AttributeError(f"Cannot assign new attributes '{key}' "
                                 f"to class {self.__class__.__name__}.")

        # Check if the type of the value matches that of the key.
        try:
            annotation = self.__annotations__[key].__origin__
        except AttributeError:
            # enables type checks for subscripted generics
            annotation = (int, float)
        if not isinstance(value, annotation):
            raise TypeError(
                f"Expected a value of type ({annotation}) for attribute "
                f"'{key}', instead got '{value}' of type ({type(value)}).")
        super().__setattr__(key, value)


@dataclass(repr = False)
class CanopyParameters(Parameters):
    """Stores canopy-specific parameters for Helios.

    This class is a super-parameter class which contains all of the parameters
    across all of the canopies within Helios, though each of the individual canopies
    may only use a fraction of these specific canopies. This is to ensure that
    different canopy parameters auto-complete when actually writing code.
    """
    leaf_length: Number                   = None
    leaf_width: Number                    = None
    leaf_size: Number                     = None
    leaf_subdivisions: List[Number]       = None
    leaf_texture_file: str                = None
    leaf_color: str                       = None
    leaf_angle_distribution: str          = None
    leaf_area_index: Number               = None
    leaf_area_density: Number             = None
    leaf_spacing_fraction: Number         = None
    leaflet_length: Number                = None
    stem_color: List[Number]              = None
    stem_subdivisions: Number             = None
    stems_per_plant: Number               = None
    stem_radius: Number                   = None
    stem_length: Number                   = None
    plant_height: Number                  = None
    grape_radius: Number                  = None
    grape_color: List[Number]             = None
    grape_subdivisions: Number            = None
    fruit_color: List[Number]             = None
    fruit_radius: Number                  = None
    fruit_subdivisions: Number            = None
    fruit_texture_file: os.PathLike       = None
    wood_texture_file: os.PathLike        = None
    wood_subdivisions: Number             = None
    clusters_per_stem: Number             = None
    plant_spacing: Number                 = None
    row_spacing: Number                   = None
    level_spacing: Number                 = None
    plant_count: List[Number]             = None
    canopy_origin: List[Number]           = None
    canopy_rotation: Number               = None
    canopy_height: Number                 = None
    canopy_extent: List[Number]           = None
    canopy_configuration: str             = None
    base_height: Number                   = None
    crown_radius: Number                  = None
    cluster_radius: Number                = None
    cluster_height_max: Number            = None
    trunk_height: Number                  = None
    trunk_radius: Number                  = None
    cordon_height: Number                 = None
    cordon_radius: Number                 = None
    cordon_spacing: Number                = None
    shoot_length: Number                  = None
    shoot_radius: Number                  = None
    shoots_per_cordon: Number             = None
    shoot_angle: Number                   = None
    shoot_angle_tip: Number               = None
    shoot_angle_base: Number              = None
    shoot_color: List[Number]             = None
    shoot_subdivisions: NumberOrMaybeList = None
    pod_color: List[Number]               = None
    pod_subdivisions: NumberOrMaybeList   = None
    pod_length: None                      = None
    germination_probability: Number       = None
    needle_width: Number                  = None
    needle_length: Number                 = None
    needle_color: List[Number]            = None
    needle_subdivisions: List[Number]     = None
    branch_length: Number                 = None
    branches_per_level: Number            = None
    buffer: str                           = None

    # sorghum options
    sorghum_stage: Number                 = None
    s1_stem_length: Number                = None
    s1_stem_radius: Number                = None
    s1_stem_subdivisions: Number          = None
    s1_leaf_size1: List[Number]           = None
    s1_leaf_size2: List[Number]           = None
    s1_leaf_size3: List[Number]           = None
    s1_leaf1_angle: Number                = None
    s1_leaf2_angle: Number                = None
    s1_leaf3_angle: Number                = None
    s1_leaf_subdivisions: List[Number]    = None
    s1_leaf_texture_file: str             = None
    s2_stem_length: Number                = None
    s2_stem_radius: Number                = None
    s2_stem_subdivisions: Number          = None
    s2_leaf_size1: List[Number]           = None
    s2_leaf_size2: List[Number]           = None
    s2_leaf_size3: List[Number]           = None
    s2_leaf_size4: List[Number]           = None
    s2_leaf_size5: List[Number]           = None
    s2_leaf1_angle: Number                = None
    s2_leaf2_angle: Number                = None
    s2_leaf3_angle: Number                = None
    s2_leaf4_angle: Number                = None
    s2_leaf5_angle: Number                = None
    s2_leaf_subdivisions: List[Number]    = None
    s2_leaf_texture_file: str             = None
    s3_stem_length: Number                = None
    s3_stem_radius: Number                = None
    s3_stem_subdivisions: Number          = None
    s3_leaf_size: List[Number]            = None
    s3_leaf_subdivisions: List[Number]    = None
    s3_number_of_leaves: Number           = None
    s3_mean_leaf_angle: Number            = None
    s3_leaf_texture_file: str             = None
    s4_stem_length: Number                = None
    s4_stem_radius: Number                = None
    s4_stem_subdivisions: Number          = None
    s4_panicle_size: List[Number]         = None
    s4_panicle_subdivisions: Number       = None
    s4_seed_texture_file: str             = None
    s4_leaf_size: List[Number]            = None
    s4_leaf_subdivisions: List[Number]    = None
    s4_number_of_leaves: Number           = None
    s4_mean_leaf_angle: Number            = None
    s4_leaf_texture_file: str             = None
    s5_stem_length: Number                = None
    s5_stem_radius: Number                = None
    s5_stem_bend: Number                  = None
    s5_stem_subdivisions: Number          = None
    s5_panicle_size: List[Number]         = None
    s5_panicle_subdivisions: Number       = None
    s5_seed_texture_file: str             = None
    s5_leaf_size: List[Number]            = None
    s5_leaf_subdivisions: List[Number]    = None
    s5_number_of_leaves: Number           = None
    s5_mean_leaf_angle: Number            = None
    s5_leaf_texture_file: str             = None


@dataclass(repr = False)
class CameraParameters(Parameters):
    """Stores camera parameters for Helios."""
    image_resolution: List[Number]  = None
    camera_position: List[Number]   = None
    camera_lookat: List[Number]     = None

    def generate_positions(self,
                           camera_type: str,
                           num_views: int,
                           origin: List[Union[int, float]] = None,
                           camera_spacing: int = 2,
                           crop_distance: Union[int, float] = 4,
                           height: int = 1,
                           aerial_parameters: dict = {}):
        """Generates camera positions from the input environment parameters."""
        self.camera_position, self.camera_lookat = generate_camera_positions(
            camera_type = camera_type, num_views = num_views,
            origin = origin, camera_spacing = camera_spacing,
            crop_distance = crop_distance, height = height,
            aerial_parameters = aerial_parameters)


@dataclass(repr = False)
class LiDARParameters(Parameters):
    """Stores LiDAR parameters for Helios."""
    origin: List[Number]    = None
    size: List[Number]      = None
    thetaMin: Number        = None
    thetaMax: Number        = None
    phiMin: Number          = None
    phiMax: Number          = None
    exitDiameter: Number    = None
    beamDivergence: Number  = None
    ASCII_format: str       = None


class HeliosOptions(AgMLSerializable):
    """Stores a set of parameter options for a `HeliosDataGenerator`.

    The primary exposed options are the `canopy_parameters` (as well as similar
    options for camera position and LiDAR, `camera/lidar_parameters' specifically),
    and the actual data generation parameters, such as the `annotation_format`.

    The `HeliosOptions` is instantiated with the name of the canopy type that you
    want to generate; from there, the parameters and ranges can be accessed through
    properties, which are set to their default values as loaded from the Helios
    configuration upon instantiating the class.

    This `HeliosOptions` object, once configured, can then be passed directly to
    a `HeliosDataGenerator` upon instantiation, and be used to generate synthetic
    data according to its specification. The options can be edited and then the
    generation process run again to obtain a new set of data.

    Parameters
    ----------
    canopy : str
        The type of plant canopy to be used in generation.
    """
    serializable = frozenset(('canopy', 'canopy_parameters',
                              'camera_parameters', 'lidar_parameters',
                              'annotation_type', 'simulation_type', 'labels'))
    
    def __new__(cls, *args, **kwargs):
        # The default configuration parameters are loaded directly from
        # the `helios_config.json` file which is constructed each time
        # Helios is installed or updated.
        cls._default_config = load_default_helios_configuration()
        return super(HeliosOptions, cls).__new__(cls)

    @verify_helios
    def __init__(self, canopy = None):
        # Check that the provided canopy is valid.
        self._initialize_canopy(canopy)

        # Initialize the default data generation parameters.
        self._annotation_type = AnnotationType.object_detection
        self._simulation_type = SimulationType.RGB
        self._labels = ['leaves']

    def __str__(self):
        return f"HeliosOptions({self._canopy})({self._to_dict()})"

    def _initialize_canopy(self, canopy):
        """Initializes Helios options from the provided canopy."""
        if canopy not in self._default_config['canopy']['types']:
            raise ValueError(
                f"Received invalid canopy type '{canopy}', expected "
                f"one of: {self._default_config['canopy']['types']}.")
        self._canopy = canopy

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
    
    @property
    def annotation_type(self) -> str:
        return self._annotation_type
    
    @annotation_type.setter
    def annotation_type(self, value: Union[AnnotationType, str]):
        self._annotation_type = AnnotationType(value)

    @property
    def simulation_type(self) -> str:
        return self._simulation_type

    @simulation_type.setter
    def simulation_type(self, value: Union[SimulationType, str]):
        self._simulation_type = SimulationType(value)

    @property
    def labels(self) -> list:
        return self._labels

    @labels.setter
    def labels(self, value: Sequence):
        if len(value) == 0:
            raise ValueError("You cannot have no labels, choose a combination "
                             "of `trunks`, `fruits`, `branches`, and `leaves`.")
        self._labels = value

    def reset(self):
        """Resets the parameters to the default for the canopy."""
        self._initialize_canopy(self._canopy)

    @staticmethod
    def _asdict(obj):
        return {k: v for k, v in asdict(obj).items() if v is not None}

    def _to_dict(self):
        """Returns the options in dictionary format."""
        return {
            'canopy': self._asdict(self._canopy_parameters),
            'camera': self._asdict(self._camera_parameters),
            'lidar': self._asdict(self._lidar_parameters)
        }


