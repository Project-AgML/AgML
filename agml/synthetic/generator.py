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
import io
import sys
import json
import datetime
import subprocess as sp
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from typing import List, Union

from dict2xml import dict2xml

from agml.framework import AgMLSerializable
from agml.backend.config import synthetic_data_save_path
from agml.synthetic.options import HeliosOptions
from agml.synthetic.options import AnnotationType, SimulationType
from agml.synthetic.config import load_default_helios_configuration, verify_helios
from agml.synthetic.compilation import (
    HELIOS_BUILD, HELIOS_EXECUTABLE, XML_PATH, PROJECT_PATH
)
from agml.synthetic.converter import HeliosDataFormatConverter


@dataclass
class GenerationInstanceOptions:
    canopy: str
    num_images: int
    annotation_type: Union[AnnotationType, str]
    simulation_type: Union[SimulationType, str]
    labels: List[str]
    output_dir: str

    def __post_init__(self):
        if self.num_images < 1:
            raise ValueError("The number of images cannot be negative.")

        try:
            self.annotation_type = AnnotationType(self.annotation_type)
        except ValueError:
            raise ValueError(
                "Expected either an `AnnotationType` parameter "
                "or corresponding string for 'annotation_type'. "
                "Valid annotation types: `object_detection`, "
                "`semantic_segmentation`, `instance_segmentation`, "
                "or for no annotations at all, `none`.")

        try:
            self.simulation_type = SimulationType(self.simulation_type)
        except ValueError:
            raise ValueError(
                "Expected either a `SimulationType` parameter "
                "or corresponding string for 'simulation_type'. "
                "Valid simulation types: rgb`, `lidar`.")

        valid_labels = ['trunks', 'leaves', 'fruits', 'branches']
        if not all(i in valid_labels for i in self.labels):
            raise ValueError(f"Got one or more invalid labels: {self.labels}."
                             f"Valid labels: {[*valid_labels]}.")
        if self.canopy in ['Tomato', 'Strawberry'] and 'trunks' in self.labels:
            raise ValueError("Tomato and Strawberry canopies do not have any trunks, "
                             "so trying to use the `trunks` label is impossible.")

    @classmethod
    def _empty(cls):
        return cls.__new__(cls)


class HeliosDataGenerator(AgMLSerializable):
    """Generates a synthetic agricultural dataset using Helios.

    The `HeliosDataGenerator` provides a Python interface to the Helios C++
    data generation software (https://github.com/PlantSimulationLab/Helios),
    used for generating synthetic agricultural data and annotations. It can
    be used to generate a wide variety of synthetic datasets, with vast
    customizability through the  available options and parameters, including
    both RGB and LiDAR simulations, annotations from object detection to
    semantic segmentation, and a variety of different crops.

    Parameters
    ----------
    options : HeliosOptions
        A `HeliosOptions` instance containing pre-initialized parameters.
        If you want to directly use the pre-set parameters, then you can
        also use the keyword argument `canopy` to bypass the options.
    canopy : str
        A specific canopy to generate images for. This can be passed for
        in place of `options` to initialize the default options.
    """
    @verify_helios
    def __init__(self, options: HeliosOptions = None, *, canopy = None):
        if options is None and canopy is not None:
            self._options = HeliosOptions(canopy)
        elif options is not None and canopy is None:
            self._options = options
        else:
            raise ValueError(
                "The `HeliosDataGenerator` needs either "
                "a set of `HeliosOptions` or a `canopy`.")
        self._canopy = self._options._canopy

        # Rather than constantly passing the different parameters used during
        # generation, such as the annotation type or the output directory,
        # throughout each method, whenever data generation begins, this class
        # is instantiated with all of the parameters for convenience. This
        # also enables wrapping all of the checks into a single method which is
        # contained within the external class as opposed to this class.
        self._generation_options = GenerationInstanceOptions._empty()

    @property
    def options(self) -> HeliosOptions:
        return self._options

    @options.setter
    def options(self, value: HeliosOptions):
        self._canopy = value._canopy
        self._options = value

    @staticmethod
    def _sanitize_name(name):
        """Cleans a name to represent a valid folder destination."""
        return name.replace(' ', '-').replace('~', '-')

    def _convert_options_to_xml(self):
        """Converts the `HeliosOptions` parameters to XML format."""
        # Convert all of the parameters to XML format.
        tree = ET.parse(io.StringIO(dict2xml(
            {'helios': self._prepare_parameters_for_generation()})))

        # Update the camera tags.
        root = tree.getroot()
        for child in root:
            if child.tag == 'camera_position':
                child.tag = 'globaldata_vec3'
                child.set('label', 'camera_position')
            if child.tag == 'camera_lookat':
                child.tag = 'globaldata_vec3'
                child.set('label', 'camera_lookat')
            if child.tag == 'image_resolution':
                child.tag = 'globaldata_int2'
                child.set('label', 'image_resolution')

        # Return the modified tree.
        return tree

    def _prepare_parameters_for_generation(self):
        """Prepares the provided Helios parameter options for generation."""
        parameters = self._options._to_dict()

        # Divide the image resolution in half. For some reason, Helios
        # appears to generate images with twice the provided resolution,
        # so in order to fix this, we need to divide the resolution in
        # two here so that it is correct during generation.
        parameters['camera']['image_resolution'] = \
            [i // 2 for i in parameters['camera']['image_resolution']]

        # Generate the ground parameters.
        ground_params = self._load_ground_parameters()
        parameters['Ground'] = ground_params

        # Convert the canopy parameters to the Helios string format.
        self._convert_dict_params_to_string(parameters)

        # Create a fresh dictionary and put all of the parameters in it.
        canopy_parameters = {
            self._canopy + "Parameters": parameters['canopy'],
            'Ground': parameters['Ground']}
        xml_params = {'canopygenerator': canopy_parameters}
        if self._generation_options.simulation_type == 'lidar':
            xml_params['scan'] = parameters['lidar']
        else:
            xml_params[''] = parameters['camera']

        # Return the parsed dictionary.
        return xml_params

    @staticmethod
    def _convert_dict_params_to_string(d):
        """Converts dictionary int/float parameters to strings."""
        for key, value in d.items():
            for param, param_value in value.items():
                if isinstance(param_value, Sequence) \
                        and not isinstance(param_value, str):
                    if isinstance(param_value[0], Sequence):
                        param_value = [str(a).replace(",", "").replace(
                            "[", " ").replace("]", "") for a in param_value]
                        value[param] = \
                            f' {" ".join(param_value)} '
                    else:
                        value[param] = \
                            f' {" ".join([str(a) for a in param_value])} '
                else:
                    value[param] = f' {str(param_value)} '

    @staticmethod
    def _load_ground_parameters():
        """Loads the parameters for the `Ground` plant type."""
        cfg = load_default_helios_configuration()
        return cfg['canopy']['parameters']['Ground']

    @staticmethod
    def _parse_output_path(output_path: str):
        output_path = os.path.abspath(output_path)
        if os.path.exists(output_path) and os.path.isdir(output_path):
            if any(os.scandir(output_path)):
                raise OSError(f"The provided directory for generation, "
                              f"'{output_path}', is not empty. Please "
                              f"clear out the contents of this directory "
                              f"and try again.")
        else:
            os.makedirs(output_path)

    def _write_config(self, *, cfg_file, xml_file):
        """Writes the config file that Helios loads from."""
        # Construct the string with the output.
        xml_file = os.path.realpath(xml_file)
        cfg = f"{self._generation_options.num_images}\n" \
              f"{self._generation_options.annotation_type.value}\n" \
              f"{self._generation_options.simulation_type.value}\n" \
              f"{' '.join(self._generation_options.labels)}\n" \
              f"{xml_file}\n{self._generation_options.output_dir}"

        # Write the configuration file to its output path.
        with open(os.path.join(PROJECT_PATH, cfg_file), 'w') as f:
            f.write(cfg)

    def generate(self, *,
                 name: str = None,
                 num_images: int = 5,
                 output_dir: os.PathLike = None,
                 convert_data: bool = True):
        """Generates a batch of synthetic data using Helios.

        This method is the one which conducts the actual data generation using
        Helios. By this point, the `HeliosDataGenerator` should have an initialized
        `HeliosOptions` class which contains all of the scene and canopy generation
        parameters, set to the user preference, and this method takes in the name
        of the run itself as well as the amount of images to generate, the final
        parameters which are used to generate the actual data.

        The `name` parameter here refers to an optional name which can be assigned
        to customize the dataset that is being generated (e.g., if it is a dataset
        of 100 strawberry plants, then it could be 'strawberry100'). If no specific
        name is provided, then the generator defaults to using today's date and time
        (e.g., a run at 12:15 PM on January 12, 2024 would be 'helios-01122024-1215').

        If multiple camera views are provided as part of the `HeliosOptions`, then the
        number of images will actually end up being `num_images` times the number of
        camera views, e.g., if there are 5 camera views and 50 images to be generated,
        then there will be 250 total images in the resulting dataset. The final output
        directory structure ends up being something like this:

        generated_images
            image0
                view00000
                view00001
                ...
            image1
                view00000
                view00001
                ...
            ...

        NOte that this is only the post-Helios output structure. By default, after
        the dataset is generated, the annotations will be converted from the Helios
        format to the relevant format within AgML. For example, annotation text files
        will be converted to COCO JSON, and the directory structure of the dataset
        will also be changed accordingly. If you want to maintain the integrity of
        the generated data for Helios, then you can provide the optional parameter
        `convert_data = False`, which will leave data as generated by Helios.

        Regardless of whether annotations are converted to the AgML format from the
        Helios format, the output directory will contain an extra hidden directory
        `.metadata`, which will contain a few extra files:

            1. The XML file which contains the parameters in the `HeliosOptions`.
            2. The `config.txt` file which is used for generation.
            3. An extra `meta.json` file which contains some additional information,
               such as the classes in the dataset, and the annotation type.

        These are used by the `AgMLDataLoader` when loading a custom Helios-generated
        dataset, and can also be used to recreate a dataset from its original parameters.

        Parameters
        ----------
        name : str
            The name of the run (and thus the generated dataset).
        num_images : int
            The number of images to generate in the dataset.
        output_dir : str
            A custom output directory where data will be generated to. If it is
            a path such as `/root/path`, then the data will be generated to the
            path `/root/path/name`. If `/root/path/name` is passed, then it will
            be saved directly to there (not `/root/path/name/name`).
        convert_data : {bool, optional}
            Whether to convert data to the AgML format after generation. If left to
            False, then data is left in the corresponding Helios format.

        Notes
        -----
        Unless a custom path is passed to this method, all datasets will be saved to
        the default *synthetic* data generation path. Note that this is distinct from
        `agml.backend.data_save_path()`. This defaults to `~/.agml/synthetic`, but can
        be changed using `agml.backend.set_synthetic_save_path()`, and then accessed
        using `agml.backend.synthetic_data_save_path()`. The distinction is important.
        """
        # Construct and sanitize the name of the run.
        if name is not None:
            name = self._sanitize_name(name)
        else:
            date = datetime.datetime.now().strftime('%m%d%Y-%H%M')
            name = f'helios-{date}'

        # Construct the output path of the dataset.
        if output_dir is not None:
            if os.path.basename(output_dir) == name:
                output_dir = os.path.dirname(output_dir)
        else:
            output_dir = synthetic_data_save_path()
        output_dir = os.path.join(output_dir, name)
        if os.path.exists(output_dir):
            self._parse_output_path(output_dir)
        else:
            os.makedirs(output_dir)

        # Construct the `GenerationInstanceOptions` class from the `HeliosOptions`
        # parameters and the newly passed input parameters.
        self._generation_options = GenerationInstanceOptions(
            canopy = self._canopy,
            num_images = num_images,
            annotation_type = self._options.annotation_type,
            simulation_type = self._options.simulation_type,
            labels = self._options.labels,
            output_dir = output_dir)

        # Create the output metadata directory.
        metadata_dir = os.path.join(output_dir, '.metadata')
        os.makedirs(metadata_dir)

        # Construct the XML parameter style file.
        xml_options = self._convert_options_to_xml()
        xml_file_base = f"style_{name}.xml"
        xml_options.write(os.path.join(XML_PATH, xml_file_base))
        xml_options.write(os.path.join(metadata_dir, xml_file_base))

        # Write the actual configuration file.
        cfg_file = os.path.join(PROJECT_PATH, f'config_{name}.txt')
        self._write_config(
            cfg_file = cfg_file, xml_file = os.path.join(XML_PATH, xml_file_base))
        self._write_config(
            cfg_file = os.path.join(metadata_dir, f'config_{name}.txt'),
            xml_file = os.path.join(metadata_dir, xml_file_base))

        # Run the actual data generation with the executable.
        process = sp.Popen([HELIOS_EXECUTABLE, cfg_file], stdout = sp.PIPE,
                           stderr = sp.STDOUT, cwd = HELIOS_BUILD,
                           universal_newlines = True,)
        for line in iter(process.stdout.readline, ""):
            sys.stdout.write(line)
        process.stdout.close()
        process.wait()

        # Remove the configuration and style files.
        os.remove(cfg_file)
        os.remove(os.path.join(XML_PATH, xml_file_base))

        # Save the extra metadata.
        with open(os.path.join(metadata_dir, 'meta.json'), 'w') as f:
            json.dump({'labels': self._generation_options.labels,
                       'image_size': self._options.camera.image_resolution,
                       'generation_date': datetime.datetime.now().strftime(
                           '%B %d, %Y %H:%M')}, f, indent = 4)

        # Check that the process successfully completed.
        if process.returncode != 0:
            if process.returncode == -11:
                raise OSError(f"Encountered an error when generating synthetic "
                              f"data. Process returned code {process.returncode}, "
                              f"suggesting that the program ran out of memory. Try "
                              f"passing a smaller environment for generation.")
            else:
                raise OSError(f"Encountered an error when generating synthetic "
                              f"data. Process returned code {process.returncode}.")

        # Convert the dataset format.
        if convert_data:
            cvt = HeliosDataFormatConverter(output_dir)
            cvt.convert()

