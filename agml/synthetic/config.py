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
import re
import sys
import math
import copy
import json
import subprocess as sp
from datetime import datetime as dt

from agml.utils.io import recursive_dirname

# Paths to the Helios module and the relevant C++ files.
HELIOS_PATH = os.path.join(recursive_dirname(__file__, 2), '_helios/Helios')
CANOPY_SOURCE = os.path.join(
    HELIOS_PATH, 'plugins/canopygenerator/src/CanopyGenerator.cpp')
CANOPY_HEADER = os.path.join(
    HELIOS_PATH, 'plugins/canopygenerator/include/CanopyGenerator.h')
LIDAR_SOURCE = os.path.join(HELIOS_PATH, 'plugins/lidar/src/LiDAR.cpp')

# Path to the stored Helios configuration JSON file in root dir.
HELIOS_CONFIG_FILE = os.path.expanduser('~/.agml/helios_config.json')


# Run the Helios configuration check.
def _check_helios_installation():
    """Checks for the latest Helios installation (and does as such).

    If no existing Helios installation is found, then this runs a fresh
    installation of Helios. If Helios is already installed, then this
    checks if a new possible version exists, and then installs if so.

    Note that this check if only run a maximum of once per day (specifically,
    the check is only run if it has not been run in the last 48 hours). This
    is to prevent constant resource-consuming checks for Git updates.
    """
    # Get the path to the Helios installation file.
    helios_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        '_helios/helios_install.sh')
    helios_dir = os.path.join(
        os.path.dirname(helios_file), 'Helios')

    # Run the installation/update. If the Helios directory is not found,
    # then run a clean installation. Otherwise, check if a new version of
    # Helios is available, then update the existing repository.
    if not os.path.exists(helios_dir):
        sys.stderr.write("Existing installation of Helios not found.\n")
        sys.stderr.write("Installing Helios to:\n")
        sys.stderr.write("\t" + helios_dir + "\n")

        # Update the last check of the Git update.
        with open(os.path.expanduser('~/.agml/config.json')) as f:
            contents = json.load(f)
        with open(os.path.expanduser('~/.agml/config.json'), 'w') as f:
            contents['last_helios_check'] = dt.now().strftime("%B %d %Y %H:%M:%S")
            json.dump(contents, f, indent = 4)

    else:
        # Check if the Git update check has been run in the last 48 hours.
        with open(os.path.expanduser('~/.agml/config.json')) as f:
            contents = json.load(f)
        last_check = contents.get('last_helios_check', dt(1970, 1, 1))
        last_check = dt.strptime(last_check, "%B %d %Y %H:%M:%S")

        # If the last check has been run less than 48 hours ago, then
        # return without having updated Helios.
        if (dt.now() - last_check).days == 0:
            return

        # Check if there is a new version available.
        sys.stderr.write(
            f"Last check for Helios update: over {(dt.now() - last_check).days} "
            f"day(s) ago. Checking for update.")

    # Execute the installation/update script.
    process = sp.Popen(['bash', helios_file],
                       stdout = sp.PIPE, universal_newlines = True)
    for line in iter(process.stdout.readline, ""):
        sys.stderr.write(line)
    process.stdout.close()

    # Update the Helios parameters.
    sys.stderr.write("Updating Helios parameter configuration.")
    _update_helios_parameters()


def _update_helios_parameters():
    """Updates all default Helios parameters (canopy, camera, LiDAR).
    
    Every time a new version of Helios is installed or updated, the default
    parameters need to be read from the Helios source files and updated in
    the local configuration file: `~/.agml/helios_config.json`.
    """
    params = {
        'canopy': _get_canopy_params(),
        'camera': _get_camera_params(),
        'lidar': _get_lidar_params()
    }

    # Save the parameters to the configuration file.
    with open(HELIOS_CONFIG_FILE, 'w') as f:
        json.dump(params, f, indent = 4)


def _get_canopy_params():
    """Updates the default canopy generation parameters for Helios."""
    # Generate the canopy types.
    with open(CANOPY_HEADER, 'r') as f:
        canopy_header_lines = f.readlines()
    canopy_types = [
        re.match('struct\\s(.*?){\\n', string).group(1).split('Parameters')[0]
        for string in canopy_header_lines if 'struct ' in string]

    # Generate the canopy parameters.
    with open(CANOPY_SOURCE) as f:
        canopy_source_lines = f.read().replace('\n', '')
    canopy_parameters = {}

    # This regex is adapted from the following source:
    # https://stackoverflow.com/questions/943391/how-to-get-the-function-declaration-or-definitions-using-regex
    # It searches for all method definitions and then gets all of the content in the method.
    for content in re.finditer(
            r"((?<=[\s:~])(\w+)\s*\(([\w\s,<>\[\].=&':/*]*?)\)"
            r"\s*(const)?\s*(?={)){(.*?)}", canopy_source_lines):
        # Ensure that we're only checking valid parameter methods.
        name = content.group(2).split('Parameters')[0]
        source = content.group(5)

        # Get all of the definitions by removing all spaces of at least length 2.
        if any(i == name for i in canopy_types) and 'std::cout' not in source:  # noqa
            canopy_parameters[name] = {}
            definitions = [
                s.replace(';', '') for s in re.split('[\\s]{2,}', source) if s != '']
            for d in definitions:
                # Split the name and the value.
                n, value = d.split(' = ')

                # If the value is defined in a method, extract the method arguments.
                if '(' in value:
                    value = re.search('\\((.*?)\\)', value).group(1)

                # Remove unnecessary terms. The `.f` removal needs to be done using
                # regex since we can't just remove regular `f` in case of paths,
                # but we also can't just replace `.f` with `0` or `.0`.
                value = value.replace(',', ' ').replace('\"', '')
                value = re.sub('(\\.|\\d)f', r'\g<1>0', value)

                # A specific case to replace values using pi.
                if 'M_PI' in value:
                    value = str(round(float(value.split('*')[0].strip()) * math.pi, 6))

                # Update the parameter dictionary.
                canopy_parameters[name][n] = value

    # Generate the initial canopy parameter ranges.
    canopy_parameter_ranges = copy.deepcopy(canopy_parameters)
    for c in canopy_parameter_ranges.keys():
        for key in list(canopy_parameter_ranges[c]):
            val = canopy_parameter_ranges[c][key]
            if val.isalpha() or '/' in val or ':' in val:
                canopy_parameter_ranges[c].pop(key)
            else:
                canopy_parameter_ranges[c][key] = [
                    [float(val.split(' ')[j]),
                     float(val.split(' ')[j])] for j
                    in range(len(val.split()))]

    # Return the three dictionaries.
    return {'types': canopy_types, 'parameters': canopy_parameters,
            'ranges': canopy_parameter_ranges}


def _get_camera_params():
    """Updates the default camera parameters for Helios."""
    # No files are read here, the parameters are hard-coded.
    camera_params = {
        'image_resolution': '600 400',
        'camera_lookat': '0 0 1',
        'camera_position': '0 -2 1'}
    camera_param_ranges = {
        i: [float(camera_params[i].split(' ')[j])
            for j in range(len(camera_params[i].split()))]
        for i in camera_params.keys()}
    return {'parameters': camera_params, 'ranges': camera_param_ranges}


def _get_lidar_params():
    """Updates the default LiDAR parameters for Helios."""
    # No files are read here, the parameters are hard-coded.
    lidar_params = {
        "origin": "0 0 0",
        "size": "250 450",
        "thetaMin": "0",
        "thetaMax": "180",
        "phiMin": "0",
        "phiMax": "360",
        "exitDiameter": "0",
        "beamDivergence": "0",
        "ASCII_format": "x y z"}
    lidar_param_ranges = {
        i: [float(item.split(' ')[j].replace('f', '0')) if i != 'ASCII_format' else
            (item.split(' ')[j].replace('f', '0')) for j in range(len(item.split()))]
        for i, item in lidar_params.items()}
    return {'params': lidar_params, 'ranges': lidar_param_ranges}



