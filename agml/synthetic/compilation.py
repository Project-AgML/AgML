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
import glob
import shutil
import tempfile
import subprocess as sp
from datetime import datetime as dt

from agml.synthetic.config import HELIOS_PATH


# Helios build and compilation paths.
PROJECT_PATH = os.path.join(
    HELIOS_PATH, 'projects/SyntheticImageAnnotation')
HELIOS_BUILD = os.path.join(
    PROJECT_PATH, 'build')
HELIOS_EXECUTABLE = os.path.join(
    PROJECT_PATH, 'build', 'SyntheticImageAnnotation')

# Helios parameter paths.
XML_PATH = os.path.join(PROJECT_PATH, 'xml')


def _compile_helios_default():
    """Compiles the default Helios library upon installation and update."""
    if not os.path.exists(PROJECT_PATH):
        raise NotADirectoryError(
            f"The expected project path {PROJECT_PATH} does not exist. "
            f"There may have been an error in installing Helios. Try "
            f"re-installing it, or raise an issue with the AgML team.")

    # If there is an existing compiled version of Helios, then cache it
    # temporarily until the new compilation is complete. If an error is
    # encountered during the compilation, then we want to have a backup.
    os.makedirs(HELIOS_BUILD, exist_ok = True)
    temp_dir = tempfile.TemporaryDirectory()
    helios_temp_dir = os.path.join(temp_dir.name, 'helios_build')
    shutil.move(HELIOS_BUILD, helios_temp_dir)

    # Check that LiDAR compilation is disabled by default. We need to
    # store the original pulled contents of the CMake file, however, so
    # that we can revert it once it's been edited for the purposes of
    # compilation. This is so that new versions of Helios can be pulled
    # from Git without any clashes due to the CMake file having been edited.
    cmake_file = os.path.join(PROJECT_PATH, 'CMakeLists.txt')
    with open(cmake_file, 'r') as f:
        default_cmake_contents = f.read()
    cmake_contents = re.sub('lidar;', '', default_cmake_contents)
    with open(cmake_file, 'w') as f:
        f.write(cmake_contents)

    # Construct arguments for the compilation.
    cmake_args = ['cmake', '..', '-G', 'Unix Makefiles',
                  f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={HELIOS_BUILD}',
                  '-DCMAKE_BUILD_TYPE=Release']
    make_args = ['cmake', '--build', '.']

    # Create the log file and clear the existing one.
    log_file = os.path.expanduser(
        f"~/.agml/.helios_compilation_log"
        f"-{dt.now().strftime('%Y%m%d-%H%M%S')}.log")
    try:
        existing_log = glob.glob(os.path.expanduser(
            '~/.agml/.helios_compilation_log-*.log'))[0]
        os.remove(existing_log)
    except:
        pass

    # Create the build directory again, since it has been moved.
    os.makedirs(HELIOS_BUILD, exist_ok = True)

    # Compile the CMake files.
    cmake_log = ""
    sys.stderr.write("Compiling Helios with CMake.\n\n")
    cmake_process = sp.Popen(cmake_args, stdout = sp.PIPE, cwd = HELIOS_BUILD,
                             stderr = sp.STDOUT, universal_newlines = True)
    for line in iter(cmake_process.stdout.readline, ""):
        cmake_log += line
        sys.stderr.write(line)
    cmake_process.stdout.close()
    with open(log_file, 'a') as f:
        f.write(cmake_log)
    cmake_return = cmake_process.wait()
    if cmake_return != 0:
        sys.stderr.write(
            f'\nEncountered an error when attempting to compile '
            f'Helios with CMake. Please report this traceback to '
            f'the AgML team. A full traceback of the compilation '
            f'process can be found at "{log_file}".')
        _compilation_failed(helios_temp_dir, temp_dir)
    sys.stdout.write('\n')
    sys.stderr.write('\n')

    # Generate the main executable.
    cmake_log = "\n"
    sys.stderr.write("Building Helios executable with CMake.\n\n")
    make_process = sp.Popen(make_args, stdout = sp.PIPE, cwd = HELIOS_BUILD,
                            stderr = sp.STDOUT, universal_newlines = True)
    for line in iter(make_process.stdout.readline, ""):
        cmake_log += line
        sys.stderr.write(line)
    make_process.stdout.close()
    with open(log_file, 'a') as f:
        f.write(cmake_log)
    make_return = make_process.wait()
    if make_return != 0:
        sys.stderr.write(
            f'\nEncountered an error when attempting to compile '
            f'Helios with CMake. Please report this traceback to '
            f'the AgML team. A full traceback of the compilation '
            f'process can be found at "{log_file}".')
        _compilation_failed(helios_temp_dir, temp_dir)

    # Print the final message.
    sys.stdout.write('\n')
    sys.stderr.write('\nHelios compilation successful!')


def _compile_helios_executable_only():
    """Compiles only the Helios executable."""
    if not os.path.exists(PROJECT_PATH):
        raise NotADirectoryError(
            f"The expected project path {PROJECT_PATH} does not exist. "
            f"There may have been an error in installing Helios. Try "
            f"re-installing it, or raise an issue with the AgML team.")

    # Construct arguments for the compilation.
    make_args = ['cmake', '--build', '.']

    # Create the log file and clear the existing one.
    log_file = os.path.expanduser(
        f"~/.agml/.helios_compilation_log"
        f"-{dt.now().strftime('%Y%m%d-%H%M%S')}.log")
    try:
        existing_log = glob.glob(os.path.expanduser(
            '~/.agml/.helios_compilation_log-*.log'))[0]
        os.remove(existing_log)
    except:
        pass

    # Generate the main executable.
    cmake_log = "\n"
    sys.stderr.write("Building Helios executable with CMake.\n\n")
    make_process = sp.Popen(make_args, stdout = sp.PIPE, cwd = HELIOS_BUILD,
                            stderr = sp.STDOUT, universal_newlines = True)
    for line in iter(make_process.stdout.readline, ""):
        cmake_log += line
        sys.stderr.write(line)
    make_process.stdout.close()
    with open(log_file, 'a') as f:
        f.write(cmake_log)
    make_return = make_process.wait()
    if make_return != 0:
        sys.stderr.write(
            f'\nEncountered an error when attempting to compile '
            f'Helios with CMake. Please report this traceback to '
            f'the AgML team. A full traceback of the compilation '
            f'process can be found at "{log_file}".')

    # Print the final message.
    sys.stdout.write('\n')
    sys.stderr.write('\nHelios compilation successful!')


def recompile_helios(executable_only = False):
    """Recompiles the Helios library with the set parameters.

    This method can be used by the user in order to recompile Helios, if, for
    instance, an error was encountered during prior compilation, or a local edit
    is made, without having to wait for the default recompilation every 48 hours.

    If just the `generate.cpp` file itself is edited, then you can pass the
    parameter `executable_only` as `True`, which will result in a much quicker
    compilation as just the generation file itself is edited.
    """
    if executable_only:
        _compile_helios_executable_only()
    else:
        _compile_helios_default()


def _compilation_failed(helios_temp_dir, temp_dir):
    """Cleanup after a failed Helios compilation."""
    shutil.rmtree(HELIOS_BUILD)
    shutil.move(helios_temp_dir, HELIOS_BUILD)
    temp_dir.cleanup()
    sys.exit(1)




