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

from agml.backend.config import _update_config, _get_config
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


def _update_cmake(lidar_enabled):
    """Updates the CMake file to enable or disable LiDAR compilation."""
    cmake_file = os.path.join(PROJECT_PATH, 'CMakeLists.txt')
    with open(cmake_file, 'r') as f:
        default_cmake_contents = f.read()
    if lidar_enabled:
        if 'lidar' not in default_cmake_contents: # add LiDAR to plugins
            cmake_contents = default_cmake_contents.replace(
                'set( PLUGINS "visualizer;canopygenerator;syntheticannotation" )',
                'set( PLUGINS "lidar;visualizer;canopygenerator;syntheticannotation" )'
            )
        else:
            cmake_contents = default_cmake_contents
    else:
        cmake_contents = re.sub('lidar;', '', default_cmake_contents)
    with open(cmake_file, 'w') as f:
        f.write(cmake_contents)


def _compile_helios_default(cmake_build_type = 'Release',
                            lidar_enabled = False,
                            parallel = True):
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

    try:
        # Update the CMake file with or without LiDAR compilation enabled.
        _update_cmake(lidar_enabled = lidar_enabled)

        # Construct arguments for the compilation.
        cmake_args = ['cmake', '..', '-G', 'Unix Makefiles',
                      f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={HELIOS_BUILD}',
                      f'-DCMAKE_BUILD_TYPE={cmake_build_type}']
        if not parallel or sys.platform == 'win32':
            make_args = ['cmake', '--build', '.']
        else:
            make_args = ['make', f'-j{os.cpu_count()}']

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
        _compilation_successful(lidar_enabled = lidar_enabled)

    # Even account for KeyboardInterrupt when compiling.
    except BaseException:
        _compilation_failed(helios_temp_dir, temp_dir)


def _compile_helios_executable_only(parallel):
    """Compiles only the Helios executable."""
    if not os.path.exists(PROJECT_PATH):
        raise NotADirectoryError(
            f"The expected project path {PROJECT_PATH} does not exist. "
            f"There may have been an error in installing Helios. Try "
            f"re-installing it, or raise an issue with the AgML team.")

    # Construct arguments for the compilation.
    if not parallel or sys.platform == 'win32':
        make_args = ['cmake', '--build', '.']
    else:
        make_args = ['make', f'-j{os.cpu_count()}']

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
    _compilation_successful(lidar_enabled = None)


def recompile_helios(executable_only = False,
                     debug_mode = False,
                     lidar_enabled = False,
                     parallel = True):
    """Recompiles the Helios library with the set parameters.

    This method can be used by the user in order to recompile Helios, if, for
    instance, an error was encountered during prior compilation, or a local edit
    is made, without having to wait for the default recompilation every 48 hours.

    If just the `generate.cpp` file itself is edited, then you can pass the
    parameter `executable_only` as `True`, which will result in a much quicker
    compilation as just the generation file itself is edited.

    Parameters
    ----------
    executable_only : bool
        Whether to compile only the `generate.cpp` executable.
    debug_mode : bool
        Whether to compile the entirety of Helios in Debug mode (defaults to
        Release). Debug mode is more verbose, but Release mode is faster.
    lidar_enabled : bool
        Whether to compile Helios with LiDAR support enabled.
    parallel : bool
        Whether to compile Helios in parallel. This is not enabled for Windows
        users, as the `make` command does not exist on Windows. This leads to
        significantly faster compilation, at the cost of high CPU usage.
    """
    if executable_only and lidar_enabled:
        raise ValueError("If you want to compile Helios with LiDAR support, "
                         "you cannot compile only the executable.")
    if executable_only and debug_mode:
        raise ValueError("If you want to compile the build in `Debug` mode, "
                         "you cannot compile only the executable.")
    if executable_only:
        _compile_helios_executable_only(parallel = parallel)
    else:
        cmake_build_type = 'Debug' if debug_mode else 'Release'
        _compile_helios_default(cmake_build_type = cmake_build_type,
                                lidar_enabled = lidar_enabled,
                                parallel = parallel)


def _compilation_successful(lidar_enabled):
    """Updates after successful Helios compilation."""
    sys.stdout.write('\n')
    sys.stderr.write('\nHelios compilation successful!\n')
    if lidar_enabled is not None:
        _update_config('lidar_enabled', lidar_enabled)


def is_helios_compiled_with_lidar():
    """Returns whether Helios was compiled with LiDAR support enabled."""
    try:
        return _get_config('lidar_enabled')
    except:
        _update_config('lidar_enabled', False)
        return False


def _compilation_failed(helios_temp_dir, temp_dir):
    """Cleanup after a failed Helios compilation."""
    shutil.rmtree(HELIOS_BUILD)
    shutil.move(helios_temp_dir, HELIOS_BUILD)
    temp_dir.cleanup()
    sys.exit(1)




