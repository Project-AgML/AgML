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
import subprocess as sp

from agml.backend.config import SUPER_BASE_DIR
from agml.synthetic.config import HELIOS_PATH
from agml.utils.io import recursive_dirname, load_code_from_string_or_file
from agml.utils.logging import log


def generate_manual_data(code = None,
                         cmake = None,
                         xml = None,
                         project_name = None,
                         project_path = None,
                         files_already_placed = False,
                         overwrite_existing_files = False,
                         force_recompile = False,
                         cmake_build_type = 'Debug'):
    """Manually generates data using Helios, given a generation script.

    This method can be used to generate data using Helios with more controls
    over not only the parameters, but the canopy generation/annotation
    creation in the actual generation itself.

    You will need to write your own C++ script which interfaces with Helios
    for this method to work; this is simply an access point which auto-compiles
    Helios and runs the generation.

    The code will be generated in a directory `Helios/projects/<project_dir>`,
    and you can either provide the project name to `project_name`, or the directory
    to the project (if you're using a different installation of Helios) in the
    argument `project_dir`. In either case, you can skip the automatic file
    placement and do it manually and then pass the `files_already_placed` argument.

    Parameters
    ----------
    code : Any
        Either the path to a C++ script, or a string of C++ code interfacing
        with Helios to run the generation.
    cmake : Any
        The CMakeLists.txt file for compiling Helios. If not provided, this will
        be automatically generated using the base template.
    xml : Any
        A potential XML stylesheet for generation (on a need-to-use basis).
    project_name : Any
        The name of the project that you want to generate (using the internal
        Helios installation).
    project_path : Any
        The path to the Helios installation + project that you want to generate
        (see above for the format).
    files_already_placed : bool
        If all of the files have been already placed, then no checks or construction
        needs to be done, only compilation and execution.
    overwrite_existing_files : bool
        Whether to overwrite existing files during generation.
    force_recompile : bool
        Whether to force recompilation each time.
    cmake_build_type : bool
        Whether to compile Helios in debug or release mode.
    """
    if not files_already_placed:
        project_path, project_name, cmake = process_and_move_files(
            code = code,
            cmake = cmake,
            xml = xml,
            project_name = project_name,
            project_path = project_path,
            overwrite_existing_files = overwrite_existing_files)
    else:
        # Still run the project name/path check.
        if project_name is None and project_path is None:
            raise ValueError("You need to provide either the project name or project path.")
        if project_name is not None and project_path is not None:
            raise ValueError("Do not provide both the project name and path, just use one.")

        if project_name is not None and project_path is None:
            project_path = os.path.join(HELIOS_PATH, 'projects', project_name)
        if project_path is not None and project_name is None:
            project_name = os.path.basename(project_path)
            if not os.path.exists(project_path):
                if not os.path.exists(recursive_dirname(project_path, 2)):
                    raise NotADirectoryError(
                        f"The parent Helios directory "
                        f"{recursive_dirname(project_path, 2)} "
                        f"could not be found. Check that Helios is installed there.")

        # Check that the necessary files exist.
        if not os.path.exists(os.path.join(project_path, 'main.cpp')):
            raise ValueError(f"Could not find `main.cpp` file at {project_path}.")
        if not os.path.exists(os.path.join(project_path, 'CMakeLists.txt')):
            raise ValueError(f"Could not find `CMakeLists.txt` file at {project_path}.")

    # Check whether Helios needs to be compiled (if any changes have been made).
    cpp_same, cmake_same = False, False
    if os.path.exists(os.path.join(
            SUPER_BASE_DIR, '.last_manual_compilation_cpp.cpp')) and not force_recompile:
        # Check if the C++ file has changed.
        with open(os.path.join(SUPER_BASE_DIR, '.last_manual_compilation_cpp.cpp'), 'r') as f:
            legacy_cpp = f.read()
        with open(os.path.join(project_path, 'main.cpp'), 'r') as f:
            current_cpp = f.read()
        if legacy_cpp == current_cpp:
            cpp_same = True

        # Check if the CMake file has changed.
        with open(os.path.join(SUPER_BASE_DIR, '.last_manual_compilation_cmake.txt'), 'r') as f:
            legacy_cmake = f.read()
        with open(os.path.join(project_path, 'CMakeLists.txt'), 'r') as f:
            current_cmake = f.read()
        if legacy_cmake == current_cmake:
            cmake_same = True
    
    # Compile Helios.
    if not cpp_same and not cmake_same:
        # Construct arguments for the compilation.
        helios_build = os.path.join(project_path, 'build')
        cmake_args = ['cmake', '..', '-G', 'Unix Makefiles',
                      f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={helios_build}',
                      f'-DCMAKE_BUILD_TYPE={cmake_build_type}']
        make_args = ['cmake', '--build', '.']
        log_file = os.path.join(SUPER_BASE_DIR, '.last_helios_manual_compilation.log')
        os.unlink(log_file)

        # Create the build directory again, since it has been moved.
        os.makedirs(helios_build, exist_ok = True)

        # Compile the CMake files.
        cmake_log = ""
        sys.stderr.write("Compiling Helios with CMake.\n\n")
        cmake_process = sp.Popen(cmake_args, stdout = sp.PIPE, cwd = helios_build,
                                 stderr = sp.STDOUT, universal_newlines = True)
        for line in iter(cmake_process.stdout.readline, ""):
            cmake_log += line
            sys.stderr.write(line)
        cmake_process.stdout.close()
        with open(log_file, 'a') as f:
            f.write(cmake_log)
        cmake_return = cmake_process.wait()
        if cmake_return != 0:
            raise ValueError(
                f'\nEncountered an error when attempting to compile '
                f'Helios with CMake. Please report this traceback to '
                f'the AgML team. A full traceback of the compilation '
                f'process can be found at "{log_file}".')
        sys.stdout.write('\n')
        sys.stderr.write('\n')

        # Generate the main executable.
        cmake_log = "\n"
        sys.stderr.write("Building Helios executable with CMake.\n\n")
        make_process = sp.Popen(make_args, stdout = sp.PIPE, cwd = helios_build,
                                stderr = sp.STDOUT, universal_newlines = True)
        for line in iter(make_process.stdout.readline, ""):
            cmake_log += line
            sys.stderr.write(line)
        make_process.stdout.close()
        with open(log_file, 'a') as f:
            f.write(cmake_log)
        make_return = make_process.wait()
        if make_return != 0:
            raise ValueError(
                f'\nEncountered an error when attempting to compile '
                f'Helios with CMake. Please report this traceback to '
                f'the AgML team. A full traceback of the compilation '
                f'process can be found at "{log_file}".')

        # Print the final message.
        sys.stdout.write('\n')
        sys.stderr.write('\nHelios compilation successful!')

    # Run the generation.
    executable = os.path.join(project_path, 'build', project_name)
    process = sp.Popen([executable], stdout = sp.PIPE, stderr = sp.STDOUT,
                       cwd = os.path.join(project_path, 'build'),
                       universal_newlines = True, )
    for line in iter(process.stdout.readline, ""):
        sys.stdout.write(line)
    process.stdout.close()
    process.wait()

    # Save the existing files for comparison in future runs.
    with open(os.path.join(SUPER_BASE_DIR, '.last_manual_compilation_cpp.cpp'), 'w') as f:
        f.write(code)
    with open(os.path.join(SUPER_BASE_DIR, '.last_manual_compilation_cmake.txt'), 'w') as f:
        f.write(cmake)


def process_and_move_files(code = None,
                           cmake = None,
                           xml = None,
                           project_name = None,
                           project_path = None,
                           overwrite_existing_files = False):
    """Processes and moves in files if provided in the manual generation method."""
    code = load_code_from_string_or_file(code)

    # Check if the code contains a reference to XML, and load the XML
    # stylesheet appropriately.
    xml_fname = None
    if 'loadXML' in code:
        if xml is None:
            log('Detected a reference to `loadXML` in the provided C++ '
                'generation file, but no XML stylesheet was provided for '
                'manual generation. Generation may fail, unless the '
                'stylesheet is already in the appropriate directory.')
        else:
            xml = load_code_from_string_or_file(xml)

            # Infer the name of the XML file.
            fname = re.search('loadXML\\((.*?)\\)', xml).group(1)
            if '"' in fname:
                fname = fname[1:-1]
            xml_fname = fname

    # Create the project directory.
    if not project_name and not project_path:
        raise ValueError("You need to provide either the project name or project path.")
    if project_name and project_path:
        raise ValueError("Do not provide both the project name and path, just use one.")

    if project_name and not project_path:
        project_path = os.path.join(HELIOS_PATH, 'projects', project_name)
    if project_path and not project_name:
        if not os.path.exists(project_path):
            if not os.path.exists(recursive_dirname(project_path, 2)):
                raise NotADirectoryError(
                    f"The parent Helios directory "
                    f"{recursive_dirname(project_path, 2)} "
                    f"could not be found. Check that Helios is installed there.")

    os.makedirs(project_path, exist_ok = True)
    if any(os.scandir(project_path)):
        if overwrite_existing_files:
            log(f"Existing files found at ({project_path}). Overwriting them.")
            os.makedirs(project_path, exist_ok = True)
        else:
            raise ValueError(
                f"Existing files found at ({project_path}). If you want to "
                f"overwrite them, use the `overwrite_existing_files` argument.")

    # Create the CMake file.
    if cmake is None:
        cmake_default_path = os.path.join(
            os.path.dirname(__file__),
            'synthetic_data_generation', 'CMakeLists.txt')
        with open(cmake_default_path, 'r') as f:
            cmake_default = f.read()
        if 'lidar' in code:
            log("Detected a reference to LiDAR annotations in the provided"
                "C++ generation file. Since no CMake file has been provided,"
                "the LiDAR plugin will be automatically used.")
            cmake_default = cmake_default.replace(
                'set( PLUGINS "visualizer;canopygenerator;syntheticannotation" )',
                'set( PLUGINS "visualizer;canopygenerator;syntheticannotation;lidar" )')
        cmake = cmake_default.replace('generate.cpp', 'main.cpp').replace(
            'SyntheticImageAnnotation', project_name
        )
    else:
        cmake = load_code_from_string_or_file(cmake)

    # Move in the necessary files.
    with open(os.path.join(project_path, 'main.cpp'), 'w') as f:
        f.write(code)
    with open(os.path.join(project_path, 'CMakeLists.txt'), 'w') as f:
        f.write(cmake)
    if xml is not None:
        with open(os.path.join(project_path, xml_fname)) as f:
            f.write(xml)

    # Return the project name and path.
    return project_path, project_name, cmake


