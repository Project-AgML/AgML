import json
import csv 
import copy 
import os
import numpy as np
from dict2xml import dict2xml
import platform
import subprocess
import sys

from skimage.io import imread, imshow
import skimage
from skimage.morphology import closing
import imantics
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

HELIOS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), '_helios/Helios')


class HeliosDataGenerator(object):

    def __init__(self, path_helios_dir=HELIOS_PATH):
        self.path_canopygen_header = os.path.join(
            path_helios_dir, 'plugins/canopygenerator/include/CanopyGenerator.h')
        self.path_canopygen_cpp = os.path.join(
            path_helios_dir, 'plugins/canopygenerator/src/CanopyGenerator.cpp')
        self.path_cmakelists = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), '_helios/Helios/projects/SyntheticImageAnnotation/CMakeLists.txt')
        self.path_main_cpp = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), '_helios/Helios/projects/SyntheticImageAnnotation/main.cpp')

        self.path_lidar_cpp = os.path.join(
            path_helios_dir, 'plugins/lidar/src/LiDAR.cpp')
        self.canopy_types = self.get_canopy_types()
        self.canopy_params = self.get_canopy_params()
        self.canopy_param_ranges = self.set_initial_canopy_param_ranges()
        self.lidar_params = self.get_lidar_params()
        self.lidar_param_ranges = self.set_initial_lidar_param_ranges()
        self.camera_params = self.get_camera_params()
        self.camera_param_ranges = self.set_initial_camera_param_ranges()

    def get_canopy_types(self):

        """
        Find all occurrences of 'struct' in canopygen_header_txt.
        Parse canopy type from struct occurrences.
        Generate a list of canopy types.
        """

        # Read CanopyGenerator.h to define potential canopy types
        with open(self.path_canopygen_header) as f:
            canopygen_header_txt = f.readlines()

        # Generate list of canopy types
        canopy_types = []
        search_term = 'struct '
        length_search_term = len(search_term)
        for i, string in enumerate(canopygen_header_txt):
            if search_term in string:
                canopy_types.append(string[length_search_term:].split('Parameters{')[0])

        return canopy_types

    def set_seed(self, seed):
        np.random.seed(seed)

    def get_canopy_params(self):

        # Flag for parsing
        param_flag = 0

        # Initialize canopy parameters dictionary
        canopy_params = {}

        # Read CanopyGenerator.cpp to define potential canopy types
        with open(self.path_canopygen_cpp) as f:
            canopygen_header_txt = f.readlines()

            # Find parameters for each canopy type
        for canopy_type in self.canopy_types:

            # Find first line of parameter definition in cpp file
            search_term = canopy_type + 'Parameters::'

            for i, string in enumerate(canopygen_header_txt):

                if param_flag == 1:
                    if string != '\n' and string != '}\n' and string != '  \n':
                        line = string.split(';\n')[0]
                        key, value = line.split(' = ')

                        if '(' in value:
                            value = value.split('(')[1].split(')')[0].replace(',', ' ')

                        canopy_params[canopy_type][key.strip()] = value.strip().strip('\"').replace('\n', '').replace(
                            ';', '').replace('.f', '0')

                        if len(canopy_params[canopy_type][key.strip()]) == 1:
                            canopy_params[canopy_type][key.strip()] = canopy_params[canopy_type][key.strip()][0]

                        if 'M_PI' in value:
                            new_val = float(value.split('*')[0]) * 3.14159
                            canopy_params[canopy_type][key.strip()] = str(new_val)

                if search_term in string:
                    canopy_params[canopy_type] = {}
                    param_flag = 1

                if '}' in string and param_flag == 1:
                    param_flag = 0

        return canopy_params

    def get_lidar_params(self):

        # Flag for parsing
        param_flag = 0

        # Initialize canopy parameters dictionary
        lidar_params = {}

        # Read LiDAR.cpp to find parameters
        with open(self.path_lidar_cpp) as f:
            LiDAR_txt = f.readlines()

            # Find Metadata line of parameter definition in cpp file
            search_term = 'ScanMetadata::ScanMetadata'

            for i, string in enumerate(LiDAR_txt):

                if param_flag == 1:

                    if string != '\n' and string != '}\n' and string != '  \n' and string != '  //Copy arguments into structure variables\n':

                        line = string.split(';\n')[0]
                        key = line.split(' = ')
                        # Initialization of values from Helios -- Need to read this values from c++
                        if key[0].strip() == 'origin':
                            lidar_params[key[0].strip()] = '0 0 0'
                        elif key[0].strip() == 'Ntheta':
                            lidar_params['size'] = '250'
                        elif key[0].strip() == 'thetaMin':
                            lidar_params[key[0].strip()] = '0'
                        elif key[0].strip() == 'thetaMax':
                            lidar_params[key[0].strip()] = '180'
                        elif key[0].strip() == 'Nphi':
                            lidar_params['size'] = lidar_params['size'] + ' 450'
                        elif key[0].strip() == 'phiMin':
                            lidar_params[key[0].strip()] = '0'
                        elif key[0].strip() == 'phiMax':
                            lidar_params[key[0].strip()] = '360'
                        elif key[0].strip() == 'exitDiameter':
                            lidar_params[key[0].strip()] = '0'
                        elif key[0].strip() == 'beamDivergence':
                            lidar_params[key[0].strip()] = '0'
                        elif key[0].strip() == 'columnFormat':
                            lidar_params['ASCII_format'] = 'x y z'

                if 'ScanMetadata::ScanMetadata' in string:
                    param_flag = 1

                if '}' in string and param_flag == 1:
                    param_flag = 0
        return lidar_params

    def get_camera_params(self):

        # Initialize canopy parameters dictionary
        camera_params = {}
        
        # Initialization of image resolution and camera position
        camera_params['image_resolution'] = '600 400'
        camera_params['camera_lookat'] = '0 0 1'
        camera_params['camera_position'] = '0 -2 1'
        

        return camera_params

    def set_initial_lidar_param_ranges(self):

        lidar_param_ranges = copy.deepcopy(self.lidar_params)

        # Check if parameter is a path or number; this assumes that all strings will be paths
        for i in lidar_param_ranges.keys():
            if i == 'ASCII_format':
                val = lidar_param_ranges[i]
                lidar_param_ranges[i] = [(val.split(' ')[j].replace('f', '0')) for j in range(len(val.split()))]
            else:
                val = lidar_param_ranges[i]
                lidar_param_ranges[i] = [float(val.split(' ')[j].replace('f', '0')) for j in range(len(val.split()))]

        return lidar_param_ranges

    def set_initial_canopy_param_ranges(self):

        canopy_param_ranges = copy.deepcopy(self.canopy_params)

        # Check if parameter is a path or number; this assumes that all strings will be paths
        for i in canopy_param_ranges.keys():
            for key in list(canopy_param_ranges[i]):
                val = canopy_param_ranges[i][key]
                if val.isalpha() or '/' in val or ':' in val:
                    canopy_param_ranges[i].pop(key)
                else:
                    canopy_param_ranges[i][key] = [
                        [float(val.split(' ')[j].replace('f', '0')), float(val.split(' ')[j].replace('f', '0'))] for j
                        in range(len(val.split()))]

        return canopy_param_ranges

    def set_initial_camera_param_ranges(self):

        camera_param_ranges = copy.deepcopy(self.camera_params)

        # Check if parameter is a path or number; this assumes that all strings will be paths
        for i in camera_param_ranges.keys():
            val = camera_param_ranges[i]
            camera_param_ranges[i] = [float(val.split(' ')[j]) for j in range(len(val.split()))]

        return camera_param_ranges

    def generate_one_datapair(self, canopy_type, simulation_type, export_format = 'xml'):

        """
        Find all occurrences of 'struct' in canopygen_header_txt.
        Parse canopy type from struct occurrences.
        Generate a list of canopy types.

        Args:
        canopy_type (string): the selected canopy type for the synthetic images
            It should contain a out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or and OrderedDict[Tensor].
        simulation_type (string): choose between RGB only and Lidar mode using 'rgb' or 'lidar'
        export_format (string): default is xml for Helios
        """

        assert canopy_type in self.canopy_types, 'Canopy type not available.'

        assert export_format in ['xml', 'csv', 'json'], 'Only xml, csv and json export formats are possible.'

        canopy_params_filtered = {k: v for k, v in self.canopy_params.items() if k.startswith(canopy_type)}

        canopy_params_filtered[canopy_type + 'Parameters'] = canopy_params_filtered.pop(canopy_type)

        canopy_params_filtered['Ground'] = {'origin': '0 0 0',
                                            'extent': '10 10',
                                            'texture_subtiles': '10 10',
                                            'texture_subpatches': '1 1',
                                            'ground_texture_file': 'plugins/canopygenerator/textures/dirt.jpg',
                                            'rotation': '0'}

        canopy_params_filtered = {'canopygenerator': canopy_params_filtered}

        if simulation_type == 'lidar':
            canopy_params_filtered['scan'] = self.lidar_params

        if simulation_type == 'rgb':
            canopy_params_filtered[''] = self.camera_params

        canopy_params_filtered = {'helios': canopy_params_filtered}
        from pprint import pprint
        # print(canopy_params_filtered)

        if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), '_helios/Helios/projects/SyntheticImageAnnotation/xml')):
            os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), '_helios/Helios/projects/SyntheticImageAnnotation/xml'))

        if export_format == 'xml':
            with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), '_helios/Helios/projects/SyntheticImageAnnotation/xml/tmp_canopy_params_image.xml'), "w") as f:
                f.write(dict2xml(canopy_params_filtered))

    def generate_data(self, n_imgs, canopy_type, simulation_type, annotation_type, label_elements='none', output_directory = ".."):


        """
        Given the path to the output of Helios, this method can be used to convert the data to a more standard format such as COCO JSON

        Args:
        n_imgs (int): The number of images that to be generated
        canopy_type (string): the selected canopy type for the synthetic images
        simulation_type (string): choose between RGB only and Lidar mode using 'rgb' or 'lidar'
        annotation_type (string): choose between semantic segmentation, object detection, and instance segmentation as 'semantic', 'object', and 'instance'.
        output_directory (string) (optional): optionally you may pass in a custom path to save the Helios output to the custom path

        """
        output_directory=os.path.abspath(output_directory)

        assert simulation_type in ['rgb', 'lidar'], 'Simulation type unavailable.'

        assert annotation_type in ['semantic', 'object', 'instance', 'none'], 'Annotation type unavailable.'

        Labels = label_elements.replace(' ', '').split(',')

        for j in range(len(Labels)):
            assert Labels[j] in ['trunk', 'leaves', 'fruits', 'branches', 'ground', 'none'], 'Label type unavailable'


        param_ranges = self.canopy_param_ranges[canopy_type]

        camera_ranges = self.camera_param_ranges

        if simulation_type == 'lidar':
            
            lidar_ranges = copy.deepcopy(self.lidar_param_ranges)
            
            # LiDAR parameters
            for key in lidar_ranges:
                # param_vals=lidar_params[key].split(' ')
                arr = [lidar_ranges[key][i] for i in range(len(lidar_ranges[key]))]
                string_arr = [str(a) for a in arr]
                self.lidar_params[key] = ' '.join(string_arr)
            # Mutiple LiDAR
            LiDARs = []
            for i in range(len(lidar_ranges['origin'])):
                for key in lidar_ranges:
                    if key == 'origin':
                        arr = [lidar_ranges[key][i] for i in range(len(lidar_ranges[key]))]
                        arr = arr[i]
                        string_arr = [str(a) for a in arr]
                        self.lidar_params[key] = ' '.join(string_arr)
                    A = copy.deepcopy(self.lidar_params)
                LiDARs.append(A)
            self.lidar_params = LiDARs

        if simulation_type == 'rgb':
            # Camera parameters
            for key in camera_ranges:
                arr = [camera_ranges[key][i] for i in range(len(camera_ranges[key]))]
                string_arr = [str(a).replace(',', '').replace('[', ' ').replace(']', ' ') for a in arr]
                self.camera_params[key] = ' '.join(string_arr)


        for n in range(n_imgs):
            params = self.canopy_params[canopy_type]
            # lidar_params = self.lidar_params

            # Context parameters
            for key in param_ranges:
                print(param_ranges[key])
                arr = [np.random.uniform(param_ranges[key][i][0], param_ranges[key][i][1]) for i in
                       range(len(param_ranges[key]))]
                string_arr = [str(a) for a in arr]
                params[key] = ' '.join(string_arr)
            self.canopy_params[canopy_type] = params
            print(self.canopy_params[canopy_type])

            self.generate_one_datapair(canopy_type, simulation_type)

            # Re-write tags of XML to have the expected Helios input
            tree = ET.parse(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                         '_helios/Helios/projects/SyntheticImageAnnotation/xml',
                                         'tmp_canopy_params_image.xml'))
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


            tree.write(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    '_helios/Helios/projects/SyntheticImageAnnotation/xml', 'tmp_canopy_params_image.xml'))

            # Modify cmake file for rgb versus lidar simulation
            with open(self.path_cmakelists) as f:
                cmakelists_txt = f.readlines()

            for i, string in enumerate(cmakelists_txt):
                if 'set( PLUGINS ' in string and simulation_type == 'lidar':
                    cmakelists_txt[i] = 'set( PLUGINS "lidar;visualizer;canopygenerator;syntheticannotation" )\n'

                if 'set( PLUGINS ' in string and simulation_type == 'rgb':
                    cmakelists_txt[i] = 'set( PLUGINS "visualizer;canopygenerator;syntheticannotation" )\n'

            # and write everything back
            with open(self.path_cmakelists, 'w') as f:
                f.writelines(cmakelists_txt)

            # Modify maincpp file for rgb versus lidar simulation
            # with open(self.path_main_cpp) as f:
            #     main_cpp = f.readlines()

            # Define paths for CMAKE compilation and output files
            current_directory = os.getcwd()
            build_dir = os.path.join(current_directory, 'Helios/projects/SyntheticImageAnnotation/build')
            if output_directory == "":
                output_dir = os.path.join(current_directory, 'Helios/projects/SyntheticImageAnnotation/output')
            else:
                assert os.path.isdir(output_directory), 'Please, introduce a valid directory'
                output_dir = output_directory + '/output'
            point_cloud_dir = os.path.join(current_directory, output_dir + '/point_cloud/')
            images_dir = os.path.join(current_directory, output_dir + '/images/')

            if not os.path.exists(build_dir):
                os.makedirs(build_dir)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            if simulation_type == 'lidar':
                if not os.path.exists(point_cloud_dir):
                    os.makedirs(point_cloud_dir)

            if simulation_type == 'rgb':
                if not os.path.exists(images_dir):
                    os.makedirs(images_dir)

            exe = os.path.join(build_dir, 'SyntheticImageAnnotation')

            # Modify main.cpp file for compilation in LIDAR and RGB case
            """
            print('Generation synthetic data: #' + str(n))
            for i, string in enumerate(main_cpp):
                
                # Managing annotation types requested by user
                if 'annotation.disableSemanticSegmentation();' in string and 'semantic' not in annotation_type:
                    main_cpp[i] = '    annotation.disableSemanticSegmentation();\n'

                if 'annotation.disableSemanticSegmentation();' in string and 'semantic' in annotation_type:
                    main_cpp[i] = '    //annotation.disableSemanticSegmentation();\n'

                if 'annotation.disableObjectDetection();' in string and 'object' not in annotation_type:
                    main_cpp[i] = '    annotation.disableObjectDetection();\n'

                if 'annotation.disableObjectDetection();' in string and 'object' in annotation_type:
                    main_cpp[i] = '    //annotation.disableObjectDetection();\n'

                if 'annotation.disableInstanceSegmentation();' in string and 'instance' not in annotation_type:
                    main_cpp[i] = '    annotation.disableInstanceSegmentation();\n'

                if 'annotation.disableInstanceSegmentation();' in string and 'instance' in annotation_type:
                    main_cpp[i] = '    //annotation.disableInstanceSegmentation();\n'
                
                # Managing LiDAR versus rgb request by user
                if '#include "L' in string and simulation_type == 'lidar':
                    main_cpp[i] = '#include "LiDAR.h"\n'

                if '#include "L' in string and simulation_type == 'rgb':
                    main_cpp[i] = '//#include "LiDAR.h"\n'

                if 'flag=' in string and simulation_type == 'lidar':
                    main_cpp[i] = 'bool flag=true;\n'

                if 'flag=' in string and simulation_type == 'rgb':
                    main_cpp[i] = 'bool flag=false;\n'

                if 'LiDARcloud lidarcloud' in string and simulation_type == 'lidar':
                    main_cpp[i] = ' LiDARcloud lidarcloud;\n'

                if 'LiDARcloud lidarcloud' in string and simulation_type == 'rgb':
                    main_cpp[i] = ' //LiDARcloud lidarcloud;\n'

                if 'lidarcloud.loadXML' in string and simulation_type == 'lidar':
                    main_cpp[i] = ' lidarcloud.loadXML("../xml/tmp_canopy_params_image.xml");\n'

                if 'lidarcloud.loadXML' in string and simulation_type == 'rgb':
                    main_cpp[i] = ' //lidarcloud.loadXML("../xml/tmp_canopy_params_image.xml");\n'

                if 'lidarcloud.syntheticScan' in string and simulation_type == 'lidar':
                    main_cpp[i] = ' lidarcloud.syntheticScan( &context);\n'

                if 'lidarcloud.syntheticScan' in string and simulation_type == 'rgb':
                    main_cpp[i] = ' //lidarcloud.syntheticScan( &context);\n'

                #
                if 'lidarcloud.exportPointCloud' in string and simulation_type == 'lidar':
                    main_cpp[i] = ' lidarcloud.exportPointCloud( "' + output_dir + '/point_cloud/synthetic_scan_' + str(
                        n) + '.xyz" );\n'

                if 'lidarcloud.exportPointCloud' in string and simulation_type == 'rgb':
                    main_cpp[
                        i] = ' //lidarcloud.exportPointCloud( "' + output_dir + '/point_cloud/synthetic_scan_' + str(
                        n) + '.xyz" );\n'

                # Change output  directory -- if requested

                if 'annotation.render(' in string and simulation_type == 'rgb':
                    main_cpp[i] = ' annotation.render( "' + images_dir + '" );\n'

                # Choose elements to label
                for k in Labels:
                        
                    if 'annotation.labelPrimitives(cgen.getTrunkUUIDs(p)' in string and simulation_type == 'rgb' and 'trunk' in Labels:
                        main_cpp[i] = ' annotation.labelPrimitives(cgen.getTrunkUUIDs(p), "trunks");\n'
                    elif  'annotation.labelPrimitives(cgen.getTrunkUUIDs(p)' in string and simulation_type == 'rgb' and 'trunk' not in Labels:
                        main_cpp[i] = ' //annotation.labelPrimitives(cgen.getTrunkUUIDs(p), "trunks");\n'
                    
                    if 'annotation.labelPrimitives(cgen.getBranchUUIDs(p)' in string and simulation_type == 'rgb' and 'branches' in Labels:
                        main_cpp[i] = ' annotation.labelPrimitives(cgen.getBranchUUIDs(p), "branches");\n'
                    elif 'annotation.labelPrimitives(cgen.getBranchUUIDs(p)' in string and simulation_type == 'rgb' and 'branches' not in Labels:
                        main_cpp[i] = ' //annotation.labelPrimitives(cgen.getBranchUUIDs(p), "branches");\n'
                    
                    if 'annotation.labelPrimitives(cgen.getLeafUUIDs(p)' in string and simulation_type == 'rgb' and 'leaves' in Labels:
                        main_cpp[i] = ' annotation.labelPrimitives(cgen.getLeafUUIDs(p), "leaves");\n'
                    elif 'annotation.labelPrimitives(cgen.getLeafUUIDs(p)' in string and simulation_type == 'rgb' and 'leaves' not in Labels:
                        main_cpp[i] = ' //annotation.labelPrimitives(cgen.getLeafUUIDs(p), "leaves");\n'

                    if 'annotation.labelPrimitives( flatten(fruitUUIDs.at(c))' in string and simulation_type == 'rgb' and 'fruits' in Labels:
                        main_cpp[i] = ' annotation.labelPrimitives( flatten(fruitUUIDs.at(c)), "clusters" );\n'
                    elif 'annotation.labelPrimitives( flatten(fruitUUIDs.at(c))' in string and simulation_type == 'rgb' and 'fruits' not in Labels:
                        main_cpp[i] = ' //annotation.labelPrimitives( flatten(fruitUUIDs.at(c)), "clusters" );\n'



            # and write everything back
            with open(self.path_main_cpp, 'w') as f:
                f.writelines(main_cpp)
            """

            default_xml_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                '_helios/Helios/projects/SyntheticImageAnnotation/xml',
                'tmp_canopy_params_image.xml')
            str_to_write = f"{annotation_type}\n{simulation_type}\n{' '.join(Labels)}\n{default_xml_path}\n{images_dir}"
            with open(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                   '_helios/Helios/projects/SyntheticImageAnnotation/config.txt'), 'w') as f:
                f.write(str_to_write)


            # System call to helios @DARIO
            # current_directory = os.getcwd()
            helios_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), '_helios')
            build_dir = os.path.join(helios_directory, 'Helios/projects/SyntheticImageAnnotation/build')
            output_dir = os.path.join(helios_directory, 'Helios/projects/SyntheticImageAnnotation/output')
            point_cloud_dir = os.path.join(helios_directory, 'Helios/projects/SyntheticImageAnnotation/output/point_cloud/')
            images_dir = os.path.join(helios_directory, 'Helios/projects/SyntheticImageAnnotation/output/images/')

            if not os.path.exists(build_dir):
                os.makedirs(build_dir)
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            if simulation_type == 'lidar':
                if not os.path.exists(point_cloud_dir):
                    os.makedirs(point_cloud_dir)

            if simulation_type == 'rgb':
                if not os.path.exists(images_dir):
                    os.makedirs(images_dir)

            exe = os.path.join(build_dir, 'SyntheticImageAnnotation')

            cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + build_dir]
            cmake_args += ['-G', 'Unix Makefiles']
            # current_directory = os.getcwd()
            helios_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), '_helios')
            build_dir = os.path.join(helios_directory, 'Helios/projects/SyntheticImageAnnotation/build')
            output_dir = os.path.join(helios_directory, 'Helios/projects/SyntheticImageAnnotation/output')
            point_cloud_dir = os.path.join(helios_directory, 'Helios/projects/SyntheticImageAnnotation/output/point_cloud/')
            images_dir = os.path.join(helios_directory, 'Helios/projects/SyntheticImageAnnotation/output/images/')

            if not os.path.exists(build_dir):
                os.makedirs(build_dir)
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            if simulation_type == 'lidar':
                if not os.path.exists(point_cloud_dir):
                    os.makedirs(point_cloud_dir)

            if n == 0:
                subprocess.run(['cmake', ".."] + cmake_args, cwd = build_dir, check = True)
            subprocess.run(['cmake', '--build', '.'], cwd = build_dir, check = True)

            subprocess.run([exe, '/Users/amoghjoshi/Documents/Amogh/Programs/Python/AgML/agml/_helios/Helios/projects/SyntheticImageAnnotation/config.txt',
                            output_dir], cwd = build_dir)

        # self.convert_data() # Conversion to standard formats @PRANAV

    def convert_data(self, Frames_path, annotation_format = 'instance_segmentation'):

        """
        Given the path to the output of Helios, this method can be used to convert the data to a more standard format such as COCO JSON

        Args:
        annotation_format (string): choose between 'instance_segmentation', 'object_detection', or 'panoptic_segmentation'
        Frames_path (string): Specify the path to the output of Helios

        """

        if annotation_format == 'object_detection':
            pass  # YOLO @PRANAV
        if annotation_format == 'semantic_segmentation':
            pass  # single channel + mapping @PRANAV
        if annotation_format == 'instance_segmentation' or annotation_format == 'object_detection':

            # data_path = sorted(os.listdir(data_path))
            frames_view = sorted(os.listdir(Frames_path))
            # Initialize a list of label numpy arrays
            # imgs = []
            # npy_arrs = []
            images = []
            if not os.path.exists('./train_images/'):
                os.mkdir('./train_images/')
            for i, frame in enumerate(frames_view):
                if frame == '.DS_Store':
                    continue


                else:

                    # Generate labels and append to list
                    image, mask, pixel_ID = self.generate_npy_arr(Frames_path, Frames_path + frame + '/')
                    plt.imsave('./train_images/' + frame + '.jpeg', image)

                    strel = skimage.morphology.selem.disk(2)

                    # %% Convert masks to COCO for detectron implementation.

                    mapping = pd.read_csv(Frames_path + frame + '/' + 'ID_mapping.txt', delim_whitespace = True,
                                          names = ['ID', 'Class'])

                    category = imantics.Category(mapping['Class'].iloc[pixel_ID],
                                                 color = imantics.Color([255, 0, 0]))  # color for debug only
                    c = 0

                    im2 = imantics.Image(image_array = image, path = 'data/train_images/' + frame + '.jpeg',
                                         id = i)  # imantics image object

                    for i in range(mask.shape[2]):  # create the polygons for each slice

                        if annotation_format == 'instance_segmentation':
                            poly = imantics.Mask(closing(mask[:, :, i], strel)).polygons()
                            ann = imantics.Annotation(image = im2, category = category, polygons = poly, id = c)
                            box = imantics.Mask(closing(mask[:, :, i], strel)).bbox()
                            # print(box)

                        if annotation_format == 'object_detection':
                            box = imantics.Mask(closing(mask[:, :, i], strel)).bbox()
                            ann = imantics.Annotation(image = im2, category = category, bbox = box, id = c)
                            # print(box)
                        c += 1

                        if np.sum(mask[:, :, i]) > 1300:
                            im2.add(ann)  # add the polygon to the image object
                            print(box)

                    images.append(im2)  # collect the image objects after they get polygons
                # %% Create Imantics dataset and export
            ds = imantics.Dataset(name = 'coco', images = images)
            obj = ds.coco()
            with open('trainval.json', 'w') as json_file:
                json.dump(obj, json_file)
            #     json_file.close()
            # obj=obj.clear()
            # del obj, ds, json_file, images

            # COCO json @PRANAV
        if annotation_format == 'panoptic_segmentation':
            pass
        if annotation_format == 'regression':
            pass

    def generate_npy_arr(self, Frames_path, frames_view_path):

        Frame_paths = frames_view_path

        # Grab dimensions
        render = imread(frames_view_path + '/RGB_rendering' + '.jpeg')
        render_xy_shape = (render.shape[0], render.shape[1])
        # print(render_xy_shape)

        # Number of instances
        frames_list = []
        exclude_files = ['RGB_rendering.jpeg', 'pixelID_combined.txt', 'ID_mapping.txt']
        frames_files = [x for x in os.listdir(frames_view_path) if x not in exclude_files]
        frames_list.append(frames_files)
        n_instances = [len(frame) for frame in frames_list]

        # Initialize numpy array of shape (x, y, n_instances)
        npy_arr = np.zeros(shape = (render_xy_shape[0], render_xy_shape[1], n_instances[0]),
                           dtype = np.bool_)  # Make boolean dtype

        # Read frames as a list
        bboxes = []
        frames = []
        i = 0
        for file_grape_arr in os.listdir(frames_view_path):

            if file_grape_arr == '.DS_Store':
                continue

            if file_grape_arr == 'pixelID_combined.txt':
                continue

            if file_grape_arr == 'ID_mapping.txt':
                continue

            if file_grape_arr == 'RGB_rendering.jpeg':
                continue

            if file_grape_arr == 'rectangular_labels.txt':
                continue

            else:
                # Extract pixel id based off name of the file. We want the number between _ and .

                pixel_id_tmp = int(file_grape_arr.split('_')[1].split('.')[0])
                # print(pixel_id_tmp)

                # Load bounding box coordinates from file
                bbox_tmp = np.loadtxt(frames_view_path + file_grape_arr, max_rows = 1)
                bbox_tmp = [int(bbox_tmp[i]) for i in range(0, 4)]
                bbox = np.copy(bbox_tmp)
                bbox[3] = render_xy_shape[0] - bbox_tmp[2]
                bbox[2] = render_xy_shape[0] - bbox_tmp[3]

                # print(bbox_tmp)

                # Load pixel positions from file
                grape_arr_tmp = np.loadtxt(frames_view_path + file_grape_arr, skiprows = 1, ndmin = 2)

                #
                grape_arr_tmp = (grape_arr_tmp == pixel_id_tmp)

                npy_arr[bbox[2]:(bbox[3] + 1), bbox[0]:(bbox[1] + 1), i] = grape_arr_tmp

                i += 1

        return render, npy_arr, pixel_id_tmp
