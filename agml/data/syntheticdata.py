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

HELIOS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), '_helios/Helios')

class HeliosDataGenerator(object):

    def __init__(self, path_helios_dir=HELIOS_PATH):
        
        self.path_canopygen_header = os.path.join(
            path_helios_dir, 'plugins/canopygenerator/include/CanopyGenerator.h')
        self.path_canopygen_cpp = os.path.join(
            path_helios_dir, 'plugins/canopygenerator/src/CanopyGenerator.cpp')
        self.path_cmakelists = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), '_helios/CMakeLists.txt')
        self.path_main_cpp = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), '_helios/main.cpp')
        self.canopy_types = self.get_canopy_types()
        self.canopy_params = self.get_canopy_params()
        self.canopy_param_ranges = self.set_initial_canopy_param_ranges()


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

                        canopy_params[canopy_type][key.strip()] = value.strip().strip('\"').replace('\n', '').replace(';', '').replace('.f', '0')
                        
                        if len(canopy_params[canopy_type][key.strip()])==1:
                            canopy_params[canopy_type][key.strip()]=canopy_params[canopy_type][key.strip()][0]

                        if 'M_PI' in value:
                            new_val = float(value.split('*')[0]) * 3.14159
                            canopy_params[canopy_type][key.strip()] = str(new_val)

                if search_term in string:
                    canopy_params[canopy_type] = {}
                    param_flag = 1

                if '}' in string and param_flag == 1:
                    param_flag = 0

        return canopy_params
        

    def set_initial_canopy_param_ranges(self):

        canopy_param_ranges = copy.deepcopy(self.canopy_params)

        # Check if parameter is a path or number; this assumes that all strings will be paths
        for i in canopy_param_ranges.keys():
            for key in list(canopy_param_ranges[i]):
                val=canopy_param_ranges[i][key]
                if val.isalpha() or '/' in val or ':' in val:
                    canopy_param_ranges[i].pop(key)
                else:
                    canopy_param_ranges[i][key] = [[float(val.split(' ')[j].replace('f', '0')), float(val.split(' ')[j].replace('f', '0'))] for j in range(len(val.split()))]

        return canopy_param_ranges


    def generate_one_datapair(self, canopy_type, simulation_type, export_format='xml'):

        assert canopy_type in self.canopy_types, 'Canopy type not available.'

        assert export_format in ['xml', 'csv', 'json'] , 'Only xml, csv and json export formats are possible.'

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
            canopy_params_filtered['scan'] = {'ASCII_format': 'x y z target_index target_count object_label',
                                            'origin': '0 0 0',
                                            'size': '250 450',
                                            'thetaMin': '30',
                                            'thetaMax': '130',
                                            'exitDiameter': '0.05',
                                            'beamDivergence': '0'}

        canopy_params_filtered = {'helios': canopy_params_filtered}

        if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), '_helios/xmloutput_for_helios')):
            os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), '_helios/xmloutput_for_helios'))

        if export_format == 'xml':
            with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), '_helios/xmloutput_for_helios/tmp_canopy_params_image.xml'), "w") as f:
                f.write(dict2xml(canopy_params_filtered))


    def generate_data(self, n_imgs, canopy_type, simulation_type):

        assert simulation_type in ['rgb', 'lidar'] , 'Specified simulation type is not available.'
        
        param_ranges=self.canopy_param_ranges[canopy_type]

        for n in range(n_imgs):
            params=self.canopy_params[canopy_type]

            for key in param_ranges:
                param_vals=params[key].split(' ')
                arr=[np.random.uniform(param_ranges[key][i][0],param_ranges[key][i][1]) for i in range(len(param_ranges[key]))]
                string_arr=[str(a) for a in arr]
                params[key]=' '.join(string_arr)
        
            self.canopy_params[canopy_type]=params
            self.generate_one_datapair(canopy_type, simulation_type)

            #### TEMPORARY
            self.canopy_params[canopy_type]['plant_count'] = '1 1'

            # Modify cmake file for rgb versus lidar simulation
            with open(self.path_cmakelists) as f:
                cmakelists_txt = f.readlines() 

            for i, string in enumerate(cmakelists_txt):
                if 'set( PLUGINS ' in string and simulation_type == 'lidar':
                    cmakelists_txt[i] = 'set( PLUGINS "lidar;visualizer;canopygenerator;" )\n'

                if 'set( PLUGINS ' in string and simulation_type == 'rgb':
                    cmakelists_txt[i] = 'set( PLUGINS "visualizer;canopygenerator;" )\n'

            # and write everything back
            with open(self.path_cmakelists, 'w') as f:
                f.writelines(cmakelists_txt)

            # Modify maincpp file for rgb versus lidar simulation
            with open(self.path_main_cpp) as f:
                main_cpp = f.readlines() 
            
            print('ITERATION ' + str(n))
            for i, string in enumerate(main_cpp):
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
                    main_cpp[i] = ' lidarcloud.loadXML("../xmloutput_for_helios/tmp_canopy_params_image.xml");\n'

                if 'lidarcloud.loadXML' in string and simulation_type == 'rgb':
                    main_cpp[i] = ' //lidarcloud.loadXML("../xmloutput_for_helios/tmp_canopy_params_image.xml");\n'

                if 'lidarcloud.syntheticScan' in string and simulation_type == 'lidar':
                    main_cpp[i] = ' lidarcloud.syntheticScan( &context);\n'

                if 'lidarcloud.syntheticScan' in string and simulation_type == 'rgb':
                    main_cpp[i] = ' //lidarcloud.syntheticScan( &context);\n'

                if 'lidarcloud.exportPointCloud' in string and simulation_type == 'lidar':
                    main_cpp[i] = ' lidarcloud.exportPointCloud( "../output/point_cloud/synthetic_scan_' + str(n) + '.xyz" );\n'

                if 'lidarcloud.exportPointCloud' in string and simulation_type == 'rgb':
                    main_cpp[i] = ' //lidarcloud.exportPointCloud( "../output/point_cloud/synthetic_scan.xyz" );\n'

                #Each run generate 10 images - then the geometry is changed
                if simulation_type == 'rgb':

                    if 'sprintf(outfile,"../output/images/RGB_rendering' in string:
                        main_cpp[i] = '  sprintf(outfile,"../output/images/RGB_rendering_' + str(n) + '_%d.jpeg",view);\n'

                    if 'sprintf(outfile,"../output/images/ID_mapping' in string:
                        main_cpp[i] = '  sprintf(outfile,"../output/images/ID_mapping_' + str(n) + '_%d.txt",view);\n'

                    if 'sprintf(outfile,"../output/images/pixelID_combined' in string:
                        main_cpp[i] = '  sprintf(outfile,"../output/images/pixelID_combined_' + str(n) + '_%d.txt",view);\n'

                    if 'sprintf(outfile,"../output/images/rectangular_labels_' in string:
                        main_cpp[i] = '  sprintf(outfile,"../output/images/rectangular_labels_' + str(n) + '_%d.txt",view);\n'
                        
                    if 'sprintf(outfile,"../output/images/pixelID2' in string:
                        main_cpp[i] = '  sprintf(outfile,"../output/images/pixelID2_' + str(n) + '_%d_%07d.txt",view,ID.at(p));\n'


            # and write everything back
            with open(self.path_main_cpp, 'w') as f:
                f.writelines(main_cpp)

            # System call to helios @DARIO
            # current_directory = os.getcwd()
            helios_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), '_helios')
            build_dir = os.path.join(helios_directory, 'build')
            output_dir = os.path.join(helios_directory, 'output')
            point_cloud_dir = os.path.join(helios_directory, 'output/point_cloud/')
            images_dir = os.path.join(helios_directory, 'output/images/')

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

            exe = os.path.join(build_dir, 'executable')

            cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + build_dir]
            cmake_args += ['-G', 'Unix Makefiles']

            if n==0:
                subprocess.run(['cmake', ".."] + cmake_args, cwd=build_dir, check=True)
            subprocess.run(['cmake', '--build', '.'], cwd=build_dir, check=True)

            subprocess.run([exe], cwd=build_dir)


        # self.convert_data() # Conversion to standard formats @PRANAV

    def convert_data(self, annotation_format, Frames_path):
        if annotation_format == 'object_detection':
            None # YOLO @PRANAV
        if annotation_format == 'semantic_segmentation':
            None # single channel + mapping @PRANAV
        if annotation_format == 'instance_segmentation' or annotation_format == 'object_detection':

            # data_path = sorted(os.listdir(data_path))
            frames_view = sorted(os.listdir(Frames_path))
            # Initialize a list of label numpy arrays
            # imgs = []
            # npy_arrs = []
            
            for i, frame in enumerate(frames_view):
                if frame == '.DS_Store':
                    continue


                else:

                    # Generate labels and append to list
                    render, npy_arr, pixel_ID = self.generate_npy_arr(Frames_path, Frames_path + frame + '/')
                    # npy_arrs.append(npy_arr)
                    # imgs.append(render)
                    
                    # Save labels as npy file
                    # if not os.path.exists('data/frames_npy/'):
                    #     os.makedirs('data/frames_npy/')
                    strel = skimage.morphology.selem.disk(2)

                    # %% Convert masks to COCO for detectron implementation.
                    images = []
                    # image_list = os.listdir('data/train_images/') # location of the jpeg images
                    # image_list.sort() # not necessary but might help to debug
                    # masks = '/home/kudeshpa/CNN-Synth-Grape-PyTorch/src/data/frames_npy/'

                    mapping=pd.read_csv(os.path.join(Frames_path + frame, 'ID_mapping.txt'), delim_whitespace=True, names=['ID', 'Class'])

                    category = imantics.Category(mapping['Class'].iloc[pixel_ID],color=imantics.Color([255,0,0])) # color for debug only
                    c = 0
                    # for s,im in enumerate(image_list):

                    image = render # grab image
                    mask = npy_arr # grab mask in the masks folder (same name but npy)

                    im2 = imantics.Image(image_array=image,path='data/train_images/'+frame,id=i) # imantics image object

                    for i in range(mask.shape[2]): # create the polygons for each slice

                        if annotation_format == 'instance_segmentation':
                            poly = imantics.Mask(closing(mask[:,:,i],strel)).polygons()
                            ann = imantics.Annotation(image=im2,category=category,polygons=poly,id=c)
                        if annotation_format == 'object_detection':
                            box = imantics.Mask(closing(mask[:,:,i],strel)).bbox()
                            ann = imantics.Annotation(image=im2,category=category,bbox=box,id=c)
                        c += 1
                        im2.add(ann) # add the polygon to the image object
                    
                    images.append(im2) # collect the image objects after they get polygons
                # %% Create Imantics dataset and export
                ds = imantics.Dataset(name='test', images=images)
                obj = ds.coco()
                with open('train.json', 'w') as json_file:
                    json.dump(obj, json_file)

                    # np.save('data/frames_npy/' + frame.split('.')[0] + '.npy', npy_arr)
                    
                     # COCO json @PRANAV
        if annotation_format == 'panoptic_segmentation':
            None
        if annotation_format == 'regression':
            None

    def generate_npy_arr(self, Frames_path, frames_view_path):


        Frame_paths = frames_view_path
        
        # Grab dimensions
        render = imread(os.path.join(frames_view_path, 'RGB_rendering' + '.jpeg'))
        render_xy_shape = (render.shape[0], render.shape[1])
        print(render_xy_shape)

        # Number of instances
        frames_list = []
        exclude_files = ['RGB_rendering.jpeg', 'pixelID_combined.txt', 'ID_mapping.txt']
        frames_files = [x for x in os.listdir(frames_view_path) if x not in exclude_files]
        frames_list.append(frames_files)
        n_instances = [len(frame) for frame in frames_list]
        print(n_instances)

        # Initialize numpy array of shape (x, y, n_instances)
        npy_arr = np.zeros(shape=(render_xy_shape[0], render_xy_shape[1], n_instances[0]), dtype=np.bool_) # Make boolean dtype

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

            else:
                # Extract pixel id based off name of the file. We want the number between _ and . 
                pixel_id_tmp = int(file_grape_arr.split('_')[1].split('.')[0])

                # Load bounding box coordinates from file
                bbox_tmp = np.loadtxt(frames_view_path + file_grape_arr, max_rows=1)
                bbox_tmp = [int(bbox_tmp[i]) for i in range(0, 4)]
                bbox = np.copy(bbox_tmp)
                bbox[3] = render_xy_shape[0] - bbox_tmp[2]  
                bbox[2] = render_xy_shape[0] - bbox_tmp[3]  

                print(bbox_tmp)

                # Load pixel positions from file
                grape_arr_tmp = np.loadtxt(frames_view_path + file_grape_arr, skiprows=1, ndmin=2)

                # 
                grape_arr_tmp = (grape_arr_tmp == pixel_id_tmp)

                npy_arr[bbox[2]:(bbox[3]+1), bbox[0]:(bbox[1]+1), i] = grape_arr_tmp

                i += 1

        return render, npy_arr, pixel_id_tmp
