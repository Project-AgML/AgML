# AgML

AgML is a comprehensive library for agricultural machine learning. Currently, AgML provides
access to a wealth of public agricultural datasets for common agricultural deep learning tasks.  


## Installation

To install the latest release of AgML, 


### Getting Started
All of the publicly available datasources available in AgML, along with their associated meta-data, are described within [`data_sources.json`](/src/assets/data_sources.json). 

You can also view this information, along with available datasets, via the public data API.

```python
import agml
print(agml.public_data_sources())
```

You can filter the sources by a number of different filters, such as the ML or agricultural task, the location
the data was collected, or the number of images:

```python
# Get all of the data sources with over 1,000 images.
print(agml.public_data_sources(n_images = '>1000'))

# Get all of the object detection data sources.
print(agml.public_data_sources(ml_task = 'object_detection'))
```

The core of AgML's public data pipeline is the [`AgMLDataLoader`](/agml/data/loader.py), which provides direct
access to the public data sources, and provides integration methods to use data sources in a variety of methods, including
an AgML training pipeline, but also existing TensorFlow or PyTorch pipelines.

```
adapi.download_dataset(dataset_name='bean_disease_uganda', dest_dir='DESTINATION_DIRECTORY_PATH_HERE')
```

Running this method will download the dataset which has been formatted in one of the standard annotation formats described below.

## Generating synthetic RGB and LiDAR data using Helios 3D Crop Modeling Software
### Getting started
The `Helios Annotation API` assumes that you have cloned [`Helios`](https://github.com/PlantSimulationLab/Helios).

```
# Import Helios Annotation API (i.e. syntheticdata)
from data import syntheticdata

# Initialize HeliosDataGenerator class (optionally specify path to Helios)
hdg = syntheticdata.HeliosDataGenerator(path_helios_dir='../../../Helios/') 

# User can define a range of values for any parameter from which values are randomly sampled. For example:
hdg.canopy_param_ranges['VSPGrapevine']['leaf_spacing_fraction'] = [[0.4, 0.8]]
hdg.canopy_param_ranges['VSPGrapevine']['leaf_subdivisions'] = [[1, 5], [1, 5]]
hdg.canopy_param_ranges['VSPGrapevine']['leaf_width'] = [[0.1, 0.3]]
hdg.canopy_param_ranges['VSPGrapevine']['grape_color'] = [[0.15, 0.20], [0.15, 0.25], [0.2, 0.3]]
hdg.canopy_param_ranges['VSPGrapevine']['cluster_radius'] = [[0.025, 0.035]]

# User can generate synthetic data from a LiDAR and an RGB camera
# ---------------- LiDAR PARAMETERS --------------------
# For the LiDAR, the user can set all the metada. If not set, the default values will be used.
hdg.lidar_param_ranges['ASCII_format'] = ['x y z object_label'] # Available: x, y,  z, zenith, azimuth, r, g, b, target_count, target_index, timestamp, deviation, intensity, object_label
hdg.lidar_param_ranges['thetaMax'] = [130]
hdg.lidar_param_ranges['thetaMin'] = [30]
hdg.lidar_param_ranges['exitDiameter'] = [0.05]
hdg.lidar_param_ranges['size'] = [250, 250]
hdg.lidar_param_ranges['origin'] = [[0, 0, 0],[1, 1, 1],[2, 2, 2]] # Multiple LiDAR's could be used, it is only need to defined the origin of each sensor
# ---------------- CAMERA PARAMETERS --------------------
# For the RGB images, the user can set the camera positions and the desire image_resoution. (Expecting this changes in Helios to be effective)
# Camera position are a set of triplets specifiying the -x, y, and z- cordinates.
hdg.camera_param_ranges['camera_position'] = [[1,-1,2],[1,1,1]]
hdg.camera_param_ranges['image_resolution'] = [1000, 500]
# ---------------- GENERATE SYNTHETIC DATA -------------
# Generate a user-specified number of simulated rgb/lidar outputs + annotations for a given canopy type in Helios
# Note1: simulation_type='lidar' requires a GPU with CUDA installed
# Note2: simulation_type='rgb' requieres access to Graphic interface
# Note3: User can specified the output directory path. If it is not defined, the output data will be saved in the current directory.
hdg.generate_data(n_imgs=10, canopy_type='VSPGrapevine', simulation_type='rgb', output_directory ="/home/username/Documents")
# ---------------- CONVERT HELIOS OUTPUT TO STANDARD COCO JSON FORMAT -------------
# Generate a user-specified number of simulated rgb/lidar outputs + annotations for a given canopy type in Helios
# Note1: simulation_type='lidar' requires a GPU with CUDA installed
# Note2: simulation_type='rgb' requieres access to Graphic interface
# Note3: User can specified the output directory path. If it is not defined, the output data will be saved in the current directory.
output_directory ="/home/username/Documents/output/images/"
hdg.convert_data(output_directory, annotation_format='object_detection')

Running this code block generates 10 synthetic data (point clouds or rgb images). 1 xml files is created on each iteration which will be used to initialize Helios crop geometries. The geometries are random combinations of the user specified range of values defined above. 
```
## Optional

### Pass in a manual seed for consistency in parameter shuffling
```
seed=10
hdg.set_seed(seed)
```
### CONVERT HELIOS OUTPUT TO STANDARD COCO JSON FORMAT 
Given the path to the output of Helios, this method can be used to convert the data to a more standard format such as COCO JSON. Choose between 'instance segmentation' and object_detection'. The default is 'instance_segmentation'.
```
output_directory ="/home/username/Documents/output/images/"
hdg.convert_data(output_directory, annotation_format='object_detection')
```

## To-Do
- Code to run Helios using a system call to c++ here

### Standard annotation formats

- Image Classification: TODO
- Image Regression: TODO
- Semantic Segmentation: TODO
- Object Detection: TODO
- Instance Segmentation: TODO
- Panoptic Segmentation: TODO

### Notes
For easily removing MACOSX files...
`zip -r dir.zip . -x ".*" -x "__MACOSX"`


