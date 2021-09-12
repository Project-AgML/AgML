# AgML
Currently AgML provides access to publicly available and synthetic agricultural datasets for deep learning model training. In the future, we aim to provide full pipelines for data access, model training, and evaluation.

## Downloading publicly available agricultural datasets
### Getting started
All of the publicly available datasources available in AgML, along with their associated meta-data, are described within [`data_sources.json`](/src/assets/data_sources.json). 

You can also view this information, along with available datasets, via the public data API.

```
from agdata import agdata
adapi = agdata.AgDataAPI()
adapi.data_srcs
```

You can see the specific dataset names available like this:

```
adapi.data_srcs.keys()
```

```
output:
> dict_keys(['bean_disease_uganda', 'carrot_weeds_germany', 'carrot_weeds_macedonia', 'plant_seedlings_aarhus', 'soybean_weed_uav_brazil'])
```

You can download the [`bean_disease_uganda`](https://github.com/AI-Lab-Makerere/ibean/) dataset like this:

```
adapi.download_dataset(dataset_name='bean_disease_uganda', dest_dir='DESTINATION_DIRECTORY_PATH_HERE')
```

Running this method will download the dataset which has been formatted in one of the standard annotation formats described below.

## Generating synthetic RGB and LiDAR data using Helios 3D Crop Modeling Software
### Getting started
The `Helios Annotation API` assumes that you have cloned [`Helios`](https://github.com/PlantSimulationLab/Helios).

```
# Import Helios Annotation API (i.e. haapi)
from haapi import haapi

# Initialize HeliosDataGenerator class (optionally specify path to Helios)
hdg = haapi.HeliosDataGenerator(path_helios_dir='../../Helios/') 

# User can define a range of values for any parameter from which values are randomly sampled
hdg.canopy_param_ranges['Tomato']['leaf_length'] = [[0.1, 0.3]]
hdg.canopy_param_ranges['Tomato']['plant_count'] = [[1, 5], [1, 5]]
hdg.canopy_param_ranges['Tomato']['canopy_origin'] = [[-1.0, 0.0], [0.5, 1.1], [0.2, 0.7]]

# Generate a user-specified number of simulated rgb/lidar outputs + annotations for a given canopy type in Helios
# Note: simulation_type='lidar' requires a GPU with CUDA installed
hdg.generate_data(n_imgs=10, canopy_type='Tomato', simulation_type='rgb')
```

Running this code block generates 10 xml files which will be used to initialize Helios crop geometries. The geometries are random combinations of the user specified range of values defined above. 

### Optional
```
# Pass in a manual seed for consistency in parameter shuffling
seed=10
hdg.set_seed(seed)
```

## To-Do
- Code to run Helios using a system call to c++ here
- Return data from Helios as a json file in standard annotation formats; i.e. segmentation, object detection, and instance segmentation.

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
