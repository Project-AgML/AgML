# AgData-API
The AgData API aims to provide easy access to centralized and standardized agricultural datasets for deep learning model training.

### Downloading datasets using AgData-API
All of the datasources included in AgData, along with their associated meta-data, are described within [`data_sources.json`](/src/assets/data_sources.json). 

You can also view this information, along with available datasets, via the AgData API.

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
