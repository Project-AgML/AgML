## Overview

iNatAg-mini is a large-scale dataset derived from the iNaturalist dataset, designed for species classification and crop/weed classification in agricultural and ecological applications. 

It consists of 2,959 species with a breakdown of 1,986 crop species and 973 weed species.The dataset contains a total of 560,844 images, making it one of the largest and most diverse datasets available for plant species identification and classification.

## List iNatAg-mini datasets

```python
from pprint import pprint
import agml

print(agml.data.public_data_sources(parent_dataset='iNatAg-mini'))
```

## Loading iNatAg-mini dataset

You can start off by using the `AgMLDataLoader` to download and load an iNatAg-mini dataset into a container:

```python
loader = agml.data.AgMLDataLoader('iNatAg-mini/sorghum_bicolor')
loader.info.summary()
```

## Examples
![Example Images for iNatAg-mini](https://github.com/Project-AgML/AgML/blob/main/docs/sample_images/iNatAg_sample_images.png)