from agml import AgMLDataLoader
from agml import public_data_sources
from agml.backend import set_seed

from agml.viz import format_image
from agml.backend import set_backend

set_seed(0)

# print(public_data_sources(ml_task = 'image_classification', annotation_format = 'directory_names'))

crop_greece_loader = AgMLDataLoader('crop_weeds_greece')

####### SPLIT TESTS #######
# res = crop_greece_loader.split(train = 254, val = 127, test = 127)
# print(len(crop_greece_loader))
# print(res)
# print([len(i) if i is not None else None for i in res])

####### BATCH TESTS #######
# crop_greece_loader.batch(8)
# print(len(crop_greece_loader))
# crop_greece_loader.batch(None)

###### GETITEM TESTS ######
# print(crop_greece_loader[0])
# crop_greece_loader.batch(8)
# print(crop_greece_loader[0])

import torchvision.transforms as T
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip
import torch.utils.data

ds = crop_greece_loader.tensorflow(
    preprocessing = Sequential([RandomFlip()])
)

import matplotlib.pyplot as plt

for i in ds:
    plt.imshow(format_image(i[0]))
    plt.show()
    break

