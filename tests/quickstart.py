import agml
import albumentations as A

import tensorflow as tf
import torch

loader = agml.data.AgMLDataLoader('apple_flower_segmentation')

# Batch the dataset into collections of 8 pieces of data:
loader.batch(8)

# Shuffle the data:
loader.shuffle()

# Apply transforms to the input images and output annotation masks:
loader.mask_to_channel_basis()
"""loader.transform(
    transform = A.RandomContrast(),
    dual_transform = A.Compose([A.RandomRotate90()])
)"""

# Split the data into train/val/test sets.
loader.split(train = 0.8, val = 0.1, test = 0.1)

# Disable processing and batching for the test data:
test_ds = loader.test_data
test_ds.batch(None)
"""test_ds.reset_prepreprocessing()"""

# Visualize the image and mask side-by-side:
"""agml.viz.visualize_image_and_mask(test_ds[0])"""

# Visualize the mask overlaid onto the image:
"""agml.viz.visualize_overlaid_masks(test_ds[0])"""

# Export the loader as a `tf.data.Dataset`:
train_ds = loader.train_data.export_tensorflow()

# Convert to PyTorch tensors without exporting.
train_ds = loader.train_data
train_ds.as_torch_dataset()
print('finished')