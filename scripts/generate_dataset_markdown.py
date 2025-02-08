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

"""
Generates a markdown file with information for a given dataset.
"""

import os

import cv2
from tqdm import tqdm

import agml
from agml.utils.data import load_public_sources
from agml.utils.io import recursive_dirname
import random
import numpy as np


class DefaultDict(dict):
    __missing__ = lambda self, key: key


import matplotlib.pyplot as plt

# Configuration Variables.
NUM_EXAMPLES = 4
AGML_REPO = "https://github.com/Project-AgML/AgML/blob/main"
LOCAL_AGML_REPO = os.path.join(recursive_dirname(__file__, 2))

# Format of the markdown file.
MARKDOWN_TEMPLATE = """
# `{name}`

## Dataset Metadata

{table}

## Examples

{examples}"""

# Unused keys.
UNUSED_KEYS = ['external_image_sources']

# Substitutions for metadata category names and values.
substitutions = DefaultDict({
    'Ml Task': 'Machine Learning Task',
    'Ag Task': 'Agricultural Task',
    'Real Synthetic': 'Real or Synthetic',
    'N Images': 'Number of Images',
    'Docs Url': 'Documentation',
    'Usa': 'United States',
    'rgb': 'RGB',
    'lidar': 'LiDAR',
    'jpg': 'JPG',
    'png': 'PNG',
    'jpeg': 'JPEG'
})


class TableFormat(object):
    """Constructs a proper table format for the given key."""

    @staticmethod
    def handle(key, value):
        method = getattr(TableFormat, key, None)
        if method is not None:
            return method(value)
        if isinstance(value, list):
            value = ", ".join(value)
        if value == '': value = "None"
        return f'| **{substitutions[to_title(key)]}** | {substitutions[value]} |\n'

    @staticmethod
    def stats(value):  # noqa
        return f"| **Stats/Mean** | [{', '.join(str(round(i, 3)) for i in value['mean'])}] |\n" \
               f"| **Stats/Standard Deviation** | [{', '.join(str(round(i, 3)) for i in value['std'])}] |\n"

    @staticmethod
    def classes(value):
        print(value)
        return "| **Classes** | {} |\n".format(', '.join(value.values()))

    @staticmethod
    def location(value):
        if value['continent'] == 'worldwide':
            return "| **Location** | Worldwide |\n"
        return "| **Location** | {}, {} |\n".format(
            substitutions[to_title(value['country'])], to_title(value['continent']))


def get_source_info(source):
    """Returns the information for a given data source."""
    return load_public_sources()[source]


def to_title(name):
    return ' '.join(i.title() for i in name.split('_'))


def build_table(json):
    """Builds a markdown table from the given json."""
    table = '| Metadata | Value |\n| --- | --- |\n'
    for key, value in json.items():
        if key in UNUSED_KEYS:
            continue
        table += TableFormat.handle(key, value)
    return table


def show_sample(loader, image_only=False, num_images=1, **kwargs):
    """A simplified convenience method that visualizes multiple samples from a loader.

    This method works for all types of annotations and visualizes multiple
    images based on the specified `num_images` parameter.

    Parameters
    ----------
    loader : AgMLDataLoader
        An AgMLDataLoader of any annotation type.
    image_only : bool
        Whether to only display the images.
    num_images : int
        The number of images to display (default is 1).

    Returns
    -------
    The matplotlib figure showing the requested number of images.
    """
    if kwargs.get('sample', None) is not None:
        sample = kwargs['sample']
    else:
        sample = loader[0]

    if image_only:
        return agml.viz.show_images(sample[0])

    if loader.task == 'object_detection':
        # Collect 4 images (or as many as available if fewer)
        samples = [loader[i] for i in range(min(len(loader), 4))]
        return [agml.viz.show_image_and_boxes(sample, info=loader.info, no_show=kwargs.get('no_show', False)) for sample
                in samples]

    elif loader.task == 'semantic_segmentation':
        samples = [loader[i] for i in range(min(len(loader), num_images))]
        return [agml.viz.show_image_and_overlaid_mask(sample, no_show=kwargs.get('no_show', False)) for sample in
                samples]

    elif loader.task == 'image_classification':
        # Fetch all available classes and initialize an empty list for samples.
        classes = loader.classes
        num_classes = len(classes)
        samples = []

        # Adjust the dictionary to store samples per class, starting from class 1.
        class_to_sample = {cls: None for cls in range(num_classes)}

        # Ensure one image from each class first (without duplication).
        for i in range(len(loader)):
            sample = loader[i]
            label = sample[1]

            # Map label to its class index (if necessary)
            if isinstance(label, (list, np.ndarray)):  # Handle one-hot encoding case
                label = np.argmax(label)  # Adjust indexing to start from 1

            if isinstance(label, dict):
                label = label.get('label', None)

            if label in class_to_sample and class_to_sample[label] is None:
                class_to_sample[label] = sample

            # Stop once we have at least one image per class
            if all(v is not None for v in class_to_sample.values()):
                break

        # Collect samples ensuring uniqueness per class until we hit num_classes.
        samples = [class_to_sample[cls] for cls in range(num_classes) if class_to_sample[cls] is not None]

        # If more images are required, duplicate randomly from the collected samples.
        if num_images > num_classes:
            additional_samples = random.choices(samples, k=num_images - num_classes)
            samples.extend(additional_samples)

        # If fewer images are requested, truncate the sample list.
        if num_images <= num_classes:
            samples = samples[:num_images]
        return agml.viz.show_images_and_labels(
            samples, info=loader.info, no_show=kwargs.get('no_show', False))


def generate_example_images(name):
    """Generates multiple example images for the given dataset."""
    agml.backend.set_seed(189)
    loader = agml.data.AgMLDataLoader(name)  # Ensure the batch size is correct
    return agml.viz.show_sample(loader, num_images=max(4, len(loader.classes)), no_show=True)


def build_examples(name):
    """Builds the example images for the given dataset and combines them into one image."""
    samples = generate_example_images(name)
    if isinstance(samples, list):
        images = []
        for idx, sample in enumerate(samples):
            # Extract the image part if the sample contains other data (like bounding boxes)
            if isinstance(sample, tuple) and isinstance(sample[0], np.ndarray):
                image = sample[0]  # The first part of the tuple is the image
            elif isinstance(sample, np.ndarray):
                image = sample  # If it's directly an image
            else:
                print(f"Error: Sample {idx} is not a valid image format. Type: {type(sample)}")
                continue

            images.append(image)

        # Limit to 4 images for the grid (2x2 layout)
        images = images[:4]

        # Create a 2x2 grid of subplots for the images
        fig, axes = plt.subplots(2, 2)

        for i, ax in enumerate(axes.flat):
            if i < len(images):
                ax.imshow(images[i])
                ax.axis('off')
            else:
                ax.axis('off')
        # Save the combined image
        save_path_local = os.path.join(LOCAL_AGML_REPO, 'docs/sample_images', f'{name}_examples.png')
        plt.savefig(save_path_local, bbox_inches='tight')
        plt.close()
        print(f"Combined image saved to {save_path_local}")

        save_path_remote = os.path.join(AGML_REPO, 'docs/sample_images', f'{name}_examples.png')
        return f'![Example Images for {name}]({save_path_remote})'

    else:
        save_path_local = os.path.join(
            LOCAL_AGML_REPO, 'docs/sample_images', f'{name}_examples.png')
        save_path_remote = os.path.join(
            AGML_REPO, 'docs/sample_images', f'{name}_examples.png')
        cv2.imwrite(save_path_local, samples)
        return f'![Example Images for {name}]({save_path_remote})'


def generate_markdown(name):
    with open(os.path.join(LOCAL_AGML_REPO, 'docs/datasets', f'{name}.md'), 'w') as f:
        f.write(MARKDOWN_TEMPLATE.format(
            name=name,
            table=build_table(get_source_info(name)),
            examples=build_examples(name)))


def update_readme(datasets):
    # Find the part of the README where datasets are listed out.
    with open(os.path.join(LOCAL_AGML_REPO, 'README.md'), 'r') as f:
        original_readme = [i[:-1] for i in f.readlines()]
    with open(os.path.join(LOCAL_AGML_REPO, 'README.md'), 'r') as f:
        original_readme_full = f.read()
    end_line = original_readme.index('## Usage Information') - 1

    # Update the README.
    for ds in tqdm(datasets):
        if ds.name not in original_readme_full:
            content = '[{name}]({url}) | {task} | {num_images} |'.format(
                name=ds.name,
                url=f'https://github.com/Project-AgML/AgML/blob/main/docs/datasets/{ds.name}.md',
                task=to_title(ds.tasks.ml),
                num_images=ds.num_images
            )
            original_readme.insert(end_line, content)
            end_line += 1

    # Rewrite the README.
    with open(os.path.join(LOCAL_AGML_REPO, 'README.md'), 'w') as f:
        f.write('\n'.join(original_readme))


if __name__ == '__main__':
    os.makedirs(os.path.join(LOCAL_AGML_REPO, 'docs/datasets'), exist_ok=True)
    os.makedirs(os.path.join(LOCAL_AGML_REPO, 'docs/sample_images'), exist_ok=True)
    datasets = agml.data.public_data_sources()
    for ds in tqdm(datasets):
        if not os.path.exists(os.path.join(LOCAL_AGML_REPO, 'docs/datasets', f'{ds.name}.md')):
            generate_markdown(ds.name)
            pass

    # Update the README with any new datasets.
    update_readme(datasets)




