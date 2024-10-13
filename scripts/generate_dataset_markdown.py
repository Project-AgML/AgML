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


class DefaultDict(dict):
    __missing__ = lambda self, key: key


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
    def stats(value): # noqa
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


# def generate_example_images(name):
#     """Generates the example images for the given dataset."""
#     agml.backend.set_seed(189)
#     loader = agml.data.AgMLDataLoader(name)
#     return agml.viz.show_sample(loader, no_show = True)

def generate_example_images(name):
    """Generates multiple example images for the given dataset."""
    agml.backend.set_seed(189)
    loader = agml.data.AgMLDataLoader(name, batch_size=4)  # Ensure the batch size is correct
    return agml.viz.show_sample(loader, num_images=4, no_show=True)


def build_examples(name):
    """Builds the example images for the given dataset."""
    sample = generate_example_images(name)
    save_path_local = os.path.join(
        LOCAL_AGML_REPO, 'docs/sample_images', f'{name}_examples.png')
    save_path_remote = os.path.join(
        AGML_REPO, 'docs/sample_images', f'{name}_examples.png')
    cv2.imwrite(save_path_local, sample)
    return f'![Example Images for {name}]({save_path_remote})'


def generate_markdown(name):
    with open(os.path.join(LOCAL_AGML_REPO, 'docs/datasets', f'{name}.md'), 'w') as f:
        f.write(MARKDOWN_TEMPLATE.format(
            name = name,
            table = build_table(get_source_info(name)),
            examples = build_examples(name)))


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
                name = ds.name,
                url = f'https://github.com/Project-AgML/AgML/blob/main/docs/datasets/{ds.name}.md',
                task = to_title(ds.tasks.ml),
                num_images = ds.num_images
            )
            original_readme.insert(end_line, content)
            end_line += 1

    # Rewrite the README.
    with open(os.path.join(LOCAL_AGML_REPO, 'README.md'), 'w') as f:
        f.write('\n'.join(original_readme))


if __name__ == '__main__':
    os.makedirs(os.path.join(LOCAL_AGML_REPO, 'docs/datasets'), exist_ok = True)
    os.makedirs(os.path.join(LOCAL_AGML_REPO, 'docs/sample_images'), exist_ok = True)
    datasets = agml.data.public_data_sources()
    for ds in tqdm(datasets):
        if not os.path.exists(os.path.join(LOCAL_AGML_REPO, 'docs/datasets', f'{ds.name}.md')):
            generate_markdown(ds.name)
            pass

    # Update the README with any new datasets.
    update_readme(datasets)





