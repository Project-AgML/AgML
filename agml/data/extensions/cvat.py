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

import json
import os
import shutil


def restructure_cvat_annotations(image_dir, cvat_dir, dataset_name, output_dir=None):
    """
    Restructure CVAT annotations and images into the AgML dataset format.

    Parameters
    ----------
    image_dir : str
        Path to the directory containing the images
    cvat_dir : str
        Path to the directory containing the CVAT annotations
    dataset_name : str
        Name of the new dataset
    output_dir : str
        Directory where the dataset will be saved. By default, saves to `~/.agml/datasets`

    Returns
    -------
    The path to the dataset
    """
    image_dir = os.path.expanduser(image_dir)
    cvat_dir = os.path.expanduser(cvat_dir)

    # Determine the dataset path
    if output_dir is None:
        import agml.backend  # Ensure that agml is imported

        dataset_path = os.path.join(agml.backend.data_save_path(), dataset_name)
    else:
        dataset_path = os.path.join(output_dir, dataset_name)

    images_output_dir = os.path.join(dataset_path, "images")
    os.makedirs(images_output_dir, exist_ok=True)

    # Copy images to the new dataset directory
    for image_file in os.listdir(image_dir):
        src_image_path = os.path.join(image_dir, image_file)
        dst_image_path = os.path.join(images_output_dir, image_file)
        shutil.copy2(src_image_path, dst_image_path)

    # Load the CVAT annotations
    cvat_annotations_path = os.path.join(cvat_dir, "instances_default.json")
    with open(cvat_annotations_path, "r") as f:
        coco = json.load(f)

    # Build a mapping from base image names to actual filenames (with extensions)
    image_files = os.listdir(image_dir)
    base_name_to_file_name = {}
    for file_name in image_files:
        base_name, ext = os.path.splitext(file_name)
        base_name_to_file_name[base_name] = file_name

    # Update the 'file_name' field in each image entry to match the actual filenames
    for image in coco.get("images", []):
        original_file_name = image["file_name"]
        base_name, _ = os.path.splitext(original_file_name)
        actual_file_name = base_name_to_file_name.get(base_name)

        if actual_file_name:
            image["file_name"] = actual_file_name
        else:
            print(f"Warning: No matching image file found for '{original_file_name}'.")
            # Optionally, handle this case as needed (e.g., remove the image from the list)

    # Save the updated annotations to 'annotations.json' in the new dataset directory
    annotations_output_path = os.path.join(dataset_path, "annotations.json")
    with open(annotations_output_path, "w") as f:
        json.dump(coco, f)

    print(f"Dataset '{dataset_name}' has been created successfully at '{dataset_path}'.")
