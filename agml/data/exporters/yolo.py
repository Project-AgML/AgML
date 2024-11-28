import os
import shutil

import yaml

import agml
from agml.utils.general import flatten
from agml.utils.logging import log


def convert_annotations_to_yolo_list(annotation_dict, image_shape, annotation_remapper):
    yolo_list = []
    height, width = image_shape
    for annotation in annotation_dict:
        x, y, w, h = annotation["bbox"]
        x_center = x + w / 2
        y_center = y + h / 2

        # scale XYWH to [0, 1]
        yolo_list.append(
            [
                annotation_remapper[annotation["category_id"]],
                x_center / width,
                y_center / height,
                w / width,
                h / width,
            ]
        )
    return yolo_list


def export_yolo(dataset, yolo_path=None):
    """Exports an object detection dataset to YOLO format, ready-to-use for YOLO training.

    This method will export an AgML dataset to the YOLO format, given its name and the
    desired output location. This is so that the data is prepared within the YOLO format -
    however, this does not translate to *integration* with the YOLO training pipeline.
    Instead, this will simply enable you to add the dataset path to the YOLO training
    configuration (and all other preprocessing steps are abstracted away in that same
    pipeline, not requiring AgML).

    Note that you can also use this function to export a custom dataset - in this case,
    rather than passing in the name of a dataset simply pass in an AgMLDataLoader with
    the dataset (and this will work for multi-dataset datasets, as well as dataset splits).

    If you provide a path to a YOLO implementation, this function will automatically write
    the corresponding files (including the `dataset.yaml` file). However, if an empty path,
    no path, or a non-YOLO path is provided, the function will simply create a new directory
    and write the files within a `metadata` directory there.

    Additionally, if the data is already split, then `train.txt`, `val.txt`, and `test.txt`
    files will be written to the data directory, which contain the split paths.

    Parameters
    ----------
    dataset : {str, AgMLDataLoader}
        The name of the dataset to export to YOLO format, or an AgMLDataLoader object.
    yolo_path : str
        The path to the directory where the YOLO-formatted dataset will be saved.

    """
    # if a name is provided, load the dataset
    if isinstance(dataset, str):
        loader = agml.data.AgMLDataLoader(dataset)
    else:
        loader = dataset

    if yolo_path is None:
        yolo_path = os.path.join(os.getcwd(), f"{loader.name}_yolo_export")
    if not os.path.exists(os.path.join(yolo_path, "data")):
        log("YOLO Export Tool did not receive a valid YOLO path. Creating a new directory for the export.")

    yolo_data_path = os.path.join(yolo_path, "datasets", loader.name)
    output_image_dir = os.path.join(yolo_data_path, "images")
    output_annotation_dir = os.path.join(yolo_data_path, "labels")
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_annotation_dir, exist_ok=True)

    # create the dataset YAML
    yaml_dump = {"path": yolo_data_path, "names": loader.num_to_class}

    # AgML indexes object detection datasets from 1 -> N, so reindex from 0 -> N-1
    class_mapper = {i: i - 1 for i in yaml_dump["names"]}
    yaml_dump["names"] = {i - 1: name for i, name in yaml_dump["names"].items()}

    # get the height/width for all the images for normalization purposes
    if loader.IS_MULTI_DATASET:
        image_info = flatten(
            [sub_loader._builder._default_coco_annotations["images"] for sub_loader in loader._loaders]
        )  # noqa
    else:
        image_info = loader._builder._default_coco_annotations["images"]
    image_info = {image["file_name"]: (image["height"], image["width"]) for image in image_info}

    # compatibility for content format for multi-dataset loaders
    if not loader.IS_MULTI_DATASET:
        loader_contents = {loader.name: loader.export_contents(export_format=None)}
    else:
        loader_contents = loader.export_contents(export_format=None)

    # check for data splits and get a list of images in each split
    all_split_images = {}
    if loader._is_split_generated():
        for split_name in ["train", "val", "test"]:
            split_content = getattr(loader, f"_{split_name}_content")
            if not loader.IS_MULTI_DATASET:
                split_content = {loader.name: split_content}

            # skip empty splits by checking the contents
            if all(i is None for i in split_content.values()):
                continue

            split_images = flatten(
                [
                    [f"{curr_name}_{os.path.basename(image)}" for image in curr_split]
                    for curr_name, curr_split in split_content.items()
                ]
            )
            all_split_images[split_name] = split_images
    else:
        # if no split generated, put all in `train`
        all_split_images["train"] = flatten(
            [
                [f"{loader.name}_{os.path.basename(image)}" for image in curr_split]
                for curr_split in loader_contents.values()
            ]
        )

    # write the text files containing the split contents
    for split_name, split_images in all_split_images.items():
        split_images = [os.path.join(output_image_dir, image) for image in split_images]
        with open(os.path.join(yolo_data_path, f"{split_name}.txt"), "w") as f:
            f.write("\n".join(split_images))

        # update the YAML file with the corresponding locs of train/val/test
        yaml_dump[split_name] = f"{split_name}.txt"

    for loader_name, loader_content in loader_contents.items():
        for image, annotation_set in loader_content.items():
            # convert from the default AgML format (COCO: [x, y, w, h])
            # to YOLO format: (normalized [x_center, y_center, w, h])
            image_shape = image_info[os.path.basename(image)]
            new_annotation_set = convert_annotations_to_yolo_list(annotation_set, image_shape, class_mapper)
            text_content = "\n".join([" ".join(map(str, annotation)) for annotation in new_annotation_set])

            # the name of the new image is {dataset_name}_{old_name}.{ext}
            image_name = f"{loader_name}_{os.path.basename(image)}"
            txt_name = os.path.splitext(image_name)[0] + ".txt"

            # save the image and the text file
            shutil.copy(image, os.path.join(output_image_dir, image_name))
            with open(os.path.join(output_annotation_dir, txt_name), "w") as f:
                f.write(text_content)

    # write the dataset.yaml file
    os.makedirs(os.path.join(yolo_path, "data"), exist_ok=True)
    with open(os.path.join(yolo_path, "data", "dataset.yaml"), "w") as f:
        yaml.safe_dump(yaml_dump, f)

    # log information about the dataset location
    information = """
    Dataset successfully exported to YOLO format. 

    You can find the dataset at the following location: {0}
    The dataset metadata is stored at {1}
    """.format(yolo_data_path, os.path.join(yolo_path, "data", "dataset.yaml"))
    log(information)

    return {
        "dataset_path": yolo_data_path,
        "metadata_path": os.path.join(yolo_path, "data", "dataset.yaml"),
    }
