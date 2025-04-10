# Contributing Guidelines

Thank you for choosing to contribute to AgML!

## Contributing Data

If you've found (or already have) a new dataset and you want to contribute the dataset to AgML,
then the instructions below will help you format and the data to the AgML standard.

### Dataset Formats

Currently, we have image classification, object detection, and semantic segmentation datasets available
in AgML. These sources are synthesized to standard annotation formats, namely the following:

- **Image Classification**: Image-To-Label-Number
- **Object Detection**: [COCO JSON](https://cocodataset.org/#format-data)
- **Semantic Segmentation**: Dense Pixel-Wise

#### Image Classification

Image classification datasets are organized in the following directory tree:

```
<dataset name>
    ├── <label 1>
    │   ├── image1.png
    │   ├── image2.png
    │   └── image3.png
    └── <label 2>
        ├── image1.png
        ├── image2.png
        └── image3.png
```

The `AgMLDataLoader` generates a mapping between each of the label names "label 1", "label 2", etc.,
and a numerical value.

#### Object Detection

Object detection datasets are constructed using COCO JSON formatting. For a general overview, see
[https://cocodataset.org/#format-data](https://cocodataset.org/#format-data).
Another good resource is [https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/cd-transform-coco.html](https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/cd-transform-coco.html).
Once you have the images and the bounding box annotations, this involves generating a dictionary with four keys:

1. `images`: A list of dictionaries with the following items:
    - The image file name (without the parent directory!) in `file_name`
    - The ID (a unique number, usually from 1 to num_images) in `id`,
    - The height/width of the image in `height` and `width`, respectively.
2. `annotations`: A list of dictionaries with each dictionary representing a _unique_ bounding box (do not stack multiple bounding boxes into a single dictionary, even if they are for the same image!), and containing:
    - The area of the bounding box in `area`.
    - The bounding box itself in `bbox`. **Note**: The bounding box should have four coordinates. The first two are the x, y of the top-left corner of the bounding box, the other two are its height and width.
    - The class label (numerical) of the image in `category_id`.
    - The **ID** (NOT the filename) of the image it corresponds to in `image_id`.
    - The ID of the bounding box in `id`. For instance, if a unique image has six corresponding bounding boxes, then each of them would be given an `id` from 1-6.
    - `iscrowd` should be set to 0 by default, unless the dataset explicitly comes with `iscrowd` as 1.
    - `ignore` should be 0 by default.
    - `segmentation` only applies for instance segmentation datasets. If converting an instance segmentation dataset to object detection, you can leave the polygonal segmentation as is. Otherwise, put this as an empty list.
3. `category`: A list of dictionaries with each category, where each of these dictionaries contains:
    - The human-readable name of the class (e.g., "strawberry") in `name`.
    - The supercategory of the class, if there are nested classes, in `supercategory`. Otherwise, just leave this as the string `"none"`.
    - The numerical ID of the class in `id`.
4. `info`: A single dictionary with metadata and information about the dataset:
    - `description`: A basic description of the dataset.
    - `url`: The URL from which the dataset was acquired.
    - `version`: The dataset version. Set to `1.0` if unknown.
    - `year`: The year in which the dataset was released.
    - `contributor`: The author(s) of the dataset.
    - `date_created`: The date when the dataset was published. Give an approximate year if unknown.

The dictionary containing this information should be written to a file called `annotations.json`, and the file structure will be:

```
<dataset name>
    ├── annotations.json
    └── images
        ├── image1.png
        ├── image2.png
        └── image3.png
```

#### Semantic Segmentation

Semantic segmentation datasets are constructed using pixel-wise annotation masks. Each image in the dataset has a corresponding
annotation mask. These masks have the following properties:

1. Two-dimensional, so no channel shape. Their complete shape will be `(image_height, image_width)`.
2. Each of the pixels will be a numerical class label or `0` for background.

The directory tree should look like follows:

```
<dataset name>
    ├── annotations
    │   ├── mask1.png
    │   ├── mask2.png
    │   └── mask3.png
    └── images
        ├── image1.png
        ├── image2.png
        └── image3.png
```


## Contributing a Dataset

If you've found a new dataset that *isn't already being used* in AgML and you want to add it, there's a few things you
need to do.

Any preprocessing code being used for the dataset can be kept in `agml/_internal/preprocess.py`, by adding an `elif` statement
to the `preprocess()` method with the dataset name. If there is no preprocessing code, then just put a `pass` statement in the block.

### Some Things to Check

- Make sure each image is in the range of 0-255 in integers as opposed to 0-1 as floats. This will prevent any loss of data that
  could adversely affect training.
- For a semantic segmentation dataset, put the masks in a `png` format as opposed to `jpg` or other.

### Compiling the Dataset

After processing and standardizing the dataset, make sure that the dataset is organized in one of the formats above, and then go to the parent directory
of the directory of the dataset (for example, if the dataset is in `/root/my_new_dataset`, go to `/root`). Then run the following command:

```shell
zip -r my_new_dataset.zip my_new_dataset -x ".*"
```

**If running on MacOS**, use the following command:

```shell
zip -r my_new_dataset.zip my_new_dataset -x ".*"  -x "__MACOSX"
```

### Updating the Source Files

Next, you need to update the `public_datasources.json` and `source_citations.json` files. These two can be found
in the `agml/_assets` folder. You will need to update the `public_datasources.json` file in the following way:

```json
    "my_new_dataset": {
        "classes": {
            "1": "class_1",
            "2": "class_2",
            "3": "class_3"
        },
        "ml_task": "See the table for the different dataset types.",
        "ag_task": "The agricultural task that is associated with the dataset.",
        "location": {
            "continent": "The continent the dataset was collected on.",
            "country": "The country the dataset was collected in."
        },
        "sensor_modality": "Usually rgb, but can include other image modalities.",
        "real_synthetic": "Are the images real or synthetically generated?",
        "platform": "handheld or ground",
        "input_data_format": "See the table for the different dataset types.",
        "annotation_format": "See the table for the different dataset types.",
        "n_images": "The total number of images in the dataset.",
        "docs_url": "Where can the user find the most clear information about the dataset?"
    }
```

**Note**: If the dataset is captured in multiple countries or you don't know where it is from,
then put "worldwide" for both "continent" and "country".

**Note**: If there is no explicit documentation for the dataset, then reach out to the AgML team
regarding what you should put. It is important that we have references to as many datasets as possible,
to allow users to acquire raw data as they desire.


#### `ml_task` and `ag_task`

The ML task can be quickly defined from the following table:

| Dataset Format | `ml_task` | `annotation_format` |
| :------------: | :-------: | :-----------------: |
| Image Classification | `image_classification` | `directory_names` |
| Object Detection | `object_detection` | `coco_json` |
| Semantic Segmentation | `semantic_segmentation` | `image` |


The `ag_task` field is more broadly defined - it should be the main task that the dataset is associated with.
For instance, any of the `*_leaf_disease_classification` datasets, alongside `bean_disease_uganda`, are all 
associated with the `disease_classification` task. The `apple_segmentation_minnesota` dataset has the label
`fruit_segmentation`. See `agml/_assets/public_datasources.json` for various examples of valid `ag_tasks`.
Generally, you should keep this field broad enough that it encompasses the dataset (e.g., instead of a specific
fruit, just put 'fruit' in general), but not as broad as the `ml_task`: it should have an agricultural component.

The `source_citations.json` file should be updated this way:

```json
    "my_new_dataset": {
        "license": "The license being used by the dataset.",
        "citation": "The paper/library to cite for the dataset."
    }
```

If the dataset has no license or has no citation, leave the corresponding lines blank.

### Uploading the Dataset

Once you've readied the dataset, create a new pull request on the AgML repository.
We will then review the changes and review next steps for adding the dataset into AgML's public data storage.



## Quality Checks

When contributing a dataset, you should abide by the following guidelines to ensure compatibility with AgML and ensure that there are no problems for users who are working with the datasets:

- Check that the dataset can be properly downloaded and loaded. It is best to instantiate an `AgMLDataLoader` and call `loader.show_sample()` in order to validate that the images and annotations are in the right format.
- Make sure you have run `python3 scripts/generate_normalization_info.py --dataset <name>` and `python3 scripts/generate_shape_info.py --dataset <name>` to generate the dataset normalization and shape information.
- Make sure that there is an entry in `agml/_internal/preprocess.py` for the dataset. Specifically, you should have a method in the class that makes up the file with the name of the method being the dataset, and the preprocessing code being that of the dataset.
  

## Development Guidelines


### Installing uv
Dependencies and admin actions are done using `uv`. To Install uv follow the guidelines in  https://docs.astral.sh/uv/getting-started/installation/, it is recommended to use the standalone installation.


### Building Project

To sync the dependencies and create a local env that fits the requirements, simply run:


```bash
make install
```

This will install both requirements and necessary development dependencies, such as `dev` and `docs` dependency groups. To build the wheels:


```bash
make build
```

### Running tests

To run all the tests with associated coverage:

```bash
make test
```


### Running scripts

For running scripts or one-offs using the project's installed enviroment
The build the associated wheels simply run:

```
uv build
```

To sync the dependencies simply run:

```
uv sync
```

### Running scripts

For running scripts or one using the project's environment:

```
uv run python <script>
```
