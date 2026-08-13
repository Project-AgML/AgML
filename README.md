<p align="center">
<img src="docs/assets/agml-logo.png" alt="agml logo" width="400" height="400">
</p>

----

### 👨🏿‍💻👩🏽‍💻🌈🪴 Want to join the [AI Institute for Food Systems team](https://aifs.ucdavis.edu/) and help lead AgML development? 🪴🌈👩🏼‍💻👨🏻‍💻

We're looking to hire a postdoc with both Python library development and ML experience. Send your resume and GitHub profile link to [jmearles@ucdavis.edu](mailto:jmearles@ucdavis.edu)!

----

## Overview
AgML is a comprehensive library for agricultural machine learning. Currently, AgML provides
access to a wealth of public agricultural datasets for common agricultural deep learning tasks. 

AgML supports both the [TensorFlow](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/) machine learning frameworks.

To browse our latest datasets, view our full set of documentation, and compare model performance on your crops / tasks of interest, please visit [project-agml.github.io](https://project-agml.github.io/).

## Installation

To install the latest release of AgML, run the following command:

```shell
pip install agml
```

## Quick Start

AgML datasets are hosted on the [Hugging Face Hub](https://huggingface.co/Project-AgML) under the `Project-AgML`
organization. You can start off by using the `HuggingFaceDataLoader` to download and load a dataset directly into a
native Hugging Face `DatasetDict`:

```python
from agml.data import HuggingFaceDataLoader

# Load a dataset from the Hub
loader = HuggingFaceDataLoader("Project-AgML/apple_flower_segmentation")

# Load a specific config/subset (e.g. an augmented variant)
loader = HuggingFaceDataLoader("Project-AgML/apple_flower_segmentation", config="augmented")
```

`HuggingFaceDataLoader` automatically casts image-like columns (`image`, `mask`, and image-typed `label` columns) to
the Hugging Face `Image` type for decoded pixel access.

You can split the dataset into train/val/test sets, with optional stratification across one or more columns:

```python
dataset = loader.split(val_size=0.1, test_size=0.1, stratify_cols="label")
# Returns a DatasetDict with 'train', 'val', and 'test' splits

# Access the underlying DatasetDict at any time
dataset = loader.dataset
```

**For any preprocessing, inference, or training beyond loading and splitting, use the
[`datasets`](https://huggingface.co/docs/datasets) and [`transformers`](https://huggingface.co/docs/transformers)
libraries directly.** Since `loader.dataset` is a native Hugging Face `DatasetDict`, it works out of the box with
`datasets`' `map`, `filter`, and `with_transform` methods for preprocessing, and with `transformers`' `Trainer`,
`Pipeline`, and model classes for training and inference — there's no separate AgML-specific processing API to learn.

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=dataset["train"].features["label"].num_classes,
)

def preprocess(batch):
    batch["pixel_values"] = processor(batch["image"], return_tensors="pt")["pixel_values"]
    return batch

dataset = dataset.with_transform(preprocess)

trainer = Trainer(model=model, train_dataset=dataset["train"], eval_dataset=dataset["val"])
trainer.train()
```

## Public Datasets

AgML contains a wide variety of public datasets from various locations across the world:

![AgML Dataset World Map](/docs/assets/agml_dataset_world_map.png)

Use the [Dataset Search](https://project-agml.github.io/datasets) experience for filtering and previews. For programmatic filtering,
`agml.data.public_data_sources(...)` supports task and modality filters.

## iNatAg and iNatAg-mini

AgML provides an API with direct access to iNatAg (and iNatAg-mini), one of the world's largest collections of agricultural images dedicated for the task of image classification. Collectively, this dataset contains over 4 million images along with detailed species classificaations and enables access to a variety of large-scale agricultural machine learning tasks. You can instantiate the iNatAg (or iNatAg-mini, a smaller variant of iNatAg for smaller-scale applications) dataset as follows:

```python
# To select a collection of scientific family names.
loader = agml.data.AgMLDataLoader.from_parent("iNatAg", filters={"family_name": ["...", "..."]})

# To select common names.
loader = agml.data.AgMLDataLoader.from_parent("iNatAg", filters={"common_name": "..."})
```

## Usage Information

### Using Public Agricultural Data

AgML aims to provide easy access to a range of existing public agricultural datasets The core of AgML's public data pipeline is
[`AgMLDataLoader`](https://github.com/Project-AgML/AgML/blob/main/agml/data/loader.py). You can use the `AgMLDataLoader` or `agml.data.download_public_dataset()` to download
the dataset locally from which point it will be automatically loaded from the disk on future runs.
From this point, the data within the loader can be split into train/val/test sets, batched, have augmentations and transforms
applied, and be converted into a training-ready dataset (including batching, tensor conversion, and image formatting).

## Annotation Formats

A core aim of AgML is to provide datasets in a standardized format, enabling the synthesizing of multiple datasets
into a single training pipeline. Datasets on the Hugging Face Hub encode annotations as columns on the underlying
`Dataset`/`DatasetDict`:

- **Image Classification**: a `label` column of type `ClassLabel`.
- **Object Detection**: an `objects` column, holding COCO-style bounding boxes (un-normlized [x_min, y_min, width, height]) and corresponding category IDs as a ClassLabel per image.
- **Semantic Segmentation**: a `mask` column, a single-channel (`L`-mode) image the same size as the corresponding image.

## Contributions

We welcome contributions! If you would like to contribute a new feature, fix an issue that you've noticed, or even just mention
a bug or feature that you would like to see implemented, please don't hesitate to use the *Issues* tab to bring it to our attention.

See the [contributing guidelines]([https://project-agml.github.io/docs/development) for more information, or the
[guide to contributing leaderboard results](https://project-agml.github.io/docs/contributing-results) if you have benchmark results to add.

## Funding
This project is partly funded by the [National AI Institute for Food Systems](https://aifs.ucdavis.edu).
