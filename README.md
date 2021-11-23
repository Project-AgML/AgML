<p align="center">
<img src="/figures/agml-logo.png" alt="agml framework" width="400" height="400">
</p>

## Overview
AgML is a comprehensive library for agricultural machine learning. Currently, AgML provides
access to a wealth of public agricultural datasets for common agricultural deep learning tasks. In the future, AgML will provide ag-specific ML functionality related to data, training, and evaluation. Here's a conceptual diagram of the overall framework. 

<p align="center">
<img src="/figures/agml-framework.png" alt="agml framework" width="350" height="291">
</p>

AgML supports both the [TensorFlow](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/) machine learning frameworks.

## Installation

To install the latest release of AgML, run the following command:

```shell
pip install agml
```

## Getting Started

### Using Public Agricultural Data

AgML aims to provide easy access to a range of existing public agricultural datasets The core of AgML's public data pipeline is 
[`AgMLDataLoader`](/agml/data/loader.py). Simply running the following line of code:

```python
loader = AgMLDataLoader('<dataset_name_here>')
```

will download the dataset locally from which point it will be automatically loaded from the disk on future runs. 
From this point, the data within the loader can be split into train/val/test sets, batched, have augmentations and transforms
applied, and be converted into a training-ready dataset (including batching, tensor conversion, and image formatting).

To see the various ways in which you can use AgML datasets in your training pipelines, check out 
the [example notebook](/examples/AgML-Data.ipynb).

## Annotation Formats

A core aim of AgML is to provide datasets in a standardized format, enabling the synthesizing of multiple datasets
into a single training pipeline. To this end, we provide annotations in the following formats:

- **Image Classification**: Image-To-Label-Number
- **Object Detection**: [COCO JSON](https://cocodataset.org/#format-data)
- **Semantic Segmentation**: Dense Pixel-Wise

## Contributions

We welcome contributions! If you would like to contribue a new feature, fix an issue that you've noticed, or even just mention
a bug or feature that you would like to see implemented, please don't hesitate to use the *Issues* tab to bring it to our attention.
