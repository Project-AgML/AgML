# AgML

AgML is a comprehensive library for agricultural machine learning. Currently, AgML provides
access to a wealth of public agricultural datasets for common agricultural deep learning tasks.  


## Installation

To install the latest release of AgML, run the following command:

```shell
pip install agml
```

## Getting Started

AgML aims to provide seamless access to resources for users of all levels. The core of AgML's public data pipeline is 
[`AgMLDataLoader`](/agml/data/loader.py). Simply running the following line of code:

```python
loader = AgMLDataLoader('<dataset_name_here>')
```

will download the dataset locally from which point it will be automatically loaded from the disk on future runs. For high-level
users who just want the dataset information, accessing the raw metadata is as easy as

```python
dataset = loader.export_contents()
```

On the other hand, users who want to integrate the loader into their existing pipelines can use a number
of methods can use a number of methods to process and export their data, including applying transforms, batching
and splitting the data, and even exporting to PyTorch DataLoaders or TensorFlow Dataset pipelines.

For more detailed information about the API, see [insert documentation link here]().

### Annotation Formats

A core aim of AgML is to provide datasets in a standardized format, enabling the synthesizing of multiple datasets
into a single training pipeline. To this end, we provide annotations in the following formats:

- **Image Classification**: Image-To-Label-Number
- **Object Detection**: [COCO JSON](https://cocodataset.org/#format-data)
- **Semantic Segmentation**: Dense Pixel-Wise

## Optional

We aim to provide additional datasets for different deep learning tasks in the future.

## Vision

AgML aims to be an end-to-end resource encompassing all facets of agricultural machine learning.

```text
Include a nicely-formatted graphic of the slide that Mason
showed in the first lab meeting showing the vision for AgML?
```


<!-- 

INTERNAL NOTE:

As new releases of AgML are published, this README is going to change significantly.
E.g., right now the 'installation' section just discusses `pip install agml`, but that
will evolve to discussing CUDA/dev versions or other features as we continue to add
features to the library. So, this is just the first template as we introduce the first releases.

-->
