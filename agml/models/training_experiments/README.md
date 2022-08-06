# AgML Benchmarking Training Scripts

This directory is not a package exported with AgML, but instead stores the training scripts used
to generate benchmarks and pretrained models on different public datasets.

The main files of interest here are:

- `*_lightning.py`: Scripts for training different types of models (image classification, semantic
   segmentation, object detection, etc.) on a single dataset, using a single GPU (or none).
- `*_distributed.py`: Full-scale training scripts for training a model type on multiple datasets,
   and utilizing multiple GPUs to do so (distributed training).
- `*_evaluation`: Scripts for evaluation of models on a certain metric (as described in the name). 

**Note**: These scripts all train PyTorch models, TensorFlow support will arrive in the future.


