import os
import shutil
import urllib.parse
from collections import Counter
import math
from typing import List, Union

try:
    from datasets import load_dataset, DatasetDict, Image, Sequence, Value, ClassLabel
except ImportError:
    raise ImportError(
        "The `datasets` library is required to use the HuggingFaceDataLoader. "
        "Please install it using `pip install datasets`."
    )

from agml.framework import AgMLSerializable
from agml.backend.config import data_save_path

class HuggingFaceDataLoader(AgMLSerializable):
    """A data loader designed for loading datasets directly into Hugging Face formats.

    This loader retrieves datasets from S3 (or local paths) and structures them
    into Hugging Face `DatasetDict` objects natively compatible with `transformers`
    or `diffusers` pipelines. It supports custom multi-column stratification.

    Parameters
    ----------
    location : str
        The location of the dataset. This can be an S3 URI (s3://...), a public URL,
        or a local directory path containing the HuggingFace-formatted dataset.
    task : str
        The computer vision task. Must be one of 'classification', 'detection', or 'segmentation'.
    local_dir : str, optional
        The local directory to download the data to. Defaults to `~/.agml/datasets/<dataset_name>`.
    """

    serializable = frozenset(("location", "task", "local_dir"))

    def __init__(self, location: str, task: str, local_dir: str = None, **kwargs):
        if task not in ['classification', 'detection', 'segmentation']:
            raise ValueError("Task must be 'classification', 'detection', or 'segmentation'.")
        
        self.location = location
        self.task = task
        
        if local_dir is None:
            # Extract a pseudo-name from the location
            base_name = os.path.basename(location.rstrip('/'))
            if not base_name:
                base_name = "hf_dataset"
            self.local_dir = os.path.join(data_save_path(), base_name)
        else:
            self.local_dir = local_dir

        self._hf_dataset = None
        self._setup_loader()

    def _setup_loader(self):
        """Downloads the dataset and loads it into a Hugging Face Dataset."""
        # 1. Handle downloading from S3 if needed
        self._download_from_s3()

        # 2. Load into Hugging Face Dataset object
        # Assume the dataset is provided in imagefolder format.
        try:
            self._hf_dataset = load_dataset('imagefolder', data_dir=self.local_dir)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load dataset as 'imagefolder' from {self.local_dir}: {e}"
            )

        # 3. Cast features based on the task
        self._cast_features()

    def _download_from_s3(self):
        """Downloads files from an S3 URI to `self.local_dir`."""
        if not self.location.startswith('s3://'):
            if os.path.isdir(self.location):
                self.local_dir = self.location
            return # Assume it's a valid path or handle HTTP elsewhere
            
        if os.path.exists(self.local_dir) and len(os.listdir(self.local_dir)) > 0:
            return  # Already downloaded

        try:
            import boto3
        except ImportError:
            raise ImportError("`boto3` is required to download S3 datasets. Run `pip install boto3`.")

        print(f"Downloading dataset from {self.location} to {self.local_dir}...")
        os.makedirs(self.local_dir, exist_ok=True)
        
        parsed_url = urllib.parse.urlparse(self.location)
        bucket_name = parsed_url.netloc
        prefix = parsed_url.path.lstrip('/')

        s3 = boto3.client('s3')
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

        for page in pages:
            if 'Contents' not in page:
                continue
            for obj in page['Contents']:
                key = obj['Key']
                if key.endswith('/'):
                    continue
                # Compute relative path
                rel_path = os.path.relpath(key, prefix)
                local_file_path = os.path.join(self.local_dir, rel_path)
                
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                s3.download_file(bucket_name, key, local_file_path)

    def _cast_features(self):
        """Casts Hugging Face Dataset features according to the specified CV task."""
        # Convert DatasetDict to a processable format to cast columns accurately
        for split_name in self._hf_dataset.keys():
            ds = self._hf_dataset[split_name]
            features = ds.features
            
            # Ensure "image" exists and is of HF Image type
            if "image" in features and not isinstance(features["image"], Image):
                self._hf_dataset[split_name] = ds.cast_column("image", Image())

            if self.task == "segmentation":
                # Segmentation masks should also be casted to the Image type for automated decoding
                label_col = "mask" if "mask" in features else "label"
                if label_col in features and not isinstance(features[label_col], Image):
                    self._hf_dataset[split_name] = ds.cast_column(label_col, Image())

    def split(self, 
              val_size: float = None, 
              test_size: float = None, 
              stratify_cols: Union[str, List[str]] = None, 
              seed: int = 42) -> DatasetDict:
        """
        Splits the dataset into train, val, and test splits.
        Supports stratified splitting across any number of columns.

        Parameters
        ----------
        val_size : float
            Proportion of the dataset to include in the validation split.
        test_size : float
            Proportion of the dataset to include in the test split.
        stratify_cols : str or list of str
            Column(s) to stratify the split on.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        DatasetDict
            A Hugging Face DatasetDict containing 'train', 'val', and 'test' splits.
        """
        # If the dataset is already split (e.g., contains train/test/val),
        # and we operate on 'train' as the base:
        base_ds = self._hf_dataset['train'] if 'train' in self._hf_dataset else self._hf_dataset

        # If both sizes are None, the caller is requesting no split —
        # return the dataset as-is (or wrapped as a DatasetDict with 'train').
        if val_size is None and test_size is None:
            if isinstance(self._hf_dataset, DatasetDict):
                return self._hf_dataset
            # Wrap a single Dataset into a DatasetDict under 'train'
            dataset_dict = DatasetDict({"train": base_ds})
            self._hf_dataset = dataset_dict
            return dataset_dict

        # Treat None for an individual split as 0.0 (i.e., not requested)
        if val_size is None:
            val_size = 0.0
        if test_size is None:
            test_size = 0.0

        if stratify_cols is None:
            stratify_by_column = None
            ds_to_split = base_ds
        else:
            if isinstance(stratify_cols, str):
                stratify_cols = [stratify_cols]
            
            # Create composite key for stratification
            def add_stratify_key(example):
                key = "_".join(str(example[col]) for col in stratify_cols)
                example["_stratify_key"] = key
                return example
                
            ds_to_split = base_ds.map(add_stratify_key)
            stratify_by_column = "_stratify_key"

            # Inspect counts per composite key to ensure stratification is possible
            try:
                values = ds_to_split[stratify_by_column]
                counts = Counter(values)
                # Basic requirement: at least 2 samples per class for initial stratified split
                too_small = {k: v for k, v in counts.items() if v < 2}
                if too_small:
                    raise ValueError(
                        "Stratified splitting requires at least 2 samples per class. "
                        f"The following stratify groups have fewer than 2 samples: {too_small}"
                    )

                # Additional requirement when doing a second split (val vs test):
                # ensure the temporary partition (val+test) will likely contain
                # at least 2 samples per class. Conservatively require
                # `count >= ceil(2 / eval_size)` for each class when `test_size > 0`.
                eval_size_check = val_size + test_size
                if test_size > 0 and eval_size_check > 0:
                    min_required_in_temp = math.ceil(2.0 / eval_size_check)
                    too_small_temp = {k: v for k, v in counts.items() if v < min_required_in_temp}
                    if too_small_temp:
                        raise ValueError(
                            "Stratified splitting into train/val/test is not possible with the current "
                            f"class counts and requested `val_size+test_size={eval_size_check}`. "
                            f"Each class must have >= {min_required_in_temp} samples to guarantee at least 2 "
                            f"samples in the combined val+test partition. The problematic groups: {too_small_temp}"
                        )
            except Exception as e:
                # If we can't compute counts or there's an issue, raise a clearer error
                raise RuntimeError(
                    f"Failed to validate stratification column '{stratify_by_column}': {e}"
                )

            # Convert the composite string key into a ClassLabel feature so
            # Hugging Face's `train_test_split` can perform stratification.
            try:
                unique_labels = sorted(ds_to_split.unique(stratify_by_column))
                ds_to_split = ds_to_split.cast_column(stratify_by_column, ClassLabel(names=unique_labels))
            except Exception:
                # If casting fails for any reason, leave the dataset as-is
                # and allow the underlying split to raise a clear error.
                pass

        # Split 1: Train vs. Temp (Val + Test)
        eval_size = val_size + test_size
        if eval_size >= 1.0 or eval_size <= 0.0:
            raise ValueError("val_size + test_size must be strictly between 0 and 1.")

        split_1 = ds_to_split.train_test_split(
            test_size=eval_size, 
            stratify_by_column=stratify_by_column, 
            seed=seed
        )
        
        train_ds = split_1['train']
        temp_ds = split_1['test']

        # Split 2: Val vs. Test
        if test_size > 0:
            relative_test_size = test_size / eval_size
            split_2 = temp_ds.train_test_split(
                test_size=relative_test_size, 
                stratify_by_column=stratify_by_column, 
                seed=seed
            )
            val_ds = split_2['train']
            test_ds = split_2['test']
        else:
            val_ds = temp_ds
            test_ds = None

        # Cleanup composite key
        dataset_dict = DatasetDict({"train": train_ds, "val": val_ds})
        if test_ds is not None:
            dataset_dict["test"] = test_ds

        if stratify_by_column:
            for split in dataset_dict.keys():
                dataset_dict[split] = dataset_dict[split].remove_columns("_stratify_key")

        self._hf_dataset = dataset_dict
        return dataset_dict

    @property
    def dataset(self) -> DatasetDict:
        """Returns the underlying Hugging Face DatasetDict."""
        return self._hf_dataset
