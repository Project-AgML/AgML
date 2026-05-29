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

class HuggingFaceDataLoader(AgMLSerializable):
    """A data loader designed for loading datasets directly into Hugging Face formats.

    This loader retrieves datasets from the Hugging Face Hub and structures them
    into Hugging Face `DatasetDict` objects natively compatible with `transformers`
    or `diffusers` pipelines. It supports custom multi-column stratification.

    Parameters
    ----------
    dataset_name : str
        The name of a dataset on the Hugging Face Hub.
    task : str
        The computer vision task. Must be one of 'classification', 'detection', or 'segmentation'.
    cache_dir : str, optional
        The local directory to cache the dataset in.
    """

    serializable = frozenset(("dataset_name", "task", "cache_dir"))

    def __init__(self, dataset_name: str, task: str, cache_dir: str = None, **kwargs):
        if task not in ['classification', 'detection', 'segmentation']:
            raise ValueError("Task must be 'classification', 'detection', or 'segmentation'.")
        
        self.dataset_name = dataset_name
        self.task = task
        self.cache_dir = cache_dir

        self._hf_dataset = None
        self._setup_loader()

    def _setup_loader(self):
        """Loads the dataset from the Hugging Face Hub."""
        try:
            self._hf_dataset = load_dataset(self.dataset_name, cache_dir=self.cache_dir)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Hugging Face dataset '{self.dataset_name}': {e}"
            )

        self._cast_features()

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
