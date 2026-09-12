import numpy as np

try:
    from datasets import DatasetDict
except ImportError:
    raise ImportError(
        "The `datasets` library is required to use the MultispectralDataLoader. "
        "Please install it using `pip install datasets`."
    )

from agml.data.hf_loader import HuggingFaceDataLoader


class MultispectralDataLoader(HuggingFaceDataLoader):
    """A `HuggingFaceDataLoader` variant for multispectral datasets.

    In addition to the base loader's functionality, this loader decodes a
    `bands` column stored on the Hub as raw binary buffers back into `numpy`
    arrays. The dataset is expected to expose the following columns:

    - `bands`: `binary`, the raw bytes of the array's buffer.
    - `bands_shape`: `list[int64]`, the array shape as `[channels, H, W]`.
    - `bands_dtype`: `string`, the `numpy` dtype name to reinterpret the buffer as.

    Decoding is applied lazily (mirroring how `Image` columns are decoded on
    access), so the raw bytes are only materialized into arrays when an
    example is actually retrieved.

    Parameters
    ----------
    dataset_name : str
        The name of a dataset on the Hugging Face Hub.
    config : str, optional
        The dataset configuration (subset) name for datasets that expose multiple
        configs (e.g. an augmented variant).
    cache_dir : str, optional
        The local directory to cache the dataset in.
    """

    def _cast_features(self):
        super()._cast_features()
        if isinstance(self._hf_dataset, DatasetDict):
            for split_name, ds in self._hf_dataset.items():
                self._hf_dataset[split_name] = self._decode_bands(ds)
        else:
            self._hf_dataset = self._decode_bands(self._hf_dataset)

    def _decode_bands(self, ds):
        """Attaches a lazy transform that decodes the `bands` column into `numpy` arrays."""
        if "bands" not in ds.features:
            return ds
        return ds.with_transform(self._bands_transform)

    @staticmethod
    def _bands_transform(batch):
        """Decodes raw `bands` byte buffers into `numpy` arrays using `bands_shape`/`bands_dtype`."""
        raw_bands = batch.get("bands")
        if raw_bands is None:
            return batch

        shapes = batch.get("bands_shape")
        dtypes = batch.get("bands_dtype")

        decoded = []
        for i, raw in enumerate(raw_bands):
            dtype = dtypes[i] if dtypes is not None else "float32"
            # `frombuffer` returns a read-only view over `raw`; copy so consumers
            # (e.g. in-place normalization/augmentation) get a writable array.
            arr = np.frombuffer(raw, dtype=np.dtype(dtype)).copy()
            shape = shapes[i] if shapes is not None else None
            if shape is not None:
                arr = arr.reshape(shape)
            decoded.append(arr)

        batch["bands"] = decoded
        return batch

    def split(self, *args, **kwargs) -> DatasetDict:
        """Splits the dataset, re-attaching the `bands` decoding transform to each split."""
        # `HuggingFaceDataLoader.split` internally calls `.map()`/`.unique()` on the
        # dataset to build/inspect the stratification key. Those operations read
        # examples through the current format, so the lazy `bands` decode transform
        # must be cleared beforehand — otherwise raw bytes get replaced by decoded
        # arrays mid-split, corrupting the underlying binary column.
        self._hf_dataset = self._reset_bands_format(self._hf_dataset)
        dataset_dict = super().split(*args, **kwargs)
        for split_name, ds in dataset_dict.items():
            dataset_dict[split_name] = self._decode_bands(ds)
        self._hf_dataset = dataset_dict
        return dataset_dict

    @staticmethod
    def _reset_bands_format(ds):
        """Clears any `bands` decoding transform, restoring the raw binary column."""
        if isinstance(ds, DatasetDict):
            return DatasetDict({name: split.with_format(None) for name, split in ds.items()})
        return ds.with_format(None)
