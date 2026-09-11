import numpy as np
import pytest
from unittest.mock import patch
from datasets import Dataset, DatasetDict

from agml.data.multispectral_hf_loader import MultispectralDataLoader


PATCH_TARGET = "agml.data.hf_loader.load_dataset"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _band_arrays(n, shape=(4, 8, 8), dtype="float32", seed=0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(shape).astype(dtype) for _ in range(n)]


def _multispectral_dataset(n=100, shape=(4, 8, 8), dtype="float32", n_classes=5):
    """In-memory Dataset with a raw-bytes `bands` column — no network required."""
    arrays = _band_arrays(n, shape=shape, dtype=dtype)
    labels = (list(range(n_classes)) * (n // n_classes + 1))[:n]
    return Dataset.from_dict(
        {
            "bands": [a.tobytes() for a in arrays],
            "bands_shape": [list(shape)] * n,
            "bands_dtype": [dtype] * n,
            "label": labels,
        }
    ), arrays


def _presplit_multispectral_dataset_dict(n=100, shape=(4, 8, 8), dtype="float32", n_classes=5):
    ds, arrays = _multispectral_dataset(n, shape, dtype, n_classes)
    split = ds.train_test_split(test_size=0.2, seed=0)
    return DatasetDict({"train": split["train"], "test": split["test"]}), arrays


def _loader(dataset=None):
    if dataset is None:
        dataset, _ = _multispectral_dataset()
    with patch(PATCH_TARGET, return_value=dataset):
        return MultispectralDataLoader("org/dataset")


# ── Band decoding ────────────────────────────────────────────────────────────

class TestBandDecoding:
    def test_bands_decoded_to_numpy_array(self):
        loader = _loader()
        example = loader.dataset[0]
        assert isinstance(example["bands"], np.ndarray)

    def test_bands_decoded_shape(self):
        shape = (6, 16, 16)
        ds, _ = _multispectral_dataset(shape=shape)
        loader = _loader(ds)
        example = loader.dataset[0]
        assert example["bands"].shape == shape

    def test_bands_decoded_dtype(self):
        ds, _ = _multispectral_dataset(dtype="float64")
        loader = _loader(ds)
        example = loader.dataset[0]
        assert example["bands"].dtype == np.dtype("float64")

    def test_bands_values_round_trip(self):
        ds, arrays = _multispectral_dataset(n=10)
        loader = _loader(ds)
        for i in range(10):
            np.testing.assert_array_equal(loader.dataset[i]["bands"], arrays[i])

    def test_bands_array_is_writable(self):
        loader = _loader()
        example = loader.dataset[0]
        example["bands"][0, 0, 0] = 123.0
        assert example["bands"][0, 0, 0] == 123.0

    def test_bands_decoded_on_dataset_dict(self):
        fake, arrays = _presplit_multispectral_dataset_dict()
        with patch(PATCH_TARGET, return_value=fake):
            loader = MultispectralDataLoader("org/dataset")
        example = loader.dataset["train"][0]
        assert isinstance(example["bands"], np.ndarray)

    def test_batched_access_decodes_all_items(self):
        ds, arrays = _multispectral_dataset(n=5)
        loader = _loader(ds)
        batch = loader.dataset[0:5]
        assert len(batch["bands"]) == 5
        for arr in batch["bands"]:
            assert isinstance(arr, np.ndarray)

    def test_other_columns_unaffected(self):
        ds, _ = _multispectral_dataset(n=10)
        loader = _loader(ds)
        example = loader.dataset[0]
        assert "label" in example
        assert isinstance(example["label"], int)

    def test_no_bands_column_is_noop(self):
        ds = Dataset.from_dict({"label": [0, 1, 0, 1]})
        loader = _loader(ds)
        example = loader.dataset[0]
        assert "bands" not in example


# ── Inheritance ──────────────────────────────────────────────────────────────

class TestInheritance:
    def test_is_subclass_of_hf_loader(self):
        from agml.data.hf_loader import HuggingFaceDataLoader
        assert issubclass(MultispectralDataLoader, HuggingFaceDataLoader)

    def test_dataset_property_returns_internal(self):
        loader = _loader()
        assert loader.dataset is loader._hf_dataset

    def test_raises_on_load_failure(self):
        with patch(PATCH_TARGET, side_effect=Exception("not found")):
            with pytest.raises(RuntimeError, match="Failed to load Hugging Face dataset"):
                MultispectralDataLoader("nonexistent/dataset")


# ── Split ────────────────────────────────────────────────────────────────────

class TestSplit:
    def test_split_preserves_band_decoding(self):
        ds, _ = _multispectral_dataset(n=100)
        loader = _loader(ds)
        result = loader.split(val_size=0.2)
        assert set(result.keys()) == {"train", "val"}
        for split in result.values():
            assert isinstance(split[0]["bands"], np.ndarray)

    def test_split_updates_internal_dataset(self):
        loader = _loader()
        result = loader.split(val_size=0.2)
        assert loader._hf_dataset is result

    def test_val_and_test_split_decode_bands(self):
        ds, _ = _multispectral_dataset(n=100)
        loader = _loader(ds)
        result = loader.split(val_size=0.15, test_size=0.15)
        for split_name in ("train", "val", "test"):
            assert isinstance(result[split_name][0]["bands"], np.ndarray)

    def test_stratified_split_preserves_band_decoding(self):
        ds, _ = _multispectral_dataset(n=100, n_classes=5)
        loader = _loader(ds)
        result = loader.split(val_size=0.2, stratify_cols="label")
        assert isinstance(result["train"][0]["bands"], np.ndarray)
        assert isinstance(result["val"][0]["bands"], np.ndarray)

    def test_bands_shapes_correct_after_split(self):
        shape = (3, 10, 10)
        ds, _ = _multispectral_dataset(n=50, shape=shape)
        loader = _loader(ds)
        result = loader.split(val_size=0.2)
        assert result["train"][0]["bands"].shape == shape
        assert result["val"][0]["bands"].shape == shape
