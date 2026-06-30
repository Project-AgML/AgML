import pytest
from unittest.mock import patch, MagicMock
from datasets import Dataset, DatasetDict, Image as HFImage, Value

from agml.data.hf_loader import HuggingFaceDataLoader


PATCH_TARGET = "agml.data.hf_loader.load_dataset"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _classification_dataset(n=100, n_classes=5):
    """In-memory Dataset with integer labels — no network required."""
    labels = (list(range(n_classes)) * (n // n_classes + 1))[:n]
    return Dataset.from_dict({"label": labels})


def _presplit_dataset_dict(n=100, n_classes=5):
    ds = _classification_dataset(n, n_classes)
    split = ds.train_test_split(test_size=0.2, seed=0)
    return DatasetDict({"train": split["train"], "test": split["test"]})


def _loader(dataset=None):
    if dataset is None:
        dataset = _classification_dataset()
    with patch(PATCH_TARGET, return_value=dataset):
        return HuggingFaceDataLoader("org/dataset")


# ── Initialisation ────────────────────────────────────────────────────────────

class TestInit:
    def test_stores_dataset_dict(self):
        fake = _presplit_dataset_dict()
        with patch(PATCH_TARGET, return_value=fake):
            loader = HuggingFaceDataLoader("org/dataset")
        assert isinstance(loader._hf_dataset, DatasetDict)

    def test_stores_plain_dataset(self):
        fake = _classification_dataset()
        with patch(PATCH_TARGET, return_value=fake):
            loader = HuggingFaceDataLoader("org/dataset")
        assert isinstance(loader._hf_dataset, Dataset)

    def test_passes_config_and_cache_dir(self):
        with patch(PATCH_TARGET, return_value=_presplit_dataset_dict()) as mock_load:
            HuggingFaceDataLoader("org/dataset", config="v2", cache_dir="/tmp/cache")
        mock_load.assert_called_once_with("org/dataset", "v2", cache_dir="/tmp/cache")

    def test_raises_on_load_failure(self):
        with patch(PATCH_TARGET, side_effect=Exception("not found")):
            with pytest.raises(RuntimeError, match="Failed to load Hugging Face dataset"):
                HuggingFaceDataLoader("nonexistent/dataset")

    def test_error_message_includes_config(self):
        with patch(PATCH_TARGET, side_effect=Exception("not found")):
            with pytest.raises(RuntimeError, match="org/dataset/v2"):
                HuggingFaceDataLoader("org/dataset", config="v2")

    def test_dataset_property_returns_internal(self):
        loader = _loader()
        assert loader.dataset is loader._hf_dataset


# ── Feature casting ───────────────────────────────────────────────────────────

class TestCastFeatures:
    def test_casts_image_column_on_dataset_dict(self):
        mock_ds = MagicMock()
        mock_ds.features = {"image": Value("string")}
        mock_ds.cast_column.return_value = mock_ds

        fake = DatasetDict({"train": mock_ds, "test": mock_ds})
        with patch(PATCH_TARGET, return_value=fake):
            HuggingFaceDataLoader("org/dataset")

        # _cast_single called once per split, each calls cast_column once
        assert mock_ds.cast_column.call_count == 2

    def test_casts_image_column_on_plain_dataset(self):
        mock_ds = MagicMock()
        mock_ds.features = {"image": Value("string")}
        mock_ds.cast_column.return_value = mock_ds

        with patch(PATCH_TARGET, return_value=mock_ds):
            HuggingFaceDataLoader("org/dataset")

        mock_ds.cast_column.assert_called_once_with("image", HFImage())

    def test_casts_mask_column(self):
        mock_ds = MagicMock()
        mock_ds.features = {"mask": Value("string")}
        mock_ds.cast_column.return_value = mock_ds

        with patch(PATCH_TARGET, return_value=mock_ds):
            HuggingFaceDataLoader("org/dataset")

        mock_ds.cast_column.assert_called_once_with("mask", HFImage())

    def test_skips_already_cast_image_column(self):
        mock_ds = MagicMock()
        mock_ds.features = {"image": HFImage()}
        mock_ds.cast_column.return_value = mock_ds

        with patch(PATCH_TARGET, return_value=mock_ds):
            HuggingFaceDataLoader("org/dataset")

        mock_ds.cast_column.assert_not_called()

    def test_does_not_cast_label_column(self):
        mock_ds = MagicMock()
        mock_ds.features = {"label": Value("string")}
        mock_ds.cast_column.return_value = mock_ds

        with patch(PATCH_TARGET, return_value=mock_ds):
            HuggingFaceDataLoader("org/dataset")

        mock_ds.cast_column.assert_not_called()

    def test_casts_both_image_and_mask(self):
        mock_ds = MagicMock()
        mock_ds.features = {"image": Value("string"), "mask": Value("string")}
        mock_ds.cast_column.return_value = mock_ds

        with patch(PATCH_TARGET, return_value=mock_ds):
            HuggingFaceDataLoader("org/dataset")

        assert mock_ds.cast_column.call_count == 2


# ── Split — no-op ─────────────────────────────────────────────────────────────

class TestSplitNoOp:
    def test_wraps_plain_dataset_in_train_key(self):
        loader = _loader(_classification_dataset())
        result = loader.split()
        assert isinstance(result, DatasetDict)
        assert "train" in result

    def test_plain_dataset_wrapping_updates_internal(self):
        loader = _loader(_classification_dataset())
        result = loader.split()
        assert loader._hf_dataset is result

    def test_returns_existing_dataset_dict_unchanged(self):
        original = _presplit_dataset_dict()
        loader = _loader(original)
        result = loader.split()
        assert isinstance(result, DatasetDict)
        assert set(result.keys()) == {"train", "test"}


# ── Split — sizing ────────────────────────────────────────────────────────────

class TestSplitSizing:
    def test_val_only(self):
        loader = _loader()
        result = loader.split(val_size=0.2)
        assert set(result.keys()) == {"train", "val"}
        assert sum(len(result[s]) for s in result) == 100

    def test_val_and_test(self):
        loader = _loader()
        result = loader.split(val_size=0.15, test_size=0.15)
        assert set(result.keys()) == {"train", "val", "test"}
        assert sum(len(result[s]) for s in result) == 100

    def test_approximate_split_proportions(self):
        loader = _loader(_classification_dataset(200))
        result = loader.split(val_size=0.2, test_size=0.1)
        assert abs(len(result["val"]) - 40) <= 3
        assert abs(len(result["test"]) - 20) <= 3
        assert abs(len(result["train"]) - 140) <= 3

    def test_split_updates_internal_dataset(self):
        loader = _loader()
        result = loader.split(val_size=0.2)
        assert loader._hf_dataset is result

    def test_total_size_gte_one_raises(self):
        loader = _loader()
        with pytest.raises(ValueError):
            loader.split(val_size=0.6, test_size=0.5)

    def test_zero_eval_size_raises(self):
        loader = _loader()
        with pytest.raises(ValueError):
            loader.split(val_size=0.0, test_size=0.0)


# ── Split — stratification ────────────────────────────────────────────────────

class TestSplitStratification:
    def test_stratified_val_split(self):
        loader = _loader(_classification_dataset(100, 5))
        result = loader.split(val_size=0.2, stratify_cols="label")
        assert set(result.keys()) == {"train", "val"}

    def test_stratified_val_test_split(self):
        loader = _loader(_classification_dataset(100, 5))
        result = loader.split(val_size=0.15, test_size=0.15, stratify_cols="label")
        assert set(result.keys()) == {"train", "val", "test"}

    def test_stratify_key_column_removed(self):
        loader = _loader(_classification_dataset(100, 5))
        result = loader.split(val_size=0.2, stratify_cols="label")
        for split in result.values():
            assert "_stratify_key" not in split.column_names

    def test_stratified_split_accepts_list_of_cols(self):
        ds = Dataset.from_dict({
            "label": (["cat", "dog"] * 50),
            "source": (["a", "b"] * 50),
        })
        loader = _loader(ds)
        result = loader.split(val_size=0.2, stratify_cols=["label", "source"])
        assert set(result.keys()) == {"train", "val"}

    def test_stratification_fails_with_too_few_samples_per_class(self):
        # 2 samples, 2 classes → each class has 1 sample, below the minimum of 2
        ds = Dataset.from_dict({"label": [0, 1]})
        loader = _loader(ds)
        with pytest.raises((ValueError, RuntimeError)):
            loader.split(val_size=0.5, stratify_cols="label")
