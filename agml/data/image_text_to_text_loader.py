import io
import os
import re
import zipfile
from collections import defaultdict
from functools import lru_cache
from typing import Optional, Union

import pandas as pd
from PIL import Image

try:
	from datasets import Dataset, DatasetDict
except ImportError:
	raise ImportError(
		"The `datasets` library is required to use loadImageTextToTextDataset. "
		"Please install it using `pip install datasets`."
	)

try:
	from huggingface_hub import snapshot_download
except ImportError:
	raise ImportError(
		"The `huggingface_hub` library is required to use loadImageTextToTextDataset. "
		"Please install it using `pip install huggingface_hub`."
	)

# Matches metadata parquet filenames such as "train-0000-of-0001.parquet",
# capturing the split name so splits can be auto-discovered.
_SPLIT_PARQUET_PATTERN = re.compile(r"^(?P<split>.+)-\d+-of-\d+\.parquet$")

# The relative path (from the repo root) where image shards and the shard index live.
_IMAGES_DIRNAME = "images"

# The parquet file mapping each image path to the shard file that contains it.
_SHARD_INDEX_FILENAME = "path_to_shard.parquet"

# The split name used when no split can be inferred from the metadata filenames.
_DEFAULT_SPLIT_NAME = "train"


class ImageTextToTextShardStore:
	"""
	Looks up and reads image bytes out of image-text-to-text shards without extracting them to disk.

	On construction, this reads `path_to_shard.parquet` into an in-memory
	`{path: shard_file}` index. Each shard's `zipfile.ZipFile` handle is opened
	lazily on first access and kept open for reuse, and reads are cached with
	an `lru_cache` since the same image path is often referenced by multiple rows.
	"""

	def __init__(self, images_dir: str):
		self.images_dir = images_dir
		index_path = os.path.join(images_dir, _SHARD_INDEX_FILENAME)
		index_df = pd.read_parquet(index_path)
		self._path_to_shard = dict(zip(index_df["path"], index_df["shard_file"]))
		self._zip_cache = {}

	def _getZip(self, shard_file: str):
		"""Returns the open `ZipFile` handle for a shard, opening it on first access."""
		zf = self._zip_cache.get(shard_file)
		if zf is None:
			shard_path = os.path.join(self.images_dir, shard_file)
			zf = zipfile.ZipFile(shard_path, "r")
			self._zip_cache[shard_file] = zf
		return zf

	@lru_cache(maxsize=4096)
	def _getBytesCached(self, path: str) -> bytes:
		"""Reads raw image bytes for a path, caching repeat reads of the same path."""
		shard_file = self._path_to_shard[path]
		return self._getZip(shard_file).read(path)

	def getBytes(self, path: str) -> bytes:
		"""Public entry point for reading the raw bytes of a single image path."""
		return self._getBytesCached(path)

	def __del__(self):
		# Best effort cleanup, the process may already be tearing down.
		for zf in getattr(self, "_zip_cache", {}).values():
			try:
				zf.close()
			except Exception:
				pass


def _makeDecodeTransform(store: ImageTextToTextShardStore):
	"""Builds the lazy `set_transform` callback that decodes the `images` column via `store`."""

	def decode(batch):
		decoded_rows = []
		for images in batch["images"]:
			decoded_row = []
			for image_ref in images:
				path = image_ref["path"] if isinstance(image_ref, dict) else image_ref
				raw_bytes = store.getBytes(path)
				decoded_row.append(Image.open(io.BytesIO(raw_bytes)))
			decoded_rows.append(decoded_row)
		batch["images"] = decoded_rows
		return batch

	return decode


def _discoverSplits(repo_dir: str) -> dict:
	"""
	Groups metadata parquet files in `repo_dir` by their auto-discovered split name.

	Filenames matching `<split>-N-of-M.parquet` are grouped under `<split>`. If
	none of the metadata parquet files match that convention, every parquet file
	at the repo root is treated as a single `train` split instead of failing.
	"""
	split_to_files = defaultdict(list)
	all_parquet_files = []
	for filename in sorted(os.listdir(repo_dir)):
		if not filename.endswith(".parquet"):
			continue
		all_parquet_files.append(os.path.join(repo_dir, filename))

		match = _SPLIT_PARQUET_PATTERN.match(filename)
		if match is not None:
			split_to_files[match.group("split")].append(os.path.join(repo_dir, filename))

	if not split_to_files and all_parquet_files:
		split_to_files[_DEFAULT_SPLIT_NAME] = all_parquet_files

	if not split_to_files:
		raise FileNotFoundError(f"No metadata parquet files were found in {repo_dir}.")

	return split_to_files


def loadImageTextToTextDataset(
	repo_id: str,
	split: Optional[str] = None,
	cache_dir: Optional[str] = None,
	token: Optional[str] = None,
) -> Union["DatasetDict", "Dataset"]:
	"""
	Loads an AgML image-text-to-text shard dataset from the Hub.

	This downloads `repo_id` with `snapshot_download`, builds an
	`ImageTextToTextShardStore` over the downloaded `images/` shards, and loads
	the metadata parquet(s) into an Arrow-backed dataset with a lazy transform
	attached that decodes the `images` column into `PIL.Image` objects on
	access. Splits are auto-discovered from the metadata filenames (e.g.
	`train-0000-of-0001.parquet` becomes `train`); if no split can be inferred,
	every row is placed under a single `train` split.

	The backing `ImageTextToTextShardStore` is attached to the returned dataset
	as a `.store` attribute rather than returned separately, so callers can
	inspect its cache or free its open file handles (`del ds.store`) without
	changing the common `ds = loadImageTextToTextDataset(...)` call shape.

	Parameters
	----------
	repo_id : str
		The Hugging Face Hub dataset repo id, e.g. `Project-AgML/AgroOmni-plainziptest`.
	split : str, optional
		If given, only that split is returned (as a `Dataset` rather than a `DatasetDict`).
	cache_dir : str, optional
		The local directory to cache the downloaded repo in.
	token : str, optional
		A Hugging Face Hub token, needed for private or gated repos.

	Returns
	-------
	DatasetDict or Dataset
		A `DatasetDict` (or a single `Dataset` if `split` was given), with the
		`ImageTextToTextShardStore` backing the lazy image decoding attached as
		its `.store` attribute.
	"""
	repo_dir = snapshot_download(repo_id=repo_id, repo_type="dataset", cache_dir=cache_dir, token=token)
	images_dir = os.path.join(repo_dir, _IMAGES_DIRNAME)
	store = ImageTextToTextShardStore(images_dir)
	decode = _makeDecodeTransform(store)

	split_to_files = _discoverSplits(repo_dir)
	dataset_dict = DatasetDict()
	for split_name, parquet_files in split_to_files.items():
		split_ds = Dataset.from_parquet(parquet_files)
		split_ds.set_transform(decode)
		dataset_dict[split_name] = split_ds

	if split is not None:
		result = dataset_dict[split]
		result.store = store
		return result

	dataset_dict.store = store
	return dataset_dict
