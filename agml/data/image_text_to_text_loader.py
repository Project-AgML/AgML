import io
import os
import re
import zipfile
from collections import OrderedDict, defaultdict
from typing import Optional, Union

import pandas as pd
import yaml
from PIL import Image

try:
	from datasets import Dataset, DatasetDict
except ImportError:
	raise ImportError(
		"The `datasets` library is required to use loadImageTextToTextDataset. "
		"Please install it using `pip install datasets`."
	)

try:
	from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
	from huggingface_hub.utils import EntryNotFoundError
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

# The dataset card holding the `configs` YAML front matter that names the default config.
_DATASET_CARD_FILENAME = "README.md"

# The default upper bound, in bytes, on each store's cache of raw image bytes.
_DEFAULT_CACHE_MAX_BYTES = 256 * 1024 * 1024


class ImageTextToTextShardStore:
	"""
	Looks up and reads image bytes out of image-text-to-text shards without extracting them to disk.

	On construction, this reads `path_to_shard.parquet` into an in-memory
	`{path: shard_file}` index. Each shard's `zipfile.ZipFile` handle is opened
	lazily on first access and kept open for reuse. Handles are tracked per
	process, so a store copied into a `DataLoader` worker (forked or pickled)
	opens its own handles rather than sharing the parent's file position.

	Reads are cached in a per-store LRU bounded by total bytes (`cache_max_bytes`),
	since the same image path is often referenced by multiple rows. Pass
	`cache_max_bytes=0` to disable the cache.
	"""

	def __init__(self, images_dir: str, cache_max_bytes: int = _DEFAULT_CACHE_MAX_BYTES):
		self.images_dir = images_dir
		self.cache_max_bytes = cache_max_bytes
		index_path = os.path.join(images_dir, _SHARD_INDEX_FILENAME)
		index_df = pd.read_parquet(index_path)
		self._path_to_shard = dict(zip(index_df["path"], index_df["shard_file"]))
		self._resetHandles()

	def _resetHandles(self):
		"""Drops all open zip handles and cached bytes, binding the store to the current process."""
		self._pid = os.getpid()
		self._zip_cache = {}
		self._bytes_cache = OrderedDict()
		self._cached_bytes = 0

	def _getZip(self, shard_file: str):
		"""Returns the open `ZipFile` handle for a shard, opening it on first access in this process."""
		if self._pid != os.getpid():
			# Forked into a new process: the inherited handles share the parent's file
			# position, so abandon them (without closing, the parent still owns them).
			self._resetHandles()
		zf = self._zip_cache.get(shard_file)
		if zf is None:
			shard_path = os.path.join(self.images_dir, shard_file)
			zf = zipfile.ZipFile(shard_path, "r")
			self._zip_cache[shard_file] = zf
		return zf

	def getBytes(self, path: str) -> bytes:
		"""Reads the raw bytes of a single image path, caching repeat reads of the same path."""
		if self._pid != os.getpid():
			self._resetHandles()
		raw_bytes = self._bytes_cache.get(path)
		if raw_bytes is not None:
			self._bytes_cache.move_to_end(path)
			return raw_bytes

		shard_file = self._path_to_shard[path]
		raw_bytes = self._getZip(shard_file).read(path)

		if len(raw_bytes) <= self.cache_max_bytes:
			self._bytes_cache[path] = raw_bytes
			self._cached_bytes += len(raw_bytes)
			while self._cached_bytes > self.cache_max_bytes:
				_, evicted = self._bytes_cache.popitem(last=False)
				self._cached_bytes -= len(evicted)
		return raw_bytes

	def close(self):
		"""Closes every open shard handle and clears the byte cache. Handles reopen lazily on next access."""
		if getattr(self, "_pid", None) == os.getpid():
			for zf in self._zip_cache.values():
				try:
					zf.close()
				except Exception:
					pass
		self._resetHandles()

	def __getstate__(self):
		# Open file handles can't be pickled (e.g. into spawned `DataLoader` workers),
		# and the byte cache shouldn't be shipped along, so drop both.
		state = self.__dict__.copy()
		state["_zip_cache"] = {}
		state["_bytes_cache"] = OrderedDict()
		state["_cached_bytes"] = 0
		state["_pid"] = None
		return state

	def __del__(self):
		# Best effort cleanup, the process may already be tearing down.
		try:
			self.close()
		except Exception:
			pass


class _DecodeTransform:
	"""
	The lazy `set_transform` callback that decodes the `images` column via `store`.

	This is a module-level class rather than a closure so the dataset it is
	attached to can be pickled into spawned `DataLoader` workers.
	"""

	def __init__(self, store: ImageTextToTextShardStore):
		self.store = store

	def __call__(self, batch):
		decoded_rows = []
		for images in batch["images"]:
			decoded_row = []
			for image_ref in images:
				path = image_ref["path"] if isinstance(image_ref, dict) else image_ref
				raw_bytes = self.store.getBytes(path)
				decoded_row.append(Image.open(io.BytesIO(raw_bytes)))
			decoded_rows.append(decoded_row)
		batch["images"] = decoded_rows
		return batch


def _discoverSplits(config_dir: str) -> dict:
	"""
	Groups metadata parquet files in `config_dir` by their auto-discovered split name.

	Filenames matching `<split>-N-of-M.parquet` are grouped under `<split>`. If
	none of the metadata parquet files match that convention, every parquet file
	in `config_dir` is treated as a single `train` split instead of failing.
	"""
	split_to_files = defaultdict(list)
	all_parquet_files = []
	for filename in sorted(os.listdir(config_dir)):
		if not filename.endswith(".parquet"):
			continue
		all_parquet_files.append(os.path.join(config_dir, filename))

		match = _SPLIT_PARQUET_PATTERN.match(filename)
		if match is not None:
			split_to_files[match.group("split")].append(os.path.join(config_dir, filename))

	if not split_to_files and all_parquet_files:
		split_to_files[_DEFAULT_SPLIT_NAME] = all_parquet_files

	if not split_to_files:
		raise FileNotFoundError(f"No metadata parquet files were found in {config_dir}.")

	return split_to_files


def _fetchDefaultConfigName(repo_id: str, token: Optional[str], cache_dir: Optional[str]) -> Optional[str]:
	"""
	Reads the default config name out of the dataset card's `configs` YAML front matter.

	This mirrors how `datasets`/the Hub itself pick a default config: a repo's
	`README.md` starts with a `---`-delimited YAML block listing each config
	under `configs`, with the one meant to load by default marked `default: true`.
	If no entry is marked, the first one listed is used, matching Hub behavior.

	Only `README.md` itself is downloaded here, not the shards, since this runs
	before we know which config's data is actually needed.
	"""
	try:
		readme_path = hf_hub_download(
			repo_id=repo_id,
			repo_type="dataset",
			filename=_DATASET_CARD_FILENAME,
			token=token,
			cache_dir=cache_dir,
		)
	except EntryNotFoundError:
		return None

	with open(readme_path, "r", encoding="utf-8") as f:
		text = f.read()

	if not text.startswith("---"):
		return None
	front_matter_end = text.find("\n---", 3)
	if front_matter_end == -1:
		return None

	try:
		card_metadata = yaml.safe_load(text[3:front_matter_end]) or {}
	except yaml.YAMLError:
		return None

	configs = card_metadata.get("configs")
	if not configs:
		return None

	for entry in configs:
		if entry.get("default"):
			return entry.get("config_name")

	return configs[0].get("config_name")


def _resolveConfig(
	repo_id: str, config: Optional[str], token: Optional[str], cache_dir: Optional[str]
) -> Optional[str]:
	"""
	Resolves which config's files to download, without downloading any shard.

	Lists the repo's files through the Hub API (a cheap metadata call) to tell
	single-config repos (files at the root) apart from multi-config repos (that
	structure repeated under a folder per config), and to know each config's
	name without pulling any shard data down. Returns `None` for a single-config
	repo, whose files sit at the repo root.
	"""
	shard_index_suffix = f"/{_IMAGES_DIRNAME}/{_SHARD_INDEX_FILENAME}"
	files = list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)

	root_is_config = f"{_IMAGES_DIRNAME}/{_SHARD_INDEX_FILENAME}" in files
	available_configs = sorted({f[: -len(shard_index_suffix)] for f in files if f.endswith(shard_index_suffix)})

	if config is not None:
		if config not in available_configs:
			if not available_configs:
				raise ValueError(f"'{repo_id}' has no configs, it is a single-config dataset, drop `config=`.")
			raise ValueError(f"Config '{config}' was not found in '{repo_id}'. Available configs: {available_configs}.")
		return config

	if root_is_config:
		return None

	if not available_configs:
		raise FileNotFoundError(f"No '{_IMAGES_DIRNAME}/' folder was found in '{repo_id}' or any of its subfolders.")

	default_config = _fetchDefaultConfigName(repo_id, token, cache_dir)
	if default_config in available_configs:
		return default_config

	return available_configs[0]


def loadImageTextToTextDataset(
	repo_id: str,
	config: Optional[str] = None,
	split: Optional[str] = None,
	cache_dir: Optional[str] = None,
	token: Optional[str] = None,
	cache_max_bytes: int = _DEFAULT_CACHE_MAX_BYTES,
) -> Union["DatasetDict", "Dataset"]:
	"""
	Loads an AgML image-text-to-text shard dataset from the Hub.

	This resolves which config to load through the Hub API first, without
	downloading any shard, then downloads only that config's files with
	`snapshot_download`. It builds an `ImageTextToTextShardStore` over the
	downloaded `images/` shards, and loads the metadata parquet(s) into an
	Arrow-backed dataset with a lazy transform attached that decodes the
	`images` column into `PIL.Image` objects on access. Splits are
	auto-discovered from the metadata filenames (e.g. `train-0000-of-0001.parquet`
	becomes `train`); if no split can be inferred, every row is placed under a
	single `train` split.

	Single-config repos keep their metadata parquet(s) and `images/` folder at
	the repo root, so the whole repo is downloaded either way. Multi-config
	repos repeat that structure under a folder per config, in which case only
	the resolved config's folder is downloaded; if `config` isn't given, the
	default config named in the dataset card's `configs` YAML front matter is
	used.

	The backing `ImageTextToTextShardStore` is attached to the returned dataset
	as a `.store` attribute rather than returned separately, so callers can
	inspect its cache or free its open file handles (`ds.store.close()`) without
	changing the common `ds = loadImageTextToTextDataset(...)` call shape.

	Parameters
	----------
	repo_id : str
		The Hugging Face Hub dataset repo id, e.g. `Project-AgML/AgroOmni-plainziptest`.
	config : str, optional
		The config (subset) name to load, for repos with multiple configs each
		under their own folder. If omitted, the dataset card's declared default
		config is used.
	split : str, optional
		If given, only that split is returned (as a `Dataset` rather than a `DatasetDict`).
	cache_dir : str, optional
		The local directory to cache the downloaded repo and the Arrow
		files built from its metadata parquet(s) in.
	token : str, optional
		A Hugging Face Hub token, needed for private or gated repos.
	cache_max_bytes : int, optional
		The upper bound, in bytes, on the store's in-memory cache of raw image
		bytes (256 MiB by default). Each `DataLoader` worker keeps its own
		cache of up to this size. Pass `0` to disable caching.

	Returns
	-------
	DatasetDict or Dataset
		A `DatasetDict` (or a single `Dataset` if `split` was given), with the
		`ImageTextToTextShardStore` backing the lazy image decoding attached as
		its `.store` attribute.
	"""
	resolved_config = _resolveConfig(repo_id, config, token, cache_dir)
	allow_patterns = None if resolved_config is None else [f"{resolved_config}/*"]
	repo_dir = snapshot_download(
		repo_id=repo_id,
		repo_type="dataset",
		cache_dir=cache_dir,
		token=token,
		allow_patterns=allow_patterns,
	)
	config_dir = repo_dir if resolved_config is None else os.path.join(repo_dir, resolved_config)
	images_dir = os.path.join(config_dir, _IMAGES_DIRNAME)
	store = ImageTextToTextShardStore(images_dir, cache_max_bytes=cache_max_bytes)
	decode = _DecodeTransform(store)

	split_to_files = _discoverSplits(config_dir)
	dataset_dict = DatasetDict()
	for split_name, parquet_files in split_to_files.items():
		split_ds = Dataset.from_parquet(parquet_files, split=split_name, cache_dir=cache_dir)
		split_ds.set_transform(decode)
		dataset_dict[split_name] = split_ds

	if split is not None:
		result = dataset_dict[split]
		result.store = store
		return result

	dataset_dict.store = store
	return dataset_dict
