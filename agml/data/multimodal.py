# Copyright 2021 UC Davis Plant AI and Biophysics Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import warnings

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".tif"}
_KNOWN_OPTIONAL_KEYS = {"origin_dataset"}


def _validate_manifest(dataset_root: str) -> None:
    """Validate a metadata.jsonl manifest against all 13 rules in the data contract."""
    manifest_path = os.path.join(dataset_root, "metadata.jsonl")

    # Rule 1: metadata.jsonl must exist.
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"metadata.jsonl not found at {manifest_path!r}. "
            f"Expected HF imagefolder layout with a metadata.jsonl manifest."
        )

    # Rule 2: At least one image file must exist directly in dataset_root (flat layout).
    image_files = set()
    for f in os.listdir(dataset_root):
        if os.path.isfile(os.path.join(dataset_root, f)) and os.path.splitext(f)[1].lower() in _IMAGE_EXTENSIONS:
            image_files.add(f)

    if not image_files:
        raise ValueError(
            f"No image files found directly in {dataset_root!r}. "
            f"Expected flat imagefolder layout: image files co-located with metadata.jsonl. "
            f"Supported extensions: {sorted(_IMAGE_EXTENSIONS)}."
        )

    # Rule 3 + 4: Parse all lines; ensure non-empty manifest.
    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"metadata.jsonl line {line_num} is not valid JSON: {exc}"
                ) from exc
            entries.append((line_num, entry))

    if not entries:
        raise ValueError("metadata.jsonl is empty — no samples to load.")

    seen_ids: set = set()
    missing_files: list = []
    duplicate_ids: list = []

    required_keys = {"file_name", "id", "messages"}

    for line_num, entry in entries:
        # Rule 5: Required keys; extra keys warn.
        if not isinstance(entry, dict):
            raise ValueError(f"metadata.jsonl line {line_num}: entry is not a JSON object.")
        missing = required_keys - entry.keys()
        if missing:
            raise ValueError(
                f"metadata.jsonl line {line_num}: missing required key(s): {sorted(missing)}."
            )
        extra = entry.keys() - required_keys - _KNOWN_OPTIONAL_KEYS
        if extra:
            warnings.warn(
                f"metadata.jsonl line {line_num} (id={entry.get('id')!r}) has unexpected "
                f"key(s): {sorted(extra)}. These will be ignored.",
                UserWarning,
                stacklevel=4,
            )

        file_name = entry["file_name"]
        sample_id = entry["id"]
        messages = entry["messages"]

        # Rule 8: file_name must be a plain filename with no directory separators (flat layout).
        if not isinstance(file_name, str) or not file_name:
            raise ValueError(
                f"metadata.jsonl line {line_num}: 'file_name' must be a non-empty string, "
                f"got {file_name!r}."
            )
        if "/" in file_name or os.sep in file_name:
            raise ValueError(
                f"metadata.jsonl line {line_num}: 'file_name' must be a plain filename with no "
                f"directory separators — flat imagefolder layout required. Got: {file_name!r}."
            )

        # Rule 6: Each file_name must reference an existing file.
        full_path = os.path.join(dataset_root, file_name)
        if not os.path.exists(full_path):
            missing_files.append(file_name)

        # Rule 7: id must be unique.
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(
                f"metadata.jsonl line {line_num}: 'id' must be a non-empty string, "
                f"got {sample_id!r}."
            )
        if sample_id in seen_ids:
            duplicate_ids.append(sample_id)
        seen_ids.add(sample_id)

        # Optional: origin_dataset must be a string if present (empty string is allowed).
        if "origin_dataset" in entry:
            origin = entry["origin_dataset"]
            if not isinstance(origin, str):
                raise ValueError(
                    f"metadata.jsonl line {line_num} (id={sample_id!r}): "
                    f"'origin_dataset' must be a string, got {type(origin).__name__!r}."
                )

        # Rule 9: messages must be a list of length >= 2.
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(
                f"metadata.jsonl line {line_num} (id={sample_id!r}): 'messages' must be a list "
                f"of at least 2 turns, got {len(messages) if isinstance(messages, list) else type(messages).__name__}."
            )

        # Rules 10, 11, 12: validate roles and content.
        valid_roles = {"user", "assistant", "system"}
        start_idx = 0
        if messages[0].get("role") == "system":
            start_idx = 1

        has_any_image = False
        for turn_idx, turn in enumerate(messages):
            if not isinstance(turn, dict):
                raise ValueError(
                    f"metadata.jsonl line {line_num} (id={sample_id!r}): "
                    f"messages[{turn_idx}] is not a dict."
                )

            # Rule 10: valid roles.
            role = turn.get("role")
            if role not in valid_roles:
                raise ValueError(
                    f"metadata.jsonl line {line_num} (id={sample_id!r}): "
                    f"messages[{turn_idx}].role must be one of {sorted(valid_roles)}, got {role!r}."
                )

            # Rule 11: alternation check (skip system turn at index 0).
            if turn_idx >= start_idx:
                expected = "user" if (turn_idx - start_idx) % 2 == 0 else "assistant"
                if role != expected:
                    raise ValueError(
                        f"metadata.jsonl line {line_num} (id={sample_id!r}): "
                        f"messages[{turn_idx}] expected role {expected!r} but got {role!r}. "
                        f"Roles must alternate user/assistant after any optional system turn."
                    )

            # Rule 12: content validation.
            content = turn.get("content")
            if not isinstance(content, list) or len(content) == 0:
                raise ValueError(
                    f"metadata.jsonl line {line_num} (id={sample_id!r}): "
                    f"messages[{turn_idx}].content must be a non-empty list."
                )
            for item_idx, item in enumerate(content):
                if not isinstance(item, dict) or "type" not in item:
                    raise ValueError(
                        f"metadata.jsonl line {line_num} (id={sample_id!r}): "
                        f"messages[{turn_idx}].content[{item_idx}] must be a dict with a 'type' field."
                    )
                item_type = item["type"]
                if item_type not in ("image", "text"):
                    raise ValueError(
                        f"metadata.jsonl line {line_num} (id={sample_id!r}): "
                        f"messages[{turn_idx}].content[{item_idx}].type must be 'image' or 'text', "
                        f"got {item_type!r}."
                    )
                if item_type == "text":
                    text_val = item.get("text")
                    if not isinstance(text_val, str) or not text_val:
                        raise ValueError(
                            f"metadata.jsonl line {line_num} (id={sample_id!r}): "
                            f"messages[{turn_idx}].content[{item_idx}] text-type item "
                            f"must have a non-empty 'text' field."
                        )
                elif item_type == "image":
                    has_any_image = True

        # Rule 13: warn if no image-type content item in any turn.
        if not has_any_image:
            warnings.warn(
                f"metadata.jsonl line {line_num} (id={sample_id!r}): no content item "
                f"with type='image' found in any turn. Text-only samples in a multimodal "
                f"dataset are unusual but legal.",
                UserWarning,
                stacklevel=4,
            )

    # Deferred errors for file and ID issues (collect all before raising).
    if missing_files:
        raise ValueError(
            f"metadata.jsonl references image file(s) that do not exist in {dataset_root!r}: "
            f"{missing_files}."
        )
    if duplicate_ids:
        raise ValueError(
            f"metadata.jsonl contains duplicate 'id' value(s): {duplicate_ids}."
        )


def load_multimodal_dataset(dataset_root: str) -> "datasets.Dataset":
    """Load a multimodal image_text_to_text dataset using HuggingFace imagefolder.

    Expects a flat directory layout:
        dataset_root/
            metadata.jsonl   (file_name values are plain filenames, no subdirectories)
            image001.jpg
            image002.jpg
            ...

    Returns a datasets.Dataset with columns:
        - image (datasets.Image)    — lazy-loaded PIL Image
        - id (str)                  — AgML identifier
        - messages (Sequence)       — HF-canonical multi-turn conversation
        - origin_dataset (str)      — source dataset name, if present in metadata.jsonl

    Parameters
    ----------
    dataset_root : str
        Path to the flat directory containing metadata.jsonl and image files.

    Returns
    -------
    datasets.Dataset
        All splits concatenated into a single flat dataset.
    """
    from datasets import load_dataset, concatenate_datasets

    _validate_manifest(dataset_root)

    ds = load_dataset("imagefolder", data_dir=dataset_root)

    # imagefolder auto-splits into train/validation/test; collapse to one flat dataset
    # since AgML handles splitting separately.
    if hasattr(ds, "keys"):
        ds = concatenate_datasets(list(ds.values()))

    return ds
