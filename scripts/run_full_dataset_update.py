#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import json
import os

import agml

# Check whether any dataset has been added that needs information updated (we do this
# by checking for any datasets in the `public_datasources.json` file that don't have
# corresponding information for the dataset statistics or other properties like shape).
with open(
    os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "agml",
        "_assets",
        "public_datasources.json",
    )
) as f:
    source_info = json.load(f)

# Get the datasets that need to be updated.
datasets = [i for i in source_info if "stats" not in source_info[i]]
print("DATASETS TO UPDATE:", datasets)

# For each of the datasets which need to be updated, run the corresponding update scripts.
for dataset in datasets:
    agml.data.download_public_dataset(dataset)  # only download dataset once
    os.system(f"python3 scripts/generate_normalization_info.py --datasets {dataset}")
    os.system(f"python3 scripts/generate_shape_info.py --datasets {dataset}")
    os.system(f"python3 scripts/generate_dataset_markdown.py")
