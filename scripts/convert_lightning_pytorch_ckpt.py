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

"""Converts PyTorch Lightning checkpoints to `nn.Module` state dicts."""

import argparse
import os
import shutil
from collections import OrderedDict
from fnmatch import fnmatch

import torch


def convert_state_dict(fpath):
    # load the file contents.
    contents = torch.load(fpath)

    # If the contents of the file are an `OrderedDict`, then
    # we don't need to extract the `state_dict`. However, the
    # nested key hierarchy might be different, so we check that.
    if isinstance(contents, OrderedDict):
        keys: list[str] = list(contents.keys())
        if len(keys) == 0:
            print("No state dict found.")
            return
        if keys[0].startswith("net"):
            out_dict = OrderedDict()
            for key in contents.keys():
                value = contents[key]
                out_dict[key.replace("net.", "")] = value
        else:
            return

    # Otherwise, get the model state dict from the contents
    # and re-save the file using the same name, just with only
    # the state dict and no PyTorch Lightning values.
    else:
        state_dict: OrderedDict = contents.get("state_dict", None)
        if state_dict is None:
            print(f"No state dict found in file {fpath}.")
            return

        # Parse the state dict and drop the first level from the keys.
        out_dict = OrderedDict()
        for key in state_dict.keys():
            value = state_dict[key]
            out_dict[key.replace("net.", "")] = value

    # Save the state dict.
    temp_path = os.path.join(os.path.dirname(fpath), "temp_state_dict.ckpt")
    shutil.copy(fpath, temp_path)  # save a copy in case an issue occurs
    os.remove(fpath)
    torch.save(out_dict, fpath.replace(".ckpt", ".pth"))
    os.remove(temp_path)
    print("Conversion Successful.")


# Parse input arguments (get the directory to search).
ap = argparse.ArgumentParser()
ap.add_argument(
    "--search_dir",
    type=str,
    required=True,
    help="The directory containing all of the checkpoints that you want"
    "to convert. This will search for all nested folders and files "
    "in the provided directory.",
)
search_dir = ap.parse_args().search_dir

# Search through and convert all of the files.
for path, subdirs, files in os.walk(os.path.abspath(os.path.normpath(search_dir))):
    for name in files:
        if fnmatch(name, "*.ckpt") or fnmatch(name, "*.pth"):
            print(f"Converting checkpoint at '{os.path.join(path, name)}'... ", end="")
            convert_state_dict(os.path.join(path, name))
