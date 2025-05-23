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

import os
import shutil
import zipfile

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.console import Console

from agml.utils.data import copyright_print, load_public_sources, maybe_you_meant
from agml.utils.logging import log


def download_dataset(dataset_name, dest_dir, redownload=False):
    """
    Downloads a dataset from the agdata-data s3 file storage.

    Parameters
    ----------
    dataset_name : str
        name of dataset to download
    dest_dir : str
        path for saving downloaded dataset
    redownload : bool
        whether to re-download the dataset
    """
    import requests

    # Validate dataset name
    source_info = load_public_sources()
    if dataset_name not in source_info.keys():
        if dataset_name.replace("-", "_") not in source_info.keys():
            msg = f"Received invalid public source: '{dataset_name}'."
            msg = maybe_you_meant(dataset_name, msg)
            raise ValueError(msg)
        else:
            log(f"Interpreted dataset '{dataset_name}' as " f"'{dataset_name.replace('-', '_')}.'")
            dataset_name = dataset_name.replace("-", "_")

    # Connect to S3 and generate unsigned URL for bucket object
    url = f"https://agdata-data.s3.us-west-1.amazonaws.com/datasets/{dataset_name}.zip"

    # Check if dataset already exists
    if dataset_name in dest_dir:
        dest_dir = os.path.dirname(dest_dir)
    exist_dir = os.path.join(dest_dir, dataset_name)
    if not redownload:
        if os.path.exists(exist_dir):
            log(f"Dataset '{dataset_name}' already exists " f"in '{exist_dir}', skipping download.")
            return
    elif os.path.exists(exist_dir) and redownload:
        shutil.rmtree(exist_dir)

    # File path of zipped dataset
    os.makedirs(dest_dir, exist_ok = True)
    dataset_download_path = os.path.join(dest_dir, os.path.basename(dataset_name) + '.zip')

    console = Console()
    progress = Progress(
        TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
        BarColumn(bar_width=10),
        "[progress.percentage]{task.percentage:.1f}%",
        "•",
        TransferSpeedColumn(),
        "•",
        DownloadColumn(),
        "•",
        TimeRemainingColumn(),
    )

    # Download object from bucket
    try:
        with progress:
            with requests.Session() as sess:
                r = sess.get(url, stream=True)
                r.raise_for_status()
                content_size = int(r.headers.get("content-length", 0))
                task_id = progress.add_task("download", filename=dataset_name + ".zip", total=content_size)
                with open(dataset_download_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        progress.update(task_id, advance=len(chunk))

    except BaseException as e:
        if os.path.exists(dataset_download_path):
            os.remove(dataset_download_path)
        raise e

    # Unzip downloaded dataset
    with zipfile.ZipFile(dataset_download_path, "r") as z:
        console.print(f"[AgML Download]: Extracting files for {dataset_name}... ", end="")
        z.extractall(path=dest_dir)
        console.print("[bold] Done!")

    # Delete zipped file
    if os.path.exists(dataset_download_path):
        os.remove(dataset_download_path)

    # Print dataset copyright info
    copyright_print(dataset_name, os.path.splitext(dataset_download_path)[0])


def download_model(model_name, dest_dir, redownload=False):
    """
    Downloads a model from the agdata-data s3 file storage.

    Parameters
    ----------
    model_name : str
        name of the dataset for the model to download
    dest_dir : str
        path for saving downloaded model
    redownload : bool
        whether to re-download the model
    """
    import requests

    # Validate dataset name
    source_info = load_public_sources()
    if model_name not in source_info.keys():
        if model_name.replace("-", "_") not in source_info.keys():
            msg = f"Received invalid public source: '{model_name}'."
            msg = maybe_you_meant(model_name, msg)
            raise ValueError(msg)
        else:
            log(f"Interpreted dataset '{model_name}' as " f"'{model_name.replace('-', '_')}.'")
            model_name = model_name.replace("-", "_")

    # Connect to S3 and generate unsigned URL for bucket object
    url = f"https://agdata-data.s3.us-west-1.amazonaws.com/models/{model_name}.pth"

    # Check if model already exists
    if model_name in dest_dir:
        dest_dir = os.path.dirname(dest_dir)
    exist_dir = os.path.join(dest_dir, model_name)
    if not redownload:
        if os.path.exists(exist_dir):
            log(f"Model '{model_name}' already exists " f"in '{exist_dir}', skipping download.")
            return
    elif os.path.exists(exist_dir) and redownload:
        shutil.rmtree(exist_dir)

    # File path of model
    os.makedirs(dest_dir, exist_ok=True)
    model_download_path = os.path.join(dest_dir, model_name + ".pth")

    progress = Progress(
        TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:.1f}%",
        "•",
        TransferSpeedColumn(),
        "•",
        DownloadColumn(),
        "•",
        TimeRemainingColumn(),
    )
    # Download object from bucket

    try:
        with progress:
            with requests.Session() as sess:
                r = sess.get(url, stream=True)
                r.raise_for_status()

                content_size = int(r.headers.get("content-length", 0))
                task_id = progress.add_task("download", filename=model_name + ".pth", total=content_size)
                with open(model_download_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        progress.update(task_id, advance=len(chunk))

    except BaseException as e:
        if os.path.exists(model_download_path):
            os.remove(model_download_path)
        raise e


def download_detector(detector_name, dest_dir, redownload=False):
    """
    Downloads a detector from the agdata-data s3 file storage.

    Parameters
    ----------
    detector_name : str
        name of the dataset for the model to download
    dest_dir : str
        path for saving downloaded model
    redownload : bool
        whether to re-download the model
    """
    import requests

    # Connect to S3 and generate unsigned URL for bucket object
    detector_name_url = detector_name.replace("+", "%2B")
    url = f"https://agdata-data.s3.us-west-1.amazonaws.com/models/detector/{detector_name_url}.zip"

    # Check if model already exists
    if detector_name in dest_dir:
        dest_dir = os.path.dirname(dest_dir)
    exist_dir = os.path.join(dest_dir, detector_name)
    if not redownload:
        if os.path.exists(exist_dir):
            log(f"Model '{detector_name}' already exists " f"in '{exist_dir}', skipping download.")
            return
    elif os.path.exists(exist_dir) and redownload:
        shutil.rmtree(exist_dir)

    # File path of model
    os.makedirs(dest_dir, exist_ok=True)
    model_download_path = os.path.join(dest_dir, detector_name + ".zip")
    console = Console()
    progress = Progress(
        TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:.1f}%",
        "•",
        TransferSpeedColumn(),
        "•",
        DownloadColumn(),
        "•",
        TimeRemainingColumn(),
    )
    # Download object from bucket
    try:
        with progress:
            with requests.Session() as sess:
                r = sess.get(url, stream=True)
                r.raise_for_status()

                content_size = int(r.headers.get("content-length", 0))
                task_id = progress.add_task("download", filename=detector_name + ".zip", total=content_size)

                with open(model_download_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        progress.update(task_id, advance=len(chunk))

    except BaseException as e:
        if os.path.exists(model_download_path):
            os.remove(model_download_path)
        raise e

    # Unzip downloaded dataset
    with zipfile.ZipFile(model_download_path, "r") as z:
        console.print(f"[bold] [AgML Download]: [normal pink] Extracting files for {detector_name}... ", end="")
        z.extractall(path=dest_dir)
        console.print("󱓞 Done! ")

    # Delete zipped file
    if os.path.exists(model_download_path):
        os.remove(model_download_path)
