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

import argparse

from experiments.benchmarking.experiment import DetectionExperiment

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="The name of the experiment.", required=True)
    ap.add_argument(
        "--experiment_dir",
        help="The name of the experiment directory.",
        required=False,
        default=None,
    )
    ap.add_argument("--dataset", nargs="+", help="Datasets to use.")
    ap.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    ap.add_argument("--epochs", type=int, default=25, help="Number of epochs.")
    ap.add_argument(
        "--generalize-detections",
        action="store_true",
        default=False,
        help="Whether to generalize all model detections to one class.",
    )
    ap.add_argument(
        "--num_workers", type=int, default=8, help="Number of workers for DataLoaders."
    )
    ap.add_argument(
        "--pretrained_weights",
        type=str,
        default=None,
        help="Either the pretrained weights or `coco`.",
    )
    args = vars(ap.parse_args())

    exp = DetectionExperiment(args)
    exp.train()
