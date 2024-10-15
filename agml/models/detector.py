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
import warnings
import datetime
import importlib

from agml.framework import AgMLSerializable
from agml.backend.config import model_save_path
from agml.utils.logging import log

from agml.models.extensions.ultralytics import install_and_configure_ultralytics

try:
    import ultralytics
    from ultralytics import YOLO
except ImportError:
    # only set up ultralytics for the first time when
    ultralytics = None


VALID_YOLO11_MODELS = ['YOLO11n', 'YOLO11s', 'YOLO11m', 'YOLO11l', 'YOLO11x']
YOLO11_MODELS_TO_PATH = {k.upper(): f'{k.lower()}.pt' for k in VALID_YOLO11_MODELS}


class Detector(AgMLSerializable):
    def __new__(cls, model_name: str = "yolov5s", *args, **kwargs):
        # only setup Ultralytics when the class is initialized
        if importlib.util.find_spec("ultralytics") is None:
            install_and_configure_ultralytics()

            import ultralytics
            from ultralytics import YOLO

        return cls(model_name, *args, **kwargs)

    def __init__(self, net, *args, **kwargs):
        if not kwargs.get("_internally_instantiated", False):
            raise TypeError("An `agml.models.Detector` should only be instantiated "
                            "from the `Detector.load(<name>)` method or the "
                            "`Detector.load_benchmark(<name>)` method.")

        self.net = net

    @staticmethod
    def train(loader, model, run_name=None, epochs=100, overwrite=False):
        """
        Train a YOLO model using the Ultralytics package.

        This method can train a custom YOLO11 model on the provided `loader`,
        which enables efficient and quick training of a state-of-the-art object
        detector directly on an AgML dataset.

        Parameters
        ----------
        loader : agml.data.DatasetLoader
            The dataset loader to use for training the model.
        model : str
            The YOLO11 model to use for training. Choose from: ['YOLO11n', '
            YOLO11s', 'YOLO11m', 'YOLO11l', 'YOLO11x'].
        run_name : str, optional
            The name to use for the run. If not provided, the run name will be
            generated from the loader and model name.
        epochs : int, optional
            The number of epochs to train the model for. Default is 100.
        overwrite : bool, optional
            Whether to overwrite the model if it already exists. Default is False.

        Returns
        -------
        The name of the run that was trained.
        """
        # generate a run name from the loader and model and date if not provided
        if run_name is None:
            this_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_name = f"{loader.__class__.__name__}_{model}_{this_time}"

        # check whether a model exists already in the AgML internal model
        # directory, if not create it, if so then provide a warning unless
        # the user specifically wants to overwrite the model
        model_save_dir = os.path.join(os.path.expanduser('~'), '.agml', 'models', run_name)
        if os.path.exists(model_save_dir):
            if os.listdir(model_save_dir) and not overwrite:
                raise ValueError(f"Model with name {run_name} already exists. "
                                 f"Set `overwrite=True` to overwrite.")

        # export the YOLO dataset in the Ultralytics package directory
        ultralytics_dir = os.path.dirname(ultralytics.__file__)
        export_path_dict = loader.export_yolo(os.path.join(ultralytics_dir))

        try:  # instantiate the YOLO model based on the available choices
            net = YOLO(YOLO11_MODELS_TO_PATH[model.upper()])
        except KeyError:
            raise ValueError(f"Invalid model choice: {model}. Choose from: {VALID_YOLO11_MODELS}")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message='.*torch.amp.autocast')
            train_result = net.train(
                epochs=epochs,
                data=export_path_dict['metadata_path'],
            )

            model_save_path = train_result.save_dir / 'weights' / 'best.pt'
            model_results_csv = train_result.save_dir / 'results.csv'

        # move the trained model to the `~/.agml/models` directory where it
        # will sit under the name `run_name` for easy loading in the future
        shutil.move(model_save_path, os.path.join(model_save_dir, 'best.pt'))
        shutil.move(model_results_csv, os.path.join(model_save_dir, 'results.csv'))
        log(f"""Model training complete. Model saved to: {model_save_dir}

        For full Ultralytics training logs and outputs, see: {train_result.save_dir}

        To load the AgML model directly, you can run: 

        import agml.models
        model = agml.models.Detector('{run_name}')

        and the model with the best weights from this training run will be loaded.
        """)

        return run_name

    @classmethod
    def load(cls, name: str = None, weights: str = None):
        # get the path to the model based on the input arguments
        if name is not None and weights is not None:
            raise ValueError("You can only provide either a `name` or `weights` argument, not both.")
        if name is not None:
            model_path = os.path.join(model_save_path(), name, 'best.pt')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model with name {name} not found in the "
                                        f"internal AgML model repository at {model_path}. If "
                                        f"you want to load a custom path to weights, pass the "
                                        f"`weights` argument instead of `name`.")
        if weights is not None:
            model_path = weights
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model weights not found at the provided path: {model_path}")

        # load the model using the Ultralytics package
        net = YOLO(model_path)  # noqa
        return cls(net=net, _internally_instantiated=True)



