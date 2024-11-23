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

import datetime
import gc
import importlib
import json
import os
import shutil
import warnings

import numpy as np
import torch

from agml.backend.config import model_save_path
from agml.data.public import public_data_sources, source
from agml.data.tools import convert_bbox_format
from agml.framework import AgMLSerializable
from agml.models.extensions.ultralytics import install_and_configure_ultralytics
from agml.utils.data import load_detector_benchmarks
from agml.utils.downloads import download_detector
from agml.utils.logging import log
from agml.viz.boxes import show_image_and_boxes

try:
    warnings.filterwarnings("ignore", message=".*Python>=3.10 is required")
    import ultralytics
    from ultralytics import YOLO
except ImportError:
    # only set up ultralytics for the first time when
    ultralytics = None


VALID_YOLO11_MODELS = ["YOLO11n", "YOLO11s", "YOLO11m", "YOLO11l", "YOLO11x"]
YOLO11_MODELS_TO_PATH = {k.upper(): f"{k.lower()}.pt" for k in VALID_YOLO11_MODELS}


class Detector(AgMLSerializable):
    """A class for object detection using the Ultralytics YOLO models.

    The `agml.models.Detector` is a simplified wrapper around an Ultralytics
    YOLO11 model, which enables easy loading and training of YOLO models, and
    use in inference pipelines and other AgML pipelines. You can train a model
    quickly using the following code:

    loader = agml.data.AgMLDataLoader('grape_detection_californiaday')
    agml.models.Detector.train(loader, model, run_name='grape_exp', epochs=100)

    You can then load the model and run inference as follows:

    grape_images = loader.take_images()
    model = agml.models.Detector.load('grape_exp')
    bboxes, classes, confidences = model(grape_images[0])

    You should use the methods `Detector.train` for training, and `Detector.load`
    and `Detector.load_benchmark` for loading models. The `Detector` class is
    designed to be a simple and easy-to-use interface for YOLO object detection.
    """

    serializable = frozenset(("net", "_verbose"))

    # the default path to save the models in the AgML internal model repository.
    DEFAULT_MODEL_PATH = os.path.join(model_save_path(), "detectors")

    def __new__(cls, net, *args, **kwargs):
        # only setup Ultralytics when the class is initialized
        if importlib.util.find_spec("ultralytics") is None:
            install_and_configure_ultralytics()

            import ultralytics
            from ultralytics import YOLO

        return super().__new__(cls)

    def __init__(self, net, *args, **kwargs):
        if not kwargs.get("_internally_instantiated", False):
            raise TypeError(
                "An `agml.models.Detector` should only be instantiated "
                "from the `Detector.load(<name>)` method or the "
                "`Detector.load_benchmark(<name>)` method."
            )

        self.net = net

        self._verbose = kwargs.get("verbose", False)
        self._benchmark_data = kwargs.get("benchmark_data", None)
        self._info = None

    def __call__(self, image, return_full=False, **kwargs):
        if isinstance(image, list) or (isinstance(image, np.ndarray) and len(image.shape) == 4):
            if return_full:
                return [self._single_input_inference(img, return_full, **kwargs) for img in image]
            bboxes, classes, confidences = [], [], []
            for img in image:
                bbox, cls, conf = self._single_input_inference(img, return_full, **kwargs)
                bboxes.append(bbox)
                classes.append(cls)
                confidences.append(conf)
            return bboxes, classes, confidences
        return self._single_input_inference(image, return_full, **kwargs)

    def _single_input_inference(self, image, return_full, **kwargs):
        # run inference on a single image
        # by default, return the predicted bboxes/classes/confidence. if
        # `return_full` is set to True, return the full result dictionary
        result_dict = self.net(image, stream=False, verbose=self._verbose)[0]
        if return_full:
            return result_dict
        result_dict = result_dict.cpu().numpy()
        boxes = result_dict.boxes
        bboxes = boxes.xyxy.astype(np.float32)
        bboxes = convert_bbox_format(bboxes, "xyxy")
        classes = boxes.cls.astype(np.int32)
        confidences = boxes.conf.astype(np.float32)
        return bboxes, classes, confidences

    def predict(self, image, return_full=False, **kwargs):
        """Runs inference on an image using the YOLO model.

        This method runs inference and returns the bounding boxes, classes,
        and confidences for the detected objects in the image. This method
        is a wrapper around the `__call__` method, which allows for easy
        inference on a single image or a list of images.

        Parameters
        ----------
        image : np.ndarray
            The image(s) to run inference on. This can accept anything
            that is valid with the Ultralytics YOLO inference format.
        return_full : bool, optional
            Whether to return the full result dictionary from the model.
            Default is False.

        Returns
        -------
        The desired output.
        """
        return self(image, return_full, **kwargs)

    def show_prediction(self, image):
        """Shows the output predictions for one input image.

        This method is useful for instantly visualizing the predictions
        for a single input image. It accepts a single input image (or
        any type of valid 'image' input, as described in the Ultralytics
        YOLO format), and then runs inference on that input image and
        displays its predictions in a matplotlib window.

        Parameters
        ----------
        image : Any
            See the Ultralytics YOLO format for allowed input types.

        Returns
        -------
        The matplotlib figure containing the image.
        """
        bboxes, labels, _ = self.predict(image)
        if isinstance(labels, int):
            bboxes, labels = [bboxes], [labels]
        return show_image_and_boxes(image, bboxes, labels, info=self._info)

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        if not isinstance(value, bool):
            raise ValueError("Verbose must be True/False.")
        self._verbose = value

    @property
    def benchmark(self):
        return self._benchmark_data

    @staticmethod
    def train(loader, model, run_name=None, epochs=100, overwrite=False, **kwargs):
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
        model_save_dir = os.path.join(Detector.DEFAULT_MODEL_PATH, run_name)
        if os.path.exists(model_save_dir):
            if os.listdir(model_save_dir) and not overwrite:
                raise ValueError(f"Model with name {run_name} already exists. " f"Set `overwrite=True` to overwrite.")
        os.makedirs(model_save_dir, exist_ok=True)

        # export the YOLO dataset in the Ultralytics package directory
        ultralytics_dir = os.path.dirname(ultralytics.__file__)
        export_path_dict = loader.export_yolo(os.path.join(ultralytics_dir))

        try:  # instantiate the YOLO model based on the available choices
            net = YOLO(YOLO11_MODELS_TO_PATH[model.upper()])
        except KeyError:
            raise ValueError(f"Invalid model choice: {model}. Choose from: {VALID_YOLO11_MODELS}")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*torch.amp.autocast")
            train_result = net.train(
                epochs=epochs,
                data=export_path_dict["metadata_path"],
                save_dir=f"runs/train/{run_name}",
                **kwargs,
            )

            model_save_path = train_result.save_dir / "weights" / "best.pt"
            model_results_csv = train_result.save_dir / "results.csv"
            model_args_path = train_result.save_dir / "args.yaml"

            # compose a complete result dictionary
            results = train_result.results_dict
            results["class_map"] = {
                "classes": train_result.names,
                "maps": train_result.maps.tolist(),
            }
            with open(os.path.join(model_save_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=4)

        # move the trained model to the `~/.agml/models` directory where it
        # will sit under the name `run_name` for easy loading in the future
        shutil.move(model_save_path, os.path.join(model_save_dir, "best.pt"))
        shutil.move(model_results_csv, os.path.join(model_save_dir, "results.csv"))
        shutil.move(model_args_path, os.path.join(model_save_dir, "args.yaml"))
        log(f"""Model training complete. Model saved to: {model_save_dir}

        For full Ultralytics training logs and outputs, see: {train_result.save_dir}

        To load the AgML model directly, you can run: 

        import agml.models
        model = agml.models.Detector.load('{run_name}')

        and the model with the best weights from this training run will be loaded.
        """)

        # clear complete cuda memory
        torch.cuda.empty_cache()
        gc.collect()

        return run_name

    @classmethod
    def load(cls, name: str = None, weights: str = None, **kwargs):
        """Loads a trained YOLO model from the AgML internal model repository.

        This method can be used to load a trained YOLO model from the AgML internal
        model repository. The model can be loaded either by providing the `name` of
        the model, which will be used to search for the model in the internal model
        repository, or by providing the `weights` path directly.

        You should use this method after training a model using the `Detector.train`
        method, which will train a model and save it to the `name` that is either
        provided or generated from the run.

        Parameters
        ----------
        name : str, optional
            The name of the model to load from the internal AgML model repository.
        weights : str, optional
            The path to the weights file to load directly.
        **kwargs : dict
            verbose : bool, optional
                Whether to enable verbose mode for the model. Default is False.

        Returns
        -------
        The loaded YOLO model.
        """
        # get the path to the model based on the input arguments
        if name is not None and weights is not None:
            raise ValueError("You can only provide either a `name` or `weights` argument, not both.")
        if name is not None:
            model_path = os.path.join(Detector.DEFAULT_MODEL_PATH, name, "best.pt")
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model with name {name} not found in the "
                    f"internal AgML model repository at {model_path}. If "
                    f"you want to load a custom path to weights, pass the "
                    f"`weights` argument instead of `name`."
                )
        if weights is not None:
            model_path = weights
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model weights not found at the provided path: {model_path}")

        # load the model using the Ultralytics package
        net = YOLO(model_path)  # noqa
        return cls(net=net, _internally_instantiated=True, **kwargs)

    @classmethod
    def load_benchmark(cls, dataset_name, yolo_model, **kwargs):
        """Loads a benchmarked YOLO model from the AgML internal model repository.

        This method can be used to load a benchmarked YOLO model from the AgML internal
        model repository. The model can be loaded by providing the `dataset_name` and
        the `yolo_model` name, which will be used to search for the model in the internal
        model repository.

        You should use this method after training a model using the `Detector.train`
        method, which will train a model and save it to the `name` that is either
        provided or generated from the run.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset that the model was trained on.
        yolo_model : str
            The name of the YOLO model to load from the internal AgML model repository.
        **kwargs : dict
            verbose : bool, optional
                Whether to enable verbose mode for the model. Default is False.

        Returns
        -------
        The loaded YOLO model.
        """
        if dataset_name not in public_data_sources(ml_task="object_detection"):
            raise ValueError(f"Dataset {dataset_name} is not a valid object detection dataset.")
        if yolo_model.upper() not in YOLO11_MODELS_TO_PATH.keys():
            if len(yolo_model) == 1:
                yolo_model = "yolo11" + yolo_model
                if yolo_model not in YOLO11_MODELS_TO_PATH.keys():
                    raise ValueError(f"YOLO model {yolo_model} is not a valid YOLO11 model.")
            raise ValueError(f"YOLO model {yolo_model} is not a valid YOLO11 model.")

        # Validate the pretrained benchmark name + model
        model_name = f"{dataset_name}+{yolo_model}"
        if model_name not in load_detector_benchmarks():
            raise ValueError(f"Could not find a valid benchmark for model ({model_name}).")

        # Download the model if it does not exist
        if not os.path.exists(os.path.join(Detector.DEFAULT_MODEL_PATH, model_name.replace("+", "-"))):
            download_detector(model_name, Detector.DEFAULT_MODEL_PATH)

        # Load the model
        benchmark_data = load_detector_benchmarks()[model_name]
        model_path = os.path.join(Detector.DEFAULT_MODEL_PATH, model_name.replace("+", "-"), "best.pt")
        net = cls.load(weights=model_path, benchmark_data=benchmark_data, **kwargs)
        net._info = source(dataset_name)
        return net
