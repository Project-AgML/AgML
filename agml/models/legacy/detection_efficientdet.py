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

import warnings

import albumentations as A
import numpy as np
import torch
from tqdm import tqdm

try:
    from ensemble_boxes.ensemble_boxes_wbf import weighted_boxes_fusion
except ImportError:
    raise ImportError(
        "Could not find an installation of the `ensemble_boxes` package. "
        "Try `pip install ensemble-boxes` to install it."
    )

try:
    from effdet import DetBenchPredict, DetBenchTrain, create_model_from_config, get_efficientdet_config
except ImportError:
    raise ImportError(
        "Could not find an installation of the `effdet` package. "
        "Try `pip install effdet==0.2.4` to install it (note that "
        "the version is important for proper functionality)."
    )

from agml.backend.tftorch import is_array_like
from agml.data.public import source
from agml.models.base import AgMLModelBase
from agml.models.benchmarks import BenchmarkMetadata
from agml.models.metrics.map import MeanAveragePrecision
from agml.models.tools import auto_move_data
from agml.utils.general import has_func
from agml.utils.image import resolve_image_size
from agml.utils.logging import log
from agml.viz.boxes import show_image_and_boxes


class DetectionModel(AgMLModelBase):
    """Wraps an `EfficientDetD4` model for agricultural object detection.

    When using the model for inference, you should use the `predict()` method
    on any set of input images. This method wraps the `forward()` call with
    additional steps which perform the necessary preprocessing on the inputs,
    including resizing, normalization, and batching, as well as additional
    post-processing on the outputs (such as converting one-hot labels to
    integer labels), which will allow for a streamlined inference pipeline.

    If you want to use the `forward()` method directly, so that you can just
    call `model(inputs)`, then make sure your inputs are properly processed.
    This can be done by calling `model.preprocess_input(inputs)` on the
    input list of images/single image and then passing that result to the model.
    This will also return a one-hot label feature vector instead of integer
    labels, in the case that you want further customization of the outputs.

    By default, when instantiating a `DetectionModel`, it is prepared in
    inference mode. In order to use this model for training, you need to convert
    it to training mode by using `DetectionModel.switch_train()`. To convert
    back to inference mode, use `DetectionModel.switch_predict()`.

    If you want to use your own custom model and/or training pipeline, without the
    existing input restrictions, then you can subclass this model. In the `super`
    call in the `__init__` method, pass the parameter `model_initialized = True`,
    which will enable you to initialize the model in your own format.

    Parameters
    ----------
    num_classes : int
        The number of classes for the `EfficientDet` model.
    image_size : int, tuple
        The shape of image inputs to the model.
    conf_threshold : float
        Filters bounding boxes by their level of confidence based on this threshold.
    """

    serializable = frozenset(("model", "num_classes", "conf_thresh", "image_size"))
    state_override = frozenset(("model",))

    _ml_task = "object_detection"

    def __init__(self, num_classes=1, image_size=512, conf_threshold=0.3, **kwargs):
        # Initialize the base modules.
        super(DetectionModel, self).__init__()

        # If being initialized by a subclass, then don't do any of
        # model construction logic (since that's already been done).
        if not kwargs.get("model_initialized", False):
            # Construct the network and load in pretrained weights.
            self._image_size = resolve_image_size(image_size)
            self._confidence_threshold = conf_threshold
            self._num_classes = num_classes
            self.model = self._construct_sub_net(
                self._num_classes,
                self._image_size,
                pretrained=kwargs.get("pretrained", True),
            )

        # Filter out unnecessary warnings.
        warnings.filterwarnings("ignore", category=UserWarning, module="ensemble_boxes")
        warnings.filterwarnings("ignore", category=UserWarning, module="effdet.bench")  # noqa

    @auto_move_data
    def forward(self, *batch):
        """Ensures that the input is valid for the model."""
        if len(batch) == 2:
            return self.model(*batch)
        return self.model(batch[0])

    @staticmethod
    def _construct_sub_net(num_classes, image_size, pretrained=False):
        cfg = get_efficientdet_config("tf_efficientdet_d4")
        cfg.update({"image_size": image_size})
        model = create_model_from_config(cfg, pretrained=pretrained, num_classes=num_classes)
        return DetBenchPredict(model)

    def switch_predict(self):
        """Prepares the model for evaluation mode."""
        state = self.model.state_dict()
        if not isinstance(self.model, DetBenchPredict):
            self.model = DetBenchPredict(self.model.model)
        self.model.load_state_dict(state)

    def switch_train(self):
        """Prepares the model for training mode."""
        state = self.model.state_dict()
        if not isinstance(self.model, DetBenchTrain):
            self.model = DetBenchTrain(self.model.model)
        self.model.load_state_dict(state)

    @property
    def original(self):  # override for detection models.
        return self.model.model

    @torch.jit.ignore()
    def reset_class_net(self, num_classes=1):
        """Reconfigures the output class net for a new number of classes.

        Parameters
        ----------
        num_classes : int
            The number of classes to reconfigure the output net to use.
        """
        if num_classes != self._num_classes:
            self.model.model.reset_head(num_classes=num_classes)

    @staticmethod
    def _preprocess_image(image, image_size):
        """Preprocesses a single input image to EfficientNet standards.

        The preprocessing steps are applied logically; if the images
        are passed with preprocessing already having been applied, for
        instance, the images are already resized or they are already been
        normalized, the operation is not applied again, for efficiency.

        Preprocessing includes the following steps:

        1. Resizing the image to size (224, 224).
        2. Performing normalization with ImageNet parameters.
        3. Converting the image into a PyTorch tensor format.

        as well as other intermediate steps such as adding a channel
        dimension for two-channel inputs, for example.
        """
        # Convert the image to a NumPy array.
        if is_array_like(image) and hasattr(image, "numpy"):
            image = image.numpy()

        # Add a channel dimension for grayscale imagery.
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        # If the image is already in channels-first format, convert
        # it back temporarily until preprocessing has concluded.
        if image.shape[0] <= 3:
            image = np.transpose(image, (1, 2, 0))

        # Resize the image to ImageNet standards.
        (w, h) = image_size
        rz = A.Resize(height=h, width=w)
        if image.shape[0] != h or image.shape[1] != w:
            image = rz(image=image)["image"]

        # Normalize the image to ImageNet standards.
        if 1 <= image.max() <= 255:
            image = image.astype(np.float32) / 255.0

        # Convert the image into a PyTorch tensor.
        image = torch.from_numpy(image).permute(2, 0, 1)

        # Return the processed image.
        return image

    @torch.jit.ignore()
    def preprocess_input(self, images, return_shapes=False):
        """Preprocesses the input image to the specification of the model.

        This method takes in a set of inputs and preprocesses them into the
        expected format for the `EfficientDetD4` object detection model.
        There are a variety of inputs which are accepted, including images,
        image paths, as well as fully-processed image tensors. The inputs
        are expanded and standardized, then run through a preprocessing
        pipeline which formats them into a single tensor ready for the model.

        Preprocessing steps include normalization, resizing, and converting
        to the channels-first format used by PyTorch models. The output
        of this method will be a single `torch.Tensor`, which has shape
        [N, C, H, W], where `N` is the batch dimension. If only a single
        image is passed, this will have a value of 1.

        This method is largely beneficial when you just want to preprocess
        images into the specification of the model, without getting the output.
        Namely, `predict()` is essentially just a wrapper around this method
        and `forward()`, so you can run this externally and then run `forward()`
        to get the original model outputs, without any extra post-processing.

        Parameters
        ----------
        images : Any
            One of the following formats (and types):
                1. A single image path (str)
                2. A list of image paths (List[str])
                3. A single image (np.ndarray, torch.Tensor)
                4. A list of images (List[np.ndarray, torch.Tensor])
                5. A batched tensor of images (np.ndarray, torch.Tensor)
        return_shapes : bool
            Whether to return the original shapes of the input images.

        Returns
        -------
        A 4-dimensional, preprocessed `torch.Tensor`. If `return_shapes`
        is set to True, it also returns the original shapes of the images.
        """
        images = self._expand_input_images(images)
        shapes = self._get_shapes(images)
        images = torch.stack([self._preprocess_image(image, self._image_size) for image in images], dim=0)
        if return_shapes:
            return images, shapes
        return images

    def _process_detections(self, detections):
        """Post-processes the output detections (boxes, labels) from the model."""
        predictions = []

        # Convert all of the output detections into predictions. This involves
        # selecting the bounding boxes, confidence scores, and classes, and then
        # dropping any of them which do not have a confidence score about the
        # threshold as determined when the class is initialized.
        for d in detections:
            # Extract the bounding boxes, confidence scores,
            # and class labels from the output detections.
            boxes, scores, classes = d[:, :4], d[:, 4], d[:, 5]

            # Only return boxes which are above the confidence threshold.
            valid_indexes = np.where(scores > self._confidence_threshold)[0]
            boxes = boxes[valid_indexes]
            scores = scores[valid_indexes]
            classes = classes[valid_indexes]
            predictions.append({"boxes": boxes, "scores": scores, "classes": classes})

        # Run weighted boxes fusion. For an exact description of how this
        # works, see the paper: https://arxiv.org/pdf/1910.13302.pdf.
        (predicted_bboxes, predicted_class_confidences, predicted_class_labels) = self._wbf(predictions)
        return predicted_bboxes, predicted_class_confidences, predicted_class_labels

    @staticmethod
    def _rescale_bboxes(predicted_bboxes, image_sizes, yxyx=False):
        """Re-scales output bounding boxes to the original image sizes."""
        scaled_boxes = []
        for bboxes, img_dims in zip(predicted_bboxes, image_sizes):
            h, w = img_dims
            if len(bboxes) > 0:
                # Move the device to the CPU.
                if hasattr(bboxes, "cpu"):
                    bboxes = bboxes.cpu()

                # Re-scale the bounding box to the appropriate format.
                scale_ratio = [w / 512, h / 512, w / 512, h / 512]
                scaled = (np.array(bboxes) * scale_ratio).astype(np.int32)

                # Convert the Pascal-VOC (xyxy) format to COCO (xywh).
                if yxyx:
                    y, x = scaled[:, 0], scaled[:, 1]
                else:
                    x, y = scaled[:, 0], scaled[:, 1]
                w, h = scaled[:, 2] - x, scaled[:, 3] - y
                scaled_boxes.append(np.dstack((x, y, w, h)))
                continue

            # Otherwise, there is no prediction for this image.
            scaled_boxes.append(np.array([]))
        return scaled_boxes

    @staticmethod
    def _wbf(predictions):
        """Runs weighted boxes fusion on the output predictions."""
        bboxes, confidences, class_labels = [], [], []

        # Fuse the predictions for each in the batch.
        for prediction in predictions:
            boxes = [(prediction["boxes"] / 512).tolist()]
            scores = [prediction["scores"].tolist()]
            labels = [prediction["classes"].tolist()]

            # Run the actual fusion and update the containers.
            boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, iou_thr=0.44, skip_box_thr=0.43)
            boxes = boxes * (512 - 1)
            bboxes.append(boxes)
            confidences.append(scores)
            class_labels.append(labels.astype(np.int32))
        return bboxes, confidences, class_labels

    @staticmethod
    def _remap_outputs(boxes, labels, confidences):
        """Remaps the outputs to the format described in `predict()`."""
        squeeze = lambda *args: tuple(list(np.squeeze(a).tolist() for a in args))
        return [squeeze(b, l, c) for b, l, c in zip(boxes, labels, confidences)]

    def _to_out(self, tensor: "torch.Tensor") -> "torch.Tensor":
        if isinstance(tensor, dict):
            tensor = tensor["detections"]
        return super()._to_out(tensor)

    @torch.no_grad()
    def predict(self, images):
        """Runs `EfficientNetD4` inference on the input image(s).

        This method is the primary inference method for the model; it
        accepts a set of input images (see `preprocess_input()` for a
        detailed specification on the allowed input parameters), then
        preprocesses them to the model specifications, forward passes
        them through the model, and finally returns the predictions.

        In essence, this method is a wrapper for `forward()` that allows
        for passing a variety of inputs. If, on the other hand, you
        have pre-processed inputs and simply want to forward pass through
        the model without having to spend computational time on what
        is now unnecessary preprocessing, you can simply call `forward()`
        and then run the post-processing as described in this method.

        Parameters
        ----------
        images : Any
            See `preprocess_input()` for the allowed input images.

        Returns
        -------
        A tuple of `n` lists, where `n` is the number of input images.
        Each of the `n` lists will contain three values consisting of
        the bounding boxes, class labels, and prediction confidences
        for the corresponding input image.
        """
        # Process the images and run inference.
        images, shapes = self.preprocess_input(images, return_shapes=True)
        out = self._to_out(self.forward(images))

        # Post-process the output detections.
        boxes, confidences, labels = self._process_detections(out)
        boxes = self._rescale_bboxes(boxes, shapes)
        ret = self._remap_outputs(boxes, labels, confidences)
        return ret[0] if len(ret) == 1 else ret

    def show_prediction(self, image):
        """Shows the output predictions for one input image.

        This method is useful for instantly visualizing the predictions
        for a single input image. It accepts a single input image (or
        any type of valid 'image' input, as described in the method
        `preprocess_input()`), and then runs inference on that input
        image and displays its predictions in a matplotlib window.

        Parameters
        ----------
        image : Any
            See `preprocess_input()` for allowed types of inputs.

        Returns
        -------
        The matplotlib figure containing the image.
        """
        image = self._expand_input_images(image)[0]
        bboxes, labels, _ = self.predict(image)
        if isinstance(labels, int):
            bboxes, labels = [bboxes], [labels]
        return show_image_and_boxes(image, bboxes, labels)

    def load_benchmark(self, dataset, strict=False):
        """Loads a benchmark for the given semantic segmentation dataset.

        This method is used to load pretrained weights for a specific AgML dataset.
        In essence, it serves as a wrapper for `load_state_dict`, directly getting
        the model from its save path in the AWS storage bucket. You can then use the
        `benchmark` property to access the metric value of the benchmark, as well as
        additional training parameters which you can use to train your own models.

        Parameters
        ----------
        dataset : str
            The name of the object detection benchmark to load.
        strict : bool
            Whether to require the same number of classes.

        Notes
        -----
        If the given benchmark has a different number of classes than this input model,
        then the class network will be loaded with random weights, while the remaining
        network weights (backbone, box network, etc.) will use the pretrained weights
        for the benchmark. This can be disabled by setting `strict = True`.
        """
        if source(dataset).tasks.ml != "object_detection":
            raise ValueError(
                f"You are trying to load a benchmark for a "
                f"{source(dataset).tasks.ml} task ({dataset}) "
                f"in an object detection model."
            )

        # Check loading strictness.
        cs = source(dataset).num_classes == self._num_classes
        if strict:
            if not cs:
                raise ValueError(
                    f"You cannot load a benchmark for a dataset '{dataset}' "
                    f"with {source(dataset).num_classes} classes, while your "
                    f"model has {self._num_classes} classes. If you want to, "
                    f"then you need to set `strict = False`."
                )

        # Load the benchmark.
        state = self._get_benchmark(dataset)
        if not strict and not cs:
            self.reset_class_net(source(dataset).num_classes)
            log(
                f"Loading a state dict for {dataset} with "
                f"{source(dataset).num_classes}, while your "
                f"model has {self._num_classes} classes. The "
                f"class network will use random weights."
            )
        self.load_state_dict(state)
        if not strict and not cs:
            self.reset_class_net(self._num_classes)
        self._benchmark = BenchmarkMetadata(dataset)

    def evaluate(self, loader, iou_threshold=0.5, method="accumulate"):
        """Runs a mean average precision evaluation on the given loader.

        This method will loop over the provided `AgMLDataLoader` and compute
        the mean average precision at the provided `iou_threshold`. This can
        be done using two methods. Using the method `average` will compute
        the mean average precision for each individual sample and then average
        over all of the samples, while using the method `accumulate` will
        compute the mean average precision over the entire dataset.

        Parameters
        ----------
        loader : AgMLDataLoader
            An object detection loader with the dataset you want to evaluate.
        iou_threshold : float
            The IoU threshold between a ground truth and predicted bounding
            box at which point they are considered the same.
        method : str
            The method to use, as described above.

        Returns
        -------
        The final calculated mean average precision.
        """
        if not 0 < iou_threshold < 1:
            raise ValueError(f"The `iou_threshold` must be between 0 and 1, got {iou_threshold}.")
        if method not in ["accumulate", "average"]:
            raise ValueError(f"Method must be either `accumulate` or `average`, got {method}.")

        # Construct the mean average precision accumulator and run the calculations.
        mean_ap = MeanAveragePrecision(num_classes=self._num_classes, iou_threshold=iou_threshold)
        bar = tqdm(loader, desc="Calculating Mean Average Precision")
        if method == "average":
            cumulative_maps = []
        for sample in bar:
            image, truth = sample
            true_box, true_label = truth["bbox"], truth["category_id"]
            bboxes, labels, conf = self.predict(image)
            mean_ap.update(
                *(
                    dict(boxes=bboxes, labels=labels, scores=conf),
                    dict(boxes=true_box, labels=true_label),
                )
            )

            # If averaging, then calculate and reset the mAP, otherwise continue.
            if method == "average":
                res = mean_ap.compute()
                cumulative_maps.append(res)  # noqa
                bar.set_postfix({"map": float(res)})
                mean_ap.reset()

        # Compute the final mAP.
        if method == "average":
            result = sum(cumulative_maps) / len(cumulative_maps)
        else:
            result = mean_ap.compute()
        return result

    def run_training(
        self,
        dataset=None,
        *,
        epochs=50,
        metrics=None,
        optimizer=None,
        lr_scheduler=None,
        lr=None,
        batch_size=8,
        loggers=None,
        train_dataloader=None,
        val_dataloader=None,
        test_dataloader=None,
        use_cpu=False,
        save_dir=None,
        experiment_name=None,
        **kwargs,
    ):
        """Trains an object detection model.

        This method can be used to train an object detection model on a given
        dataset (which should be an `AgMLDataLoader`. Alternatively, if you already
        have separate dataloaders for training, validation, and testing, you can
        pass them in as keyword arguments). This method will train the model for
        the given number of epochs, and then return the trained model.

        You can take advantage of keyword arguments to provide additional training
        parameters, e.g., a custom optimizer or optimizer name. If nothing is provided
        for these parameters (see below for an extended list), then defaults are used.

        This method provides a simple interface for training models, but it is not
        a fully-flexible or customizable training loop. If you need more control over
        the training loop, then you should manually define your arguments. Furthermore,
        if you need custom control over the training loop, then you should reimplement
        the training/validation/test loops on your own in the original model class.

        Parameters
        ----------
        model : AgMLModelBase
            The model to train.
        dataset : AgMLDataLoader
            The name of the dataset to use for training. This should be an AgMLDataLoader
            with the data split in the intended splits, and all preprocessing/transforms
            already applied to the loader. This method will automatically figure out the
            splits from the dataloader.
        epochs : int
            The number of epochs to train for.
        metrics : {str, List[str]}
            The metrics to use for training. If none are provided, then the default
            metrics are used (mean average precision). This also happens to be the only
            currently supported metric.
        optimizer : torch.optim.Optimizer
            The optimizer to use for training. If none is provided, then the default
            optimizer is used (AdamW).
        lr_scheduler : torch.optim.lr_scheduler._LRScheduler
            The learning rate scheduler to use for training. If none is provided, then
            no learning rate scheduler is used.
        lr : float
            The learning rate to use for training. If none is provided, then the default
            learning rate is used (0.0002 if num_classes is 1 else 0.0008).
        batch_size : int
            The batch size to use for training. If none is provided, then the default
            batch size is used (8).
        loggers : Any
            The loggers to use for training. If none are provided, then the default
            loggers are used (TensorBoard)

        train_dataloader : torch.utils.data.DataLoader
            The dataloader to use for training. If none is provided, then the dataloader
            is loaded from the dataset.
        val_dataloader : torch.utils.data.DataLoader
            The dataloader to use for validation. If none is provided, then the dataloader
            is loaded from the dataset.
        test_dataloader : torch.utils.data.DataLoader
            The dataloader to use for testing. If none is provided, then the dataloader
            is loaded from the dataset.

        use_cpu : bool
            If True, then the model will be trained on the CPU, even if a GPU is available.
            This is useful for debugging purposes (or if you are on a Mac, where MPS
            acceleration may be buggy).
        save_dir : str
            The directory to save the model and any logs to. If none is provided, then
            the model is saved to the current working directory in a folder which is
            called `agml_training_logs`.
        experiment_name : str
            The name of the experiment. If none is provided, then the experiment name
            is set to a custom format (the task + the dataset + the current date).

        kwargs : dict
            num_workers : int
                The number of workers to use for the dataloaders. If none is provided,
                then the number of workers is set to half of the available CPU cores.

        Returns
        -------
        AgMLModelBase
            The trained model with the best loaded weights. This model can be used for
            inference, or for further training.
        """
        self.switch_train()

        from agml.models.training.basic_trainers import train_detection

        return train_detection(
            self,
            dataset=dataset,
            epochs=epochs,
            metrics=metrics,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            lr=lr,
            batch_size=batch_size,
            loggers=loggers,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            use_cpu=use_cpu,
            save_dir=save_dir,
            experiment_name=experiment_name,
            **kwargs,
        )

    def _prepare_for_training(self, metrics=(), optimizer=None, **kwargs):
        """Prepares the model for training."""

        # Initialize the metrics.
        self.map = None
        if len(metrics) > 0:
            for metric in metrics:
                # So far, only the mAP metric is supported.
                if isinstance(metric, str):
                    if metric.lower() not in ["ap", "map"]:
                        raise ValueError(
                            "Unfortunately, the only currently supported metric "
                            "is mean average precision (use 'ap' or 'map' "
                            "to enable mean average precision)."
                        )

                    self.map = MeanAveragePrecision(
                        num_classes=self._num_classes,
                        iou_threshold=kwargs.get("iou_threshold", 0.5),
                    )

        # Initialize the optimizer/learning rate scheduler.
        if isinstance(optimizer, str):
            optimizer_class = optimizer.capitalize()
            if not has_func(torch.optim, optimizer_class):
                raise ValueError(
                    f"Expected a valid optimizer name, but got '{optimizer_class}'. "
                    f"Check `torch.optim` for a list of valid optimizers."
                )

            optimizer = getattr(torch.optim, optimizer_class)(
                self.parameters(),
                lr=kwargs.get("lr", 0.0002 if self._num_classes == 1 else 0.0008),
            )
        elif isinstance(optimizer, torch.optim.Optimizer):
            pass  # nothing to do
        else:
            raise TypeError(f"Expected an optimizer name or a torch optimizer, but got '{type(optimizer)}'.")

        scheduler = kwargs.get("lr_scheduler", None)
        if scheduler is not None:
            # No string auto-initialization, the LR scheduler must be pre-configured.
            if isinstance(scheduler, str):
                raise ValueError(
                    f"If you want to use a learning rate scheduler, you must initialize "
                    f"it on your own and pass it to the `lr_scheduler` argument. "
                )
            elif not isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
                raise TypeError(f"Expected a torch LR scheduler, but got '{type(scheduler)}'.")

        self._optimization_parameters = {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def configure_optimizers(self):
        opt = self._optimization_parameters["optimizer"]
        scheduler = self._optimization_parameters["lr_scheduler"]
        if scheduler is None:
            return opt
        return [opt], [scheduler]

    def training_step(self, batch, batch_idx, *args, **kwargs):
        images, annotations, _ = batch
        losses = self(images, annotations)
        return losses

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        images, annotations, targets = batch
        outputs = self(images, annotations)
        detections = outputs["detections"]

        # Calculate the mean average precision.
        if not self.trainer.sanity_checking and self.map is not None:
            boxes, confidences, labels = self._process_detections(self._to_out(detections))
            if hasattr(boxes, "cpu"):
                boxes = boxes.cpu()
            if hasattr(annotations["bbox"], "cpu"):
                annotations["bbox"] = annotations["bbox"].cpu()
            boxes = self._rescale_bboxes(boxes, [[512, 512]] * len(images))
            annotations["bbox"] = self._rescale_bboxes(
                annotations["bbox"],
                [
                    [
                        512,
                        512,
                    ]
                ]
                * len(images),
                yxyx=True,
            )

            for pred_box, pred_label, pred_conf, true_box, true_label in zip(
                boxes, labels, confidences, annotations["bbox"], annotations["cls"]
            ):
                metric_update_values = (
                    dict(
                        boxes=self._to_out(torch.tensor(pred_box, dtype=torch.float32)),
                        labels=self._to_out(torch.tensor(pred_label, dtype=torch.int32)),
                        scores=self._to_out(torch.tensor(pred_conf)),
                    ),
                    dict(
                        boxes=self._to_out(torch.tensor(true_box, dtype=torch.float32)),
                        labels=self._to_out(torch.tensor(true_label, dtype=torch.int32)),
                    ),
                )
                self.map.update(*metric_update_values)

                # Log the MAP values.
                map_ = self.map.compute().detach().cpu().numpy().item()
                self.log("val_map", map_, prog_bar=True, logger=True, sync_dist=True)

        self.log("val_loss", outputs["loss"], prog_bar=True, logger=True, sync_dist=True)
        return outputs["loss"]

    def on_validation_epoch_end(self):
        if self.map is not None:
            self.map.reset()

    def test_step(self, batch, batch_idx, *args, **kwargs):
        images, annotations, targets = batch
        outputs = self(images, annotations)

        # Calculate the mean average precision.
        if not self.trainer.sanity_checking and self.map is not None:
            boxes, confidences, labels = self._process_detections(self._to_out(outputs["detections"]))
            if hasattr(boxes, "cpu"):
                boxes = boxes.cpu()
            if hasattr(annotations["bbox"], "cpu"):
                annotations["bbox"] = annotations["bbox"].cpu()
            boxes = self._rescale_bboxes(boxes, [[512, 512]] * len(images))
            annotations["bbox"] = self._rescale_bboxes(
                annotations["bbox"],
                [
                    [
                        512,
                        512,
                    ]
                ]
                * len(images),
                yxyx=True,
            )

            for pred_box, pred_label, pred_conf, true_box, true_label in zip(
                boxes, labels, confidences, annotations["bbox"], annotations["cls"]
            ):
                metric_update_values = (
                    dict(
                        boxes=self._to_out(torch.tensor(pred_box, dtype=torch.float32)),
                        labels=self._to_out(torch.tensor(pred_label, dtype=torch.int32)),
                        scores=self._to_out(torch.tensor(pred_conf)),
                    ),
                    dict(
                        boxes=self._to_out(torch.tensor(true_box, dtype=torch.float32)),
                        labels=self._to_out(torch.tensor(true_label, dtype=torch.int32)),
                    ),
                )
                self.map.update(*metric_update_values)

                # Log the MAP values.
                map_ = self.map.compute().detach().cpu().numpy().item()
                self.log("test_map", map_, prog_bar=True, logger=True, sync_dist=True)

        self.log("test_loss", outputs["loss"], prog_bar=True, logger=True, sync_dist=True)
        return outputs["loss"]
