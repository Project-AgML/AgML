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

import torch
import numpy as np
import albumentations as A

from ensemble_boxes.ensemble_boxes_wbf import weighted_boxes_fusion

from effdet import (
    create_model_from_config,
    get_efficientdet_config,
    DetBenchPredict,
    DetBenchTrain
)

from agml.models.base import AgMLModelBase
from agml.models.tools import auto_move_data
from agml.data.public import source
from agml.backend.tftorch import is_array_like
from agml.viz.boxes import visualize_image_and_boxes


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

    This model can be subclassed in order to run a full training job; the
    actual transfer `EfficientDetD4` model can be accessed through the
    parameter `model`, and you'll need to implement methods like `training_step`,
    `configure_optimizers`, etc. See PyTorch Lightning for more information.
    """
    serializable = frozenset(("model", "confidence_threshold", "source"))
    state_override = frozenset(("model", ))

    def __init__(self, dataset = None, conf_threshold = 0.3, **kwargs):
        # Initialize the base modules.
        super(DetectionModel, self).__init__()

        # If being initialized by a subclass, then don't do any of
        # model construction logic (since that's already been done).
        if not kwargs.get('model_initialized', False):
            # Construct the network and load in pretrained weights.
            self._confidence_threshold = conf_threshold
            self._source = source(dataset)
            self.model = self._construct_sub_net(dataset)

        # Filter out unnecessary warnings.
        warnings.filterwarnings(
            'ignore', category = UserWarning, module = 'ensemble_boxes')
        warnings.filterwarnings(
            'ignore', category = UserWarning, module = 'effdet.bench') # noqa

    @auto_move_data
    def forward(self, batch):
        return self.model(batch)

    @staticmethod
    def _construct_sub_net(dataset):
        cfg = get_efficientdet_config('tf_efficientdet_d4')
        cfg.update({"image_size": (512, 512)})
        model = create_model_from_config(
            cfg, pretrained = False,
            num_classes = source(dataset).num_classes)
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
    def original(self): # override for detection models.
        return self.model.model

    @torch.jit.ignore()
    def reset_class_net(self, num_classes = 1):
        """Reconfigures the output class net for a new number of classes.

        Parameters
        ----------
        num_classes : int
            The number of classes to reconfigure the output net to use.
        """
        if num_classes != self._source.num_classes:
            self.model.model.reset_head(num_classes = num_classes)

    @staticmethod
    def _preprocess_image(image):
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
        if is_array_like(image) and hasattr(image, 'numpy'):
            image = image.numpy()

        # Add a channel dimension for grayscale imagery.
        if image.ndim == 2:
            image = np.expand_dims(image, axis = -1)

        # If the image is already in channels-first format, convert
        # it back temporarily until preprocessing has concluded.
        if image.shape[0] <= 3:
            image = np.transpose(image, (1, 2, 0))

        # Resize the image to ImageNet standards.
        h = w = 512
        rz = A.Resize(height = h, width = w)
        if image.shape[0] != h or image.shape[1] != w:
            image = rz(image = image)['image']

        # Normalize the image to ImageNet standards.
        if 1 <= image.max() <= 255:
            image = image.astype(np.float32) / 255.

        # Convert the image into a PyTorch tensor.
        image = torch.from_numpy(image).permute(2, 0, 1)

        # Return the processed image.
        return image

    @torch.jit.ignore()
    def preprocess_input(self, images, return_shapes = False):
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
        images = torch.stack(
            [self._preprocess_image(
                image) for image in images], dim = 0)
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
        (predicted_bboxes,
         predicted_class_confidences,
         predicted_class_labels) = self._wbf(predictions)
        return predicted_bboxes, predicted_class_confidences, predicted_class_labels

    @staticmethod
    def _rescale_bboxes(predicted_bboxes, image_sizes):
        """Re-scales output bounding boxes to the original image sizes."""
        scaled_boxes = []
        for bboxes, img_dims in zip(predicted_bboxes, image_sizes):
            h, w = img_dims
            if len(bboxes) > 0:
                # Re-scale the bounding box to the appropriate format.
                scale_ratio = [w / 512, h / 512, w / 512, h / 512]
                scaled = (np.array(bboxes) * scale_ratio).astype(np.int32)

                # Convert the Pascal-VOC (xyxy) format to COCO (xywh).
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
            boxes, scores, labels = weighted_boxes_fusion(
                boxes, scores, labels,
                iou_thr = 0.44, skip_box_thr = 0.43)
            boxes = boxes * (512 - 1)
            bboxes.append(boxes)
            confidences.append(scores)
            class_labels.append(labels.astype(np.int32))
        return bboxes, confidences, class_labels

    @staticmethod
    def _remap_outputs(boxes, labels, confidences):
        """Remaps the outputs to the format described in `predict()`."""
        squeeze = lambda *args: tuple(list(
            np.squeeze(a).tolist() for a in args))
        return [squeeze(b, l, c)
                for b, l, c in zip(boxes, labels, confidences)]

    @staticmethod
    def _to_out(tensor: "torch.Tensor") -> "torch.Tensor":
        if isinstance(tensor, dict):
            tensor = tensor['detections']
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
        images, shapes = self.preprocess_input(images, return_shapes = True)
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
        return visualize_image_and_boxes(image, bboxes, labels)


