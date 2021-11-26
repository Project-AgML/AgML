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

from agml.backend.tftorch import tf
from agml.data.managers.training import TrainState
from agml.data.object import DataObject
from agml.utils.logging import log


class TFExporter(object):
    """Exports an `AgMLDataLoader` as a `tf.data.Dataset`.

    This class manages the methods and conversions between `AgMLDataLoader`
    parameters, managers, and methods, into TensorFlow native methods.
    """
    def __init__(self, task, builder):
        self._task = task
        self._builder = builder

    def digest_transforms(self, transforms, resizing):
        """Parses the transforms for the `AgMLDataLoader`."""
        self._transforms = {
            k: state for k, state in transforms.items()}
        self._size = resizing if resizing is not None else (512, 512)

    def assign_state(self, state):
        """Updates the training state for the `tf.data.Dataset`."""
        if state in [TrainState.EVAL,
                     TrainState.EVAL_TF,
                     TrainState.EVAL_TORCH,
                     TrainState.FALSE]:
            self._state = 'eval'
        else:
            self._state = 'train'

    def _build_from_contents_by_type(self, builder):
        """Parses the provided mapping into a valid set of contents."""
        if self._task == 'image_classification':
            ds = self._build_image_classification(builder)
            return ds.map(self._image_classification_load)
        elif self._task == 'semantic_segmentation':
            ds = self._build_semantic_segmentation(builder)
            return ds.map(self._semantic_segmentation_load)
        else:
            ds = self._build_object_detection(builder)
            return ds.map(self._object_detection_load)

    def _apply_resizing_by_type(self, ds):
        """Applies resizing based on the task."""
        if self._task == 'image_classification':
            return ds.map(self._image_classification_resize)
        elif self._task == 'semantic_segmentation':
            return ds.map(self._semantic_segmentation_resize)
        else:
            return ds.map(self._object_detection_resize)

    def build(self, batch_size = None):
        """Builds the `tf.data.Dataset` using the provided parameters."""
        # Construct the dataset from the contents.
        ds = self._build_from_contents_by_type(self._builder)

        # Apply the digested transforms and resizing.
        ds = self._apply_resizing_by_type(ds)
        if self._state != 'eval':
            # No transforms for object detection, since it is near impossible
            # for TensorFlow's graph mode to use COCO JSON dictionaries.
            if self._task == 'object_detection':
                if len(self._transforms) != 0:
                    log("Got transforms when exporting an `AgMLDataLoader`"
                        "to a `tf.data.Dataset`. These transforms will not be "
                        "applied. To use transforms in TensorFlow, use the "
                        "`as_keras_sequence()` method instead.")
            else:
                tfm = self._apply_transforms
                if len(self._transforms) != 0:
                    ds = ds.map(tfm)

        # Apply batching and prefetching, then return the dataset.
        if batch_size is not None:
            ds = ds.batch(batch_size)
            ds = ds.prefetch(batch_size)
        return ds

    # The following methods are used to parse the input
    # contents into valid methods for the loaders.

    @staticmethod
    def _build_image_classification(builder):
        images, labels = builder.export_contents(
            export_format = 'arrays')
        images, labels = tf.constant(images), tf.constant(labels)
        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        return ds.shuffle(len(images))

    @staticmethod
    def _build_semantic_segmentation(builder):
        images, masks = builder.export_contents(
            export_format = 'arrays')
        images, masks = tf.constant(images), tf.constant(masks)
        ds = tf.data.Dataset.from_tensor_slices((images, masks))
        return ds.shuffle(len(images))

    @staticmethod
    def _build_object_detection(builder):
        images, annotations = builder.export_contents(
            export_format = 'arrays')
        images = tf.constant(images)
        processed_annotations = [
            DataObject._parse_coco(a) for a in annotations]
        features = {'bbox': [], 'category_id': [], 'area': [],
                    'image_id': [], 'iscrowd': [], 'segmentation': []}
        for a_set in processed_annotations:
            for feature in features.keys():
                features[feature].append(a_set[feature]) # noqa
        for feature in features.keys():
            features[feature] = tf.ragged.constant(features[feature])
        feature_ds = tf.data.Dataset.from_tensor_slices(features)
        ds = tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(images), feature_ds))
        return ds.shuffle(len(images))

    # The following methods are used for loading images and
    # annotations for each of the different types of tasks.

    @staticmethod
    @tf.function
    def _image_classification_load(image, label):
        image = tf.cast(tf.image.decode_jpeg(
            tf.io.read_file(image)), tf.float32) / 255.
        return image, tf.convert_to_tensor(label)

    @staticmethod
    @tf.function
    def _semantic_segmentation_load(image, mask):
        image = tf.cast(tf.image.decode_jpeg(
            tf.io.read_file(image)), tf.float32) / 255.
        mask = tf.image.decode_jpeg(tf.io.read_file(mask))
        return image, mask

    @staticmethod
    @tf.function
    def _object_detection_load(image, coco):
        image = tf.cast(tf.image.decode_jpeg(
            tf.io.read_file(image)), tf.float32) / 255.
        ret_coco = coco.copy()
        for key in coco.keys():
            try: ret_coco[key] = coco[key].to_tensor()
            except: pass
        return image, ret_coco

    # The following methods apply resizing to the data.

    def _image_classification_resize(self, image, label):
        image = (tf.image.resize(
            image, self._size, method = 'nearest'), tf.float32)
        return image, label

    def _semantic_segmentation_resize(self, image, mask):
        image = tf.cast(tf.image.resize(
            image, self._size, method = 'nearest'), tf.float32)
        mask = tf.cast(tf.image.resize(
            mask, self._size, method = 'nearest'), tf.float32)
        return image, mask

    def _object_detection_resize(self, image, coco):
        # Helper for the `tf.py_function` for object detection.
        def _resize_image_and_bboxes(image, coco_boxes):
            nonlocal size
            y_scale, x_scale = image.shape[0:2]
            stack_boxes = tf.stack(
                [coco_boxes[:, 0] / x_scale,
                 coco_boxes[:, 1] / y_scale,
                 coco_boxes[:, 2] / x_scale,
                 coco_boxes[:, 3] / y_scale], axis = -1)
            image = tf.cast(tf.image.resize(
                image, size), tf.float32)
            y_new, x_new = image.shape[0:2]
            new_stack = tf.cast(tf.stack(
                [stack_boxes[:, 0] * x_new,
                 stack_boxes[:, 1] * y_new,
                 stack_boxes[:, 2] * x_new,
                 stack_boxes[:, 3] * y_new], axis = -1
            ), tf.int32)
            areas = new_stack[:, 2] * new_stack[:, 3]
            return image, new_stack, areas

        # The actual resizing can't take place in graph mode, so we
        # dispatch to a `tf.py_function` to do the resizing, then
        # re-assign the values back to the COCO JSON dictionary.
        size = self._size
        image, ret_coco_boxes, ret_areas = tf.py_function(
            _resize_image_and_bboxes,
            [image, coco['bbox']],
            [tf.float32, tf.int32, tf.int32])
        coco['bbox'] = ret_coco_boxes
        coco['area'] = ret_areas
        return image, coco

    # The following method manages the application of transforms.

    def _apply_transforms(self, image, annotation):
        # Helper for the `tf.py_function` for most transforms.
        def _py_apply(img, ann):
            nonlocal transforms
            img, ann = img.numpy(), ann.numpy()
            for key, state in transforms.items():
                if key == 'transform':
                    for t in state:
                        img = t(img)
                elif key == 'target_transform':
                    for t in state:
                        ann = t(ann)
                else:
                    for t in state:
                        img, ann = t(img, ann)

            return img, ann

        # The actual transforming can't take place in graph mode
        # (in most cases), so we dispatch and reassign.
        transforms = self._transforms
        image, annotation = tf.py_function(
            _py_apply,
            [image, annotation],
            [tf.float32, tf.int32])
        return image, annotation









