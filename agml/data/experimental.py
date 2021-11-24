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

"""
Experimental data utilities that are in development.
"""
import functools

import numpy as np


__all__ = ['generate_keras_segmentation_dual_transform']


def generate_keras_segmentation_dual_transform(*layers):
    """Generates a `dual_transform` pipeline from Keras preprocessing layers.

    This method takes in Keras preprocessing layers and generates a
    transformation pipeline for the `dual_transform` argument in
    *semantic segmentation* loaders, which applies the transform in the
    same fashion to both the image and annotation.

    This is due to the fact that TensorFlow has its operation-level
    random states different than its module-level random state, so
    the layers need to have their seeds manually set in order to work.

    In essence, for each of the preprocessing layers passed, this
    method conducts the following operations:

    > def preprocessing_transform(image, annotation):
    >    layer = functools.partial(KerasPreprocessingLayer, **kwargs)
    >    seed = np.random.randint(BUFFER_SIZE) # up to sys.maxsize
    >    image = layer(image, seed = seed)
    >    annotation = layer(annotation, seed = seed)
    >    return image, annotation

    It then repeats this transform for all of the preprocessing layers
    passed, and returns a method which has this behavior wrapped into
    it and can perform it when the preprocessing is actually conducted.

    Parameters
    ----------
    layers : Any
       Either a Sequential model with preprocessing layers, or a
       set of instantiated preprocessing layers.

    Returns
    -------
    """
    import tensorflow as tf
    if len(layers) == 1:
        if isinstance(layers[0], tf.keras.Sequential):
            layers = layers[0].layers

    # These methods perform the behavior indicated in the
    # code snippet above (for each of the layers given).
    def _single_preprocessing_layer_base(layer_, build_dict):
        def _internal(image, annotation, seed):
            instantiated_layer = functools.partial(layer_, **build_dict)
            seed_update = {}
            if seed is not None:
                seed_update['seed'] = seed
            image = instantiated_layer(**seed_update)(image)
            annotation = instantiated_layer(**seed_update)(annotation)
            return image, annotation
        return _internal

    preprocessing_methods, use_seeds = [], []
    for layer in layers:
        config = layer.get_config()
        if 'seed' in config:
            config.pop('seed')
            use_seeds.append(True)
        else:
            use_seeds.append(False)
        preprocessing_methods.append(
            _single_preprocessing_layer_base(layer.__class__, config))

    def _execute_preprocessing(layers_, use_seeds_):
        def _execute(image, annotation):
            for p_layer, seed_ in zip(layers_, use_seeds_):
                seed = np.random.randint(2147483647) if seed_ else None
                image, annotation = p_layer(image, annotation, seed = seed)
            return image, annotation
        return _execute
    return _execute_preprocessing(preprocessing_methods, use_seeds)







