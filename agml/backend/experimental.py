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
"""Experimental features in AgML, these may be added or removed permanently."""

from agml.framework import AgMLSerializable


__all__ = ['allow_nested_data_splitting']


class AgMLExperimentalFeatureWrapper(AgMLSerializable):
    """Stores all of the feature values."""
    _NESTED_SPLITTING = False

    def __init__(self):
        raise ValueError("This class should not be instantiated!")

    @classmethod
    def allow_nested_data_splitting(cls, value: bool) -> None:
        """Enables/disables nested splitting of `AgMLDataLoader`s.

        This method can be used to either enable or disable a feature which
        allows an sub-AgMLDataLoader which has been split to a custom data
        split to be split again. This is an experimental feature, which
        would allow for multiple levels of nested splits (e.g., you could
        have a `loader.train_data.val_data.test_data`).

        Parameters
        ----------
        value : bool
            Whether to enable or disable the feature.

        Notes
        -----
        This method must be called at the start of each script in order
        to function, otherwise it will default to `False`.
        """
        if not isinstance(value, bool):
            raise TypeError("Expected either True or False.")
        cls._NESTED_SPLITTING = value

    @classmethod
    def nested_splitting(cls):
        return cls._NESTED_SPLITTING


# While the `AgMLExperimentalFeatureWrapper` class controls the actual values,
# we expose each of its toggle methods as part of the `agml.backend` API.
allow_nested_data_splitting = AgMLExperimentalFeatureWrapper.allow_nested_data_splitting

