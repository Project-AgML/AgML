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

import functools

import numpy as np


class seed_context(object):
    """Creates a context with a custom random seed, then resets it.

    This allows for setting a custom seed (for reproducibility) in a
    context, then resetting the original state after exiting, to
    prevent interfering with the program execution.
    """

    def __init__(self, seed):
        self._seed = seed
        self._prev_state = None

    def __enter__(self):
        self._prev_state = np.random.get_state()
        np.random.seed(self._seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        np.random.set_state(self._prev_state)
        self._prev_state = None

    def reset(self):
        np.random.seed(self._seed)


_RANDOM_STATE_DOC = """random_state : int
\t    An integer representing a random seed to run the method using."""


def inject_random_state(f):
    """Runs a method inside of a context if a random seed is provided."""
    if "{random_state}" in f.__doc__:
        f.__doc__ = f.__doc__.format(random_state = _RANDOM_STATE_DOC)
    else:
        f.__doc__ += _RANDOM_STATE_DOC

    @functools.wraps(f)
    def _run(*args, random_state = None, **kwargs):
        # If no random state is provided, run the method as normal.
        if random_state is None:
            return f(*args, **kwargs)

        # Otherwise, run the method in a context with the random seed.
        if not isinstance(random_state, int):
            raise TypeError(
                f"Expected `random_state` argument to be "
                f"an integer, instead got ({random_state}) "
                f"of type ({type(random_state)}).")

        with seed_context(random_state):
            result = f(*args, **kwargs)
        return result
    return _run

