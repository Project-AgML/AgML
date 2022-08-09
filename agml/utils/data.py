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
import sys
import json
import shutil
import difflib
import functools


@functools.lru_cache(maxsize = None)
def load_public_sources() -> dict:
    """Loads the public data sources JSON file."""
    with open(os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            '_assets/public_datasources.json')) as f:
        return json.load(f)


@functools.lru_cache(maxsize = None)
def load_citation_sources() -> dict:
    """Loads the citation sources JSON file."""
    with open(os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            '_assets/source_citations.json')) as f:
        return json.load(f)


@functools.lru_cache(maxsize = None)
def load_model_benchmarks() -> dict:
    """Loads the citation sources JSON file."""
    with open(os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            '_assets/model_benchmarks.json')) as f:
        return json.load(f)


def maybe_you_meant(name, msg, source = None) -> str:
    """Suggests potential correct spellings for an invalid name."""
    source = source if source is not None \
                    else load_public_sources().keys()
    suggestion = difflib.get_close_matches(name, source)
    if len(suggestion) == 0:
        return msg
    return msg + f" Maybe you meant: '{suggestion[0]}'?"


def copyright_print(name, location = None):
    """Prints out license/copyright info after a dataset download."""
    content = load_citation_sources()[name]
    license = content['license'] # noqa
    citation = content['citation']

    def _bold(msg): # noqa
        return '\033[1m' + msg + '\033[0m'

    if location is None:
        first_msg = "Citation information for " + _bold(name) + ".\n"
    else:
        first_msg = "You have just downloaded " + _bold(name) + ".\n"

    _LICENSE_TO_URL = {
        'CC BY-SA 4.0': 'https://creativecommons.org/licenses/by-sa/4.0/',
        'CC BY-SA 3.0': 'https://creativecommons.org/licenses/by-sa/3.0/',
        'CC BY-NC 3.0': 'https://creativecommons.org/licenses/by-nc/3.0/',
        'CC BY-NC SA 3.0': 'https://creativecommons.org/licenses/by-nc/3.0/',
        'MIT': 'https://opensource.org/licenses/MIT',
        'GPL-3.0': 'https://opensource.org/licenses/GPL-3.0',
        'US Public Domain': 'https://www.usa.gov/government-works'}
    if license == '':
        license_msg = "This dataset has " \
                      + _bold("no license") + ".\n"
    else:
        license_msg = "This dataset is licensed under the " \
                      + _bold(license) + " license.\n"
        license_msg += "To learn more, visit: " \
                       + _LICENSE_TO_URL[license] + "\n"

    if citation == '':
        citation_msg = "This dataset has no associated citation."
    else:
        citation_msg = "When using this dataset, please cite the following:\n\n"
        citation_msg += citation

    docs = load_public_sources()[name]['docs_url']
    docs_msg = ("\nYou can find additional information about "
                "this dataset at:\n" + docs)

    columns = shutil.get_terminal_size((80, 24)).columns
    max_print_length = max(min(
        columns, max([len(i) for i in [
            *citation_msg.split('\n'), *license_msg.split('\n')]])), columns)
    print('\n' + '=' * max_print_length)
    print(first_msg)
    print(license_msg)
    print(citation_msg)
    print(docs_msg)
    print("\nThis message will " + _bold("not") + " be automatically shown\n" 
          "again. To view this message again, in an AgMLDataLoader\n" +
          "run `loader.info.citation_summary()`. Otherwise, you\n" +
          "can use `agml.data.source(<name>).citation_summary().`")

    if location is not None:
        print(f"\nYou can find your dataset at {location}.")
    print('=' * max_print_length)
    sys.stdout.flush()





