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

import difflib
import functools
import json
import os
import shutil
import sys

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


@functools.lru_cache(maxsize=None)
def load_public_sources() -> dict:
    """Loads the public data sources JSON file."""
    with open(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "_assets/public_datasources.json",
        )
    ) as f:
        return json.load(f)


@functools.lru_cache(maxsize=None)
def load_citation_sources() -> dict:
    """Loads the citation sources JSON file."""
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "_assets/source_citations.json")) as f:
        return json.load(f)


@functools.lru_cache(maxsize=None)
def load_model_benchmarks() -> dict:
    """Loads the citation sources JSON file."""
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "_assets/model_benchmarks.json")) as f:
        return json.load(f)


@functools.lru_cache(maxsize=None)
def load_detector_benchmarks() -> dict:
    """Loads the citation sources JSON file."""
    with open(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "_assets/detector_benchmarks.json",
        )
    ) as f:
        return json.load(f)


def maybe_you_meant(name, msg, source=None) -> str:
    """Suggests potential correct spellings for an invalid name."""
    source = source if source is not None else load_public_sources().keys()
    suggestion = difflib.get_close_matches(name, source)
    if len(suggestion) == 0:
        return msg
    return msg + f" Maybe you meant: '{suggestion[0]}'?"


def copyright_print(name, location=None):
    """Prints out license/copyright info after a dataset download."""
    content = load_citation_sources()[name]
    license_info = content["license"]  # noqa
    citation = content["citation"]

    # Construct the title text
    title_text = Text(f"Dataset: {name}", style="bold cyan")

    if location is not None:
        location_text = Text.assemble(("You have just downloaded ", "bold cyan"), (name, "green"))
    else:
        location_text = Text.assemble(("Citation information for", "bold cyan"), (name, "green"))

    _LICENSE_TO_URL = {
        "CC BY-SA 4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
        "CC BY-SA 3.0": "https://creativecommons.org/licenses/by-sa/3.0/",
        "CC BY-NC 3.0": "https://creativecommons.org/licenses/by-nc/3.0/",
        "CC BY-NC SA 3.0": "https://creativecommons.org/licenses/by-nc/3.0/",
        "CC BY-NC-SA 4.0": "https://creativecommons.org/licenses/by-nc/4.0/",
        "MIT": "https://opensource.org/licenses/MIT",
        "GPL-3.0": "https://opensource.org/licenses/GPL-3.0",
        "US Public Domain": "https://www.usa.gov/government-works",
        "CC0: Public Domain": "https://creativecommons.org/publicdomain/zero/1.0/",
        "Apache 2.0": "https://www.apache.org/licenses/LICENSE-2.0",
        'CC BY-NC 4.0': 'https://creativecommons.org/licenses/by-nc/4.0/',
        'CC BY-NC-SA 4.0': 'https://creativecommons.org/licenses/by-nc-sa/4.0/',
        'CC BY 4.0': 'https://creativecommons.org/licenses/by/4.0/deed.en'}
    }

    # Create license message
    if not license_info:  # Handle empty license
        license_msg = Text("License: None specified", style="yellow")
    else:
        license_msg = Text.assemble(("This dataset is licensed under ", "bold cyan"), (f"{license_info}", "green"))
        if license_info in _LICENSE_TO_URL:
            license_more_info = Text(" To learn more about this license, visit ", style="bold cyan")
            license_url = Text(f"{_LICENSE_TO_URL[license_info]}", justify="center", style="white")
        else:
            license_more_info = ""
            license_url = ""

    # Create citation message
    if citation == "":
        citation_msg = Text("This dataset has no associated citation.", style="yellow")
        citation_url = ""
    else:
        citation_msg = (Text("When using this dataset, please cite the following: \n", style="bold cyan"),)
        citation_url = Text(citation, justify="center", style="white")

    # Dataset documentation message
    docs = load_public_sources()[name]["docs_url"]
    docs_msg = Text("You can find additional information about this dataset at: ", style="bold  cyan")
    docs_url = Text(docs, justify="center", style="white")

    combined_message = Text.assemble(
        title_text,
        "\n\n",
        location_text,
        "\n\n",
        license_msg,
        " ",
        license_more_info,
        license_url,
        "\n\n",
        citation_msg,
        citation_url,
        "\n\n",
        docs_msg,
        docs_url,
    )
    console = Console()

    # Create and print the rich Panel
    panel = Panel(
        combined_message,
        title="Copyright, Citation, and Documenation Information",
        subtitle=title_text,
        border_style="bright_yellow",
        highlight=True,
        expand=False,  # Prevent unnecessary whitespace
    )
    console.print(panel)

    # Instructions on how to reprint (using rich Text)
    instructions = Text.assemble(
        ("\nThis message will ",),
        (
            "not ",
            "bold",
        ),
        ("be automatically shown again. To view this message again,  in an AgMLDataLoader run "),
        ("`loader.info.citation_summary()` "),
        (" Otherwise, you can use `agml.data.source(<dataset_name>).citation_summary()`.",),
    )
    warning_panel = Panel(
        instructions,
        title="Note",
        border_style="yellow",
        expand=False,
        highlight=True,
    )
    console.print(warning_panel)

    if location is not None:
        console.print(f"\n [bold] You can find your dataset at {location}.")
