import json
from importlib import resources
from textwrap import dedent

from jinja2 import StrictUndefined
from jinja2.sandbox import SandboxedEnvironment

# Check whether any dataset has been added that needs information updated (we do this
# by checking for any datasets in the `public_datasources.json` file that don't have
# corresponding information for the dataset statistics or other properties like shape).


def _render_dataset() -> str:
    public_datasets = json.load(resources.files("agml._assets").joinpath("public_datasources.json").open("r"))

    public_datasets = {
        "datasets": sorted(
            [(value | {"dataset_name": key}) for key, value in public_datasets.items()],
            key=lambda dep: str(dep["dataset_name"]).lower(),
        )
    }
    # return public_datasets

    template_datasets = dedent(
        """

        {% macro dat_line(dat) -%}
        [{{dat.dataset_name}}](datasets/{{dat.dataset_name}}.html) | {{ dat.ml_task }} | {{ dat.n_images }}  | {{ dat.docs_url }} |
        {%- endmacro %}

        {% if datasets -%}

        ## Public Dataset Listing

        | Dataset | Task | Number of Images | Documentation|
        | :--- | :---: |-----------:| :----|
        {% for dataset in datasets -%}
        {{ dat_line(dataset) }}
        {% endfor %}
        {% endif %}
        """,
    )
    jinja_env = SandboxedEnvironment(undefined=StrictUndefined)
    return jinja_env.from_string(template_datasets).render(**public_datasets)


print(_render_dataset())
