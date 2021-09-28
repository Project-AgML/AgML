import os
import re
import json
import functools

@functools.lru_cache(maxsize = None)
def load_public_sources():
    """Loads the public data sources JSON file."""
    with open(os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            '_assets/public_datasources.json')) as f:
        return json.load(f)

def to_camel_case(s):
    """Converts a given string `s` to camel case."""
    s = re.sub(r"(_|-)+", " ", s).title().replace(" ", "") # noqa
    return ''.join(s)

def resolve_list_value(l):
    """Determines whether a list contains one or multiple values."""
    if len(l) == 1:
        return l[0]
    return l

