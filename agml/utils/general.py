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

def resolve_tuple_pair(inp1, inp2, custom_error = None):
    """Determines whether `inp1` contains two values or
    they are distributed amongst `inp1` and `inp2`."""
    if isinstance(inp1, (list, tuple)) and not inp2:
        try:
            image, mask = inp1
        except ValueError:
            if custom_error is not None:
                raise ValueError(custom_error)
            else:
                raise ValueError(
                    "Expected either a tuple with two values "
                    "or two values across two arguments.")
        else:
            return inp1
    return inp1, inp2
