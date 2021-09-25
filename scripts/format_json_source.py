"""
Convenience script to reformat the public data source
JSON file in a nicer-looking format (for readability).
"""
import os
import json

SOURCE_PATH = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), '_assets', 'public_datasources.json')

with open(SOURCE_PATH, 'r') as f:
    contents = json.load(f)
with open(SOURCE_PATH, 'w') as f:
    json.dump(contents, f, indent = 4)

