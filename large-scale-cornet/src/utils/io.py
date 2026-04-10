# src/utils/io.py

"""
I/O utility functions for JSON serialization and file operations
"""

import json
import os
import numpy as np


def convert_to_serializable(obj):
    """
    Recursively convert objects that are not serializable by the default JSON encoder,
    with special handling for numpy arrays.

    Description:
      This function traverses dictionaries, lists, or tuples and converts any numpy arrays
      to lists. It returns the converted object which can be safely serialized to JSON.

    Inputs:
      - obj (any): The object to convert (can be a numpy array, dict, list, tuple, or other types).

    Returns:
      - A JSON-serializable version of the input object.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj


def save_json(data, filename):
    """
    Save a Python dictionary or list to a JSON file.

    Description:
      Serializes the input data to JSON and writes it to the specified file. If the target
      directory does not exist, it is created automatically.

    Inputs:
      - data (dict or list): The Python object to be serialized and saved.
      - filename (str): The path to the output JSON file.

    Returns:
      - None
    """
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

