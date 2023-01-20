import numpy as np
import os


def read_file(file_path):
    """Read file content on file_path."""
    fd = open(file_path, "r")
    content = fd.read()
    fd.close()

    return content


def read_files(folder):
    """List all files from selected folder."""
    filenames = next(os.walk(folder), (None, None, []))[2]
    return filenames


def apply_function_on_ndarray(array: np.ndarray, function: object):
    """Apply function over numpy ndarray"""
    return np.array([function(xi) for xi in array])


def round_nearest_half(number):
    """Round a number to the nearest half."""
    return round(number * 2) / 2


def round_nearest_nth(number, n_fraction):
    """Round a number to the nearest fraction (if 10, it will be a decimal)"""
    return round(number * n_fraction) / n_fraction


def len_type(obj: object) -> None:
    "Get type and Len from object provided"
    print(f"Type: {type(obj)}")
    print(f"Len: {len(obj)}")


def shape_type(obj: object) -> None:
    "Get shape and type from object provided"
    print(f"Type: {type(obj)}")
    print(f"Shape: {obj.shape}")
