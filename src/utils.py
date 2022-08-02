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


# Notebook strip to import models
# module_path = os.path.abspath(os.path.join('..','..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

# from src.modeling.functions_autoML import auto_ML

import os
import numpy as np


def read_file(file_path):
    """Read file content on file_path."""
    fd = open(file_path, 'r')
    content = fd.read()
    fd.close()

    return content


def read_files(folder):
    """List all files from selected folder."""
    filenames = next(
        os.walk(folder),
        (None, None, []))[2]
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


def getlayer_out_dim_convt1d(l_in, stride, kernel_size) -> int:
    """Calculate the output dimension of a transposed 1DConv layer"""
    l_out = (l_in - 1)*stride + kernel_size
    return l_out


def getlayer_out_dim_conv1d(l_in, stride, kernel_size) -> int:
    """Calculate the output dimension of a 1DConv layer"""
    l_out = ((l_in - (kernel_size-1) - 1)//stride) + 1
    return l_out


def len_type(obj: object) -> None:
    "Get type and Len from object provided"
    print(f"Type: {type(obj)}")
    print(f"Len: {len(obj)}")


def shape_type(obj: object) -> None:
    "Get shape and type from object provided"
    print(f"Type: {type(obj)}")
    print(f"Shape: {obj.shape}")


def split_list_into_batch_of_size_n(dataset: list, batch_size: int) -> list:
    """_summary_

    Args:
        dataset (list): _description_
        batch_size (int): _description_

    Returns:
        list: _description_
    """
    return [dataset[i:i + batch_size] for i in range(0, len(dataset),
                                                     batch_size)]
