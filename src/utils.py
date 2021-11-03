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
