import numpy as np
import pickle
import os


class DataDumper():
    """Module responsable for loading data. Receives a folder
    and set it to open and explore its path.
    """

    def __init__(self) -> None:
        pass

    def save_file(self, file: object, folder_path: str,
                  file_name: str) -> None:
        """Save a python object as pickle on specified folder."""
        with open(f"{folder_path}{file_name}.pkl", 'wb') as f:
            pickle.dump(file, f, protocol=pickle.HIGHEST_PROTOCOL)

    def save_dict_values(self, dic: dict,
                         folder_path: str,
                         label: str) -> None:
        """Save dictionary values on specified folder.
        If value is a numpy object, save as npy, otherwise as pkl.

        Args:
            dic (dict): dictionary whose values will be saved.
            Keys will be the name of files.
            folder_path (str): folder where files will be saved.
            label (str): label identifier files names.
        """
        for key, value in dic.items():
            if isinstance(value, np.ndarray):
                np.save(
                    file=f"{folder_path}{key}_{label}.npy",
                    arr=value
                )
            else:
                self.save_file(
                    file=value,
                    folder_path=folder_path,
                    file_name=f"{label}_{key}"
                )

    def check_exist_folder_create(self, path: str,
                                  overwrite_folder: bool = False) -> None:
        """Check if a folder path already existis. If overwrite_folder flag
        is set, it overwrites, otherwise, it doesn't.

        Args:
            path (str): folder path.
        """
        try:
            os.makedirs(path, exist_ok=overwrite_folder)
        except FileExistsError:
            print(f"""Folder {path} already exists.
Please change the label to do not overwrite previous figures and models.""")
            return 0
