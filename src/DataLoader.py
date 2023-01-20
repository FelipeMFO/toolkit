import os
import pickle


class DataLoader():
    """Module responsable for loading data. Receives a folder
    and set it to open and explore its path.
    """

    def __init__(self, folder: str) -> None:
        """Initializer. Receives the path of the folder which
        will be used to explorate.

        Args:
            folder (str): path of hte folder.
        """
        self.folder_path = folder
        self.subdirectories = [x[0] for x in os.walk(folder)][1:]
        self.files = next(os.walk(folder), (None, None, []))[2]
        self.subdirectories_path = \
            [self.folder_path +
                subdirectory for subdirectory in self.subdirectories]
        self.files_path = \
            [self.folder_path + file for file in self.files]

    def load_pickle(self, file: str) -> object:
        """_summary_

        Args:
            file (str): _description_

        Returns:
            object: _description_
        """
        if not file.endswith(".pkl"):
            file += ".pkl"
        with open(f"{self.folder_path}{file}", 'rb') as handle:
            ans = pickle.load(handle)
        return ans
