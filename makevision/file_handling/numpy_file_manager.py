import numpy as np

from makevision.core import FileManager, Data


class NumpyFileManager(FileManager):

    def load(self, path: str) -> dict:
        """Load data from a .npy file."""
        try:
            return np.load(path, allow_pickle=True).item()
        except (IOError, ValueError, EOFError):
            return {}

    def save(self, path: str, data: Data) -> None:
        """Save data to a .npy file."""
        data_dict = data.convert()
        try:
            np.save(path, data_dict, allow_pickle=True)
        except IOError as e:
            raise IOError(f"Error writing to file {path}: {e}")
        except Exception as e:
            raise Exception(f"An unexpected error occurred: {e}")
