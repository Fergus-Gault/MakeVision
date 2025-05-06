from makevision.core import FileManager, Data
import numpy as np

class NumpyFileManager(FileManager):

    def load(self, path: str) -> dict:
        """Load data from a .npy file."""
        try:
            return np.load(path, allow_pickle=True).item()
        except (IOError, ValueError, EOFError):
            # Return empty dict if file is empty or corrupted
            return {}
        
    def save(self, path: str, data: Data) -> None:
        """Save data to a .npy file."""
        data_dict = data.convert()
        np.save(path, data_dict, allow_pickle=True)