import json
from typing import Dict

from makevision.core import Data, FileManager
from makevision.core.exceptions import FileNotJsonError


class JsonFileManager(FileManager):
    """Base class for JSON file managers."""

    def load(self, path: str) -> Dict:
        """Load data from a JSON file."""
        if not path.endswith('.json'):
            raise FileNotJsonError()

        try:
            with open(path, 'r') as file:
                data = json.load(file)
        except json.JSONDecodeError:
            return {}

        return data

    def save(self, path: str, data: Data) -> None:
        """Save data to a JSON file."""
        data_dict = data.convert()
        try:
            with open(path, 'w') as file:
                json.dump(data_dict, file, indent=4)
        except TypeError as e:
            raise TypeError(f"Data cannot be serialized to JSON: {e}")
        except IOError as e:
            raise IOError(f"Error writing to file {path}: {e}")
        except Exception as e:
            raise Exception(f"An unexpected error occurred: {e}")
