from makevision.core import FileManager, Data
import json
import numpy as np
from makevision.core.exceptions import FileNotJsonError
import os

class JsonFileManager(FileManager):
    def save(self, path: str, data: Data) -> None:
        """Save data to a JSON file."""

        def numpy_to_json_serializable(obj):
            """Convert numpy arrays to lists recursively."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif obj is None:
                return None
            else:
                return obj

        # Build data dictionary dynamically from object attributes
        data_dict = {}
        for attr_name in vars(data):
            attr_value = getattr(data, attr_name)
            data_dict[attr_name] = numpy_to_json_serializable(attr_value)

        with open(path, 'w') as file:
            json.dump(data_dict, file, indent=4)

    def load(self, path: str) -> dict:
        """Load data from a JSON file."""
        if not path.endswith('.json'):
            raise FileNotJsonError()
        
        if not os.path.exists(path):
            return {}

        with open(path, 'r') as file:
            data = json.load(file)

        return data
