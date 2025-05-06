from makevision.core import FileManager, Data
import json
from makevision.core.exceptions import FileNotJsonError

class JsonFileManager(FileManager):
    """Base class for JSON file managers."""

    def load(self, path: str) -> dict:
        """Load data from a JSON file."""
        if not path.endswith('.json'):
            raise FileNotJsonError()

        try:
            with open(path, 'r') as file:
                data = json.load(file)
        except json.JSONDecodeError:
            return {}  # Or raise an exception, depending on your needs

        return data

    def save(self, path: str, data: Data) -> None:
        """Save data to a JSON file."""
        data_dict = data.convert()  # Use the data's convert method
        with open(path, 'w') as file:
            json.dump(data_dict, file, indent=4)
