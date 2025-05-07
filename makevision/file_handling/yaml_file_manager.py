import os
import yaml
from typing import Any, Dict

from makevision.core import Data, FileManager
from makevision.core.exceptions import FileNotYamlError


class YamlFileManager(FileManager):
    def load(self, path: str) -> Dict:
        """Load data from a YAML file."""
        if not path.endswith(('.yaml', '.yml')):
            raise FileNotYamlError()

        if not os.path.exists(path):
            return {}

        with open(path, 'r') as file:
            data = yaml.safe_load(file)
        return data

    def save(self, path: str, data: Data) -> None:
        """Save data to a YAML file."""
        data_dict = data.convert()
        try:
            with open(path, 'w') as file:
                yaml.dump(data_dict, file, default_flow_style=False)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error writing to file {path}: {e}")
        except IOError as e:
            raise IOError(f"Error writing to file {path}: {e}")
        except Exception as e:
            raise Exception(f"An unexpected error occurred: {e}")
