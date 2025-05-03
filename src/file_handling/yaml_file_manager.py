from src.core import FileManager, Data
import yaml
from src.core.exceptions import FileNotYamlError
import os

class YamlFileManager(FileManager):
    def save(self, path: str, data: Data) -> None:
        """Save data to a YAML file."""
        with open(path, 'w') as file:
            yaml.dump(data.data, file, default_flow_style=False)
    
    def load(self, path: str) -> dict:
        """Load data from a YAML file."""
        if not path.endswith('.yaml', '.yml'):
            raise FileNotYamlError()
        
        if not os.path.exists(path):
            return {}

        with open(path, 'r') as file:
            data = yaml.safe_load(file)
        return data
