from makevision.core import FileManager, Data
from makevision.core.exceptions import FileNotYamlError

import yaml
import os

class YamlFileManager(FileManager):
    def load(self, path: str) -> dict:
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
        with open(path, 'w') as file:
            yaml.dump(data_dict, file, default_flow_style=False)
    
