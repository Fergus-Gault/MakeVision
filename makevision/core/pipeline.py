from abc import ABC, abstractmethod

class Pipeline(ABC):
    """Abstract base class for program pipelines."""
    @abstractmethod
    def run(self, *args, **kwargs) -> None:
        """
        Run the pipeline.
        This method should be implemented by plugins
        to define the program flow.
        """
        pass