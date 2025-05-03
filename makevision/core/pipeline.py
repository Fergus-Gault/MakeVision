from abc import ABC, abstractmethod

class Pipeline(ABC):
    """Abstract base class for program pipelines."""
    @abstractmethod
    def run(self, *args, **kwargs) -> None:
        """Run the pipeline with the given arguments."""
        pass