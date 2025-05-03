from abc import ABC, abstractmethod
from typing import Any

class Network(ABC):
    """Abstract base class for network components."""

    @abstractmethod
    def connect(self, *args, **kwargs) -> None:
        """Establish a connection with the given arguments."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the network."""
        pass

    @abstractmethod
    def send_data(self, data: Any) -> None:
        """Send data over the network."""
        pass

    @abstractmethod
    def receive_data(self) -> None:
        """Receive data from the network."""
        pass