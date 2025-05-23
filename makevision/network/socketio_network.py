from makevision.core import Network
from typing import Dict


class SocketIONetwork(Network):
    def __init__(self, config: Dict):
        self.config = config
        self.socket = None
        self.connect()
        self.socket.on('connect', self.on_connect)

    def connect(self):
        """Establish a connection to the Socket.IO server."""
        import socketio
        self.socket = socketio.Client()
        self.socket.connect(self.config['url'])

    def send_data(self, data: Dict, endpoint: str):
        """
        Send data over the socketio network to a specific endpoint.

        Args:
            data (Dict): The data to send.
            endpoint (str): The endpoint to send the data to.

        Raises:
            ConnectionError: If the socket is not connected.
        """
        if self.socket:
            self.socket.emit(endpoint, data)
        else:
            raise ConnectionError(
                "Socket not connected. Call connect() first.")

    def receive_data(self):
        pass

    def disconnect(self):
        self.socket.disconnect()

    def on_connect(self):
        """Handler for when connection is established."""
        for endpoint in self.config['endpoints']:
            self.socket.emit('join', endpoint)
