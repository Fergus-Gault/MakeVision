from src.core import Network
import socketio

class SocketIONetwork(Network):
    def __init__(self, config: dict):
        self.config = config
        self.socket = None

        self.socket.on('connect', self.on_connect)

    def connect(self):
        """Establish a connection to the Socket.IO server."""
        self.socket = socketio.Client()
        self.socket.connect(self.config['url'])

    def send_data(self, data: dict, endpoint: str):
        """Send data to the Socket.IO server."""
        if self.socket:
            self.socket.emit(endpoint, data)
        else:
            raise ConnectionError("Socket not connected. Call connect() first.")
        
    def disconnect(self):
        self.socket.disconnect()
        
    def on_connect(self):
        """Handler for when connection is established."""
        for endpoint in self.config['endpoints']:
            self.socket.emit('join', endpoint)