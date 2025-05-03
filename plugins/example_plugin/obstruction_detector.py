from makevision.core import ObstructionDetector

class ExampleObstructionDetector(ObstructionDetector):

    def detect_obstruction(self, frame):
        return False
    
    def get_obstruction_coordinates(self, frame):
        pass