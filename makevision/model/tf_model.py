from makevision.core import Model


class TfModel(Model):
    def __init__(self, model_path: str):
        """Initialize the TensorFlow model with the given model path."""
        super().__init__(model_path)

    def load_model(self, model_path: str):
        """Load the TensorFlow model from the specified path."""
        import tensorflow as tf
        self.model = tf.keras.models.load_model(model_path)
