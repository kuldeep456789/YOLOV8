from ultralytics.nn.tasks import DetectionModel
import torch.serialization

# Add the DetectionModel to the safe globals list for torch loading
torch.serialization.add_safe_globals([DetectionModel])

class SafeYOLO:
    """A wrapper for YOLO that handles safe model loading."""
    
    def __init__(self, model_path):
        """Initialize the YOLO model with safe loading."""
        from ultralytics import YOLO
        # Original YOLO initialization with model path
        self.model = YOLO(model_path)
    
    def __call__(self, *args, **kwargs):
        """Forward the call to the underlying YOLO model."""
        return self.model(*args, **kwargs)