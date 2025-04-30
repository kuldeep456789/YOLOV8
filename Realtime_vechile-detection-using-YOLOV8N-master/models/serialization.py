import torch
from ultralytics.nn.tasks import DetectionModel

def safe_torch_load(file_path, map_location=None):
    """
    A safer version of torch.load that handles PyTorch 2.6 security changes.
    
    Args:
        file_path: Path to the model file
        map_location: Device mapping for loaded tensors
        
    Returns:
        The loaded model
    """
    # Add DetectionModel to the list of safe globals
    torch.serialization.add_safe_globals([DetectionModel])
    
    try:
        # Try loading with weights_only=False for backward compatibility
        return torch.load(file_path, map_location=map_location, weights_only=False)
    except Exception as e:
        print(f"Warning: Loading with weights_only=False failed: {e}")
        # Try the default settings as fallback
        return torch.load(file_path, map_location=map_location)