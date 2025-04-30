import torch
import os

def torch_safe_load(file):
    """Modified torch_safe_load function to handle the new PyTorch 2.6 security changes."""
    if os.path.isfile(file):
        try:
            # First attempt with weights_only=False (less secure but more compatible)
            return torch.load(file, map_location="cpu", weights_only=False), file
        except Exception as e:
            print(f"Warning: Failed to load '{file}' with weights_only=False: {e}")
            try:
                # Second attempt with the original PyTorch 2.6 default (more secure)
                return torch.load(file, map_location="cpu"), file
            except Exception as e:
                print(f"Error: Failed to load '{file}' with default settings: {e}")
                # If both attempts fail, raise the exception
                raise e
    else:
        raise FileNotFoundError(f"Model file not found: {file}")