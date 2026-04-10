# src/utils/helpers.py

"""
Helper utility functions for environment detection, package management, and device management
"""

import sys
import subprocess


# ============================================================================
# Environment Detection
# ============================================================================

def is_google_colab():
    """
    Determine if the current runtime is Google Colab.

    Inputs:
      - None

    Returns:
      - bool: True if running on Google Colab, False otherwise.
    """
    try:
        import google.colab  # noqa
        return True
    except ImportError:
        return False


def install_missing_packages():
    """
    Install a predefined list of missing packages if running in Google Colab.

    Inputs:
      - None

    Returns:
      - None
    """
    if is_google_colab():
        packages = ["scipy", "tqdm"]
        for pkg in packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


# ============================================================================
# Device Management (PyTorch)
# ============================================================================

def get_device():
    """
    Get available device (cuda or cpu).
    
    Returns:
        str: 'cuda' if available, else 'cpu'
    """
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        return 'cpu'


def to_device(tensor_or_model, device=None):
    """
    Move tensor or model to specified device.
    
    Parameters:
        tensor_or_model: PyTorch tensor or model
        device: target device (if None, auto-detect)
        
    Returns:
        tensor or model on target device
    """
    if device is None:
        device = get_device()
    return tensor_or_model.to(device)


def empty_cache():
    """Clear GPU cache if CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

