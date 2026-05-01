# src/__init__.py

"""
ASD Face Recognition Modeling with CNN
E/I imbalance and internal noise cause weak neural representations and face recognition challenges in ASD
"""

__version__ = "1.0.0"

# Import main modules for easier access
from . import models
from . import data
from . import analysis
from . import utils

__all__ = [
    'models',
    'data',
    'analysis',
    'utils',
]
