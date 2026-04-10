# src/data/__init__.py

"""
Data module: PyTorch dataset loading and preprocessing
"""

# PyTorch preprocessing
from .preprocessing import (
    FaceDataset,
    get_dataloader,
    collate_samples_to_arrays,
    get_split_dataloaders,
    FaceDatasetFromCSV,
    get_processed_dataset,
    get_processed_dataloader,
    get_processed_split_dataloaders
)

# Data loader (download and prepare datasets)
from .loader import (
    check_dataset_exists,
    prepare_balanced_lfw,
    validate_dataset
)

__all__ = [
    # PyTorch loaders
    'FaceDataset',
    'get_dataloader',
    'collate_samples_to_arrays',
    'get_split_dataloaders',
    'FaceDatasetFromCSV',
    'get_processed_dataset',
    'get_processed_dataloader',
    'get_processed_split_dataloaders',
    # Dataset setup
    'check_dataset_exists',
    'prepare_balanced_lfw',
    'validate_dataset',
]
