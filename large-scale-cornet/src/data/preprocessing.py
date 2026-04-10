# src/data/preprocessing.py

"""
Data preprocessing: PyTorch-based data loading and preprocessing for face recognition
"""

from __future__ import annotations

import os
import csv
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split


class FaceDataset(Dataset):
    """
    Efficient PyTorch Dataset with lazy loading for face recognition.
    
    Features:
    - Lazy loading: only reads images when requested
    - Memory efficient: stores file paths, not raw images
    - Preserves 1-pixel padding logic
    - Supports ImageNet normalization for CORnet
    """
    
    def __init__(self, root_dir, target_size=224, pad_value=1.0, 
                 normalize=True, transform=None):
        """
        Initialize face dataset with lazy loading.
        
        Parameters:
            root_dir: path to dataset root (root_dir/PersonName/image1.jpg)
            target_size: resize dimension (224 for CORnet, 128 for custom CNN)
            pad_value: padding border value (1.0 = white, 0.0 = black)
            normalize: whether to apply ImageNet normalization
            transform: optional additional transforms
        """
        self.root_dir = root_dir
        self.target_size = target_size
        self.pad_value = pad_value
        self.normalize = normalize
        self.user_transform = transform
        
        # Storage for metadata only
        self.samples = []
        self.class_to_idx = {}
        self.classes = []
        
        # Build dataset index
        self._find_classes()
        self._make_dataset()
        self._build_transforms()
        
        print(f"FaceDataset initialized: {len(self.classes)} classes, {len(self.samples)} samples")
    
    def _find_classes(self):
        """Scan directory structure to identify classes."""
        classes = sorted([d for d in os.listdir(self.root_dir) 
                         if os.path.isdir(os.path.join(self.root_dir, d)) 
                         and not d.startswith('.')])
        
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.classes = classes
    
    def _make_dataset(self):
        """Collect all image paths without loading images."""
        for target_class in self.classes:
            class_dir = os.path.join(self.root_dir, target_class)
            
            if not os.path.isdir(class_dir):
                continue
            
            for root, _, fnames in sorted(os.walk(class_dir)):
                for fname in sorted(fnames):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        path = os.path.join(root, fname)
                        item = (path, self.class_to_idx[target_class], target_class)
                        self.samples.append(item)
    
    def _build_transforms(self):
        """Build preprocessing pipeline."""
        self.base_transform = transforms.Compose([
            transforms.Resize(256),       
            transforms.CenterCrop(self.target_size), 
            transforms.ToTensor(),        
        ])
        
        # ImageNet normalization for CORnet
        if self.normalize:
            self.normalizer = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        else:
            self.normalizer = None
    
    def __len__(self):
        """Return total number of samples."""
        return len(self.samples)
    
    def __getitem__(self, index):
        """
        Load and preprocess a single image (lazy loading).
            
        Returns:
            tuple: (image_tensor, class_label, image_path)
        """
        path, target, class_name = self.samples[index]
        
        # Load image
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")
            img = Image.new('RGB', (self.target_size, self.target_size), (128, 128, 128))
        
        # Basic preprocessing
        img = self.base_transform(img)
        
        # Apply 1-pixel padding (preserving original logic)
        img = torch.nn.functional.pad(
            img, 
            (1, 1, 1, 1),
            mode='constant', 
            value=self.pad_value
        )
        
        # Normalize if enabled
        if self.normalizer is not None:
            img = self.normalizer(img)
        
        # Additional user transforms
        if self.user_transform is not None:
            img = self.user_transform(img)
        
        return img, target, path
    
    def get_class_distribution(self):
        """Compute class distribution statistics."""
        distribution = {cls: 0 for cls in self.classes}
        for _, _, class_name in self.samples:
            distribution[class_name] += 1
        return distribution


class FaceDatasetFromCSV(FaceDataset):
    """
    FaceDataset variant that builds the sample list from a labels.csv file.

    This is useful when you want a deterministic, reproducible sample ordering
    (e.g., for RSA/correlation matrices) that matches a precomputed CSV.

    Expected CSV format:
      - First column: img_id (e.g., 'juliana_0.png')
      - Remaining columns: one-hot class columns (e.g., 'juliana', 'kim', ...)
    """

    def __init__(
        self,
        training_data_dir: str,
        labels_csv_path: str,
        target_size: int = 224,
        pad_value: float = 1.0,
        normalize: bool = True,
        transform=None,
        strict: bool = False,
    ):
        self.labels_csv_path = labels_csv_path
        self.strict = strict
        super().__init__(
            root_dir=training_data_dir,
            target_size=target_size,
            pad_value=pad_value,
            normalize=normalize,
            transform=transform,
        )

    def _find_classes(self):
        """Read class names from the labels CSV header."""
        if not os.path.exists(self.labels_csv_path):
            raise FileNotFoundError(f"labels_csv_path not found: {self.labels_csv_path}")

        with open(self.labels_csv_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)

        if not header or len(header) < 2 or header[0] != "img_id":
            raise ValueError(
                "labels.csv must have header like: img_id,<class1>,<class2>,..."
            )

        classes = [c.strip() for c in header[1:] if c.strip()]
        self.classes = classes
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    def _make_dataset(self):
        """Build samples from CSV rows in CSV order."""
        self.samples = []

        with open(self.labels_csv_path, newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "img_id" not in reader.fieldnames:
                raise ValueError("labels.csv is missing required 'img_id' column.")

            class_fields = [c for c in reader.fieldnames if c != "img_id"]
            missing = [c for c in self.classes if c not in class_fields]
            if missing:
                raise ValueError(
                    f"labels.csv is missing class columns: {missing}. "
                    f"Found: {class_fields}"
                )

            for row in reader:
                img_id = (row.get("img_id") or "").strip()
                if not img_id:
                    continue

                # Determine class from one-hot columns
                class_name = None
                for c in self.classes:
                    v = (row.get(c) or "").strip()
                    if v in ("1", "1.0", "True", "true"):
                        class_name = c
                        break

                # Fallback: infer class from filename prefix
                if class_name is None:
                    class_name = img_id.split("_")[0]

                if class_name not in self.class_to_idx:
                    msg = f"Unknown class '{class_name}' for img_id='{img_id}'."
                    if self.strict:
                        raise ValueError(msg)
                    print(f"Warning: {msg} Skipping.")
                    continue

                path = os.path.join(self.root_dir, class_name, img_id)
                if not os.path.exists(path):
                    msg = f"Missing image file: {path}"
                    if self.strict:
                        raise FileNotFoundError(msg)
                    print(f"Warning: {msg} Skipping.")
                    continue

                item = (path, self.class_to_idx[class_name], class_name)
                self.samples.append(item)


def get_dataloader(data_dir, batch_size=32, target_size=224, 
                   normalize=True, num_workers=4, shuffle=False,
                   labels_csv=None, pad_value=1.0, transform=None, strict=False):
    """
    Create DataLoader for face recognition experiments.
    
    This is the recommended way to load data for CORnet analysis.
    
    Parameters:
        data_dir: path to dataset root (or processed_root if labels_csv provided)
        batch_size: batch size
        target_size: image resize dimension (224 for CORnet, 128 for CNN)
        normalize: whether to apply ImageNet normalization
        num_workers: number of parallel workers (0 for debugging, 4-8 for production)
        shuffle: whether to shuffle dataset
        labels_csv: path to labels.csv for CSV-based loading (optional)
        pad_value: padding value (default: 1.0)
        transform: additional transforms (optional)
        strict: strict mode for CSV loading (default: False)
        
    Returns:
        DataLoader: PyTorch DataLoader ready for iteration
    """
    # Choose dataset type based on labels_csv
    if labels_csv is not None:
        training_data_dir = os.path.join(data_dir, "training_data")
        dataset = FaceDatasetFromCSV(
            training_data_dir=training_data_dir,
            labels_csv_path=labels_csv,
            target_size=target_size,
            pad_value=pad_value,
            normalize=normalize,
            transform=transform,
            strict=strict
        )
    else:
        dataset = FaceDataset(
            root_dir=data_dir,
            target_size=target_size,
            pad_value=pad_value,
            normalize=normalize,
            transform=transform
        )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"DataLoader created: {len(dataset)} samples, batch_size={batch_size}, shuffle={shuffle}")
    
    return loader


def get_processed_dataset(
    processed_root: str,
    target_size: int = 224,
    pad_value: float = 1.0,
    normalize: bool = True,
    transform=None,
    strict: bool = False,
):
    """
    Convenience helper for the local processed dataset layout:
      processed_root/
        training_data/<person>/<img>.png
        labels.csv

    Returns:
        FaceDatasetFromCSV
    """
    training_data_dir = os.path.join(processed_root, "training_data")
    labels_csv_path = os.path.join(processed_root, "labels.csv")
    return FaceDatasetFromCSV(
        training_data_dir=training_data_dir,
        labels_csv_path=labels_csv_path,
        target_size=target_size,
        pad_value=pad_value,
        normalize=normalize,
        transform=transform,
        strict=strict,
    )


def get_processed_dataloader(
    processed_root: str,
    batch_size: int = 32,
    target_size: int = 224,
    pad_value: float = 1.0,
    normalize: bool = True,
    num_workers: int = 4,
    shuffle: bool = False,
    transform=None,
    strict: bool = False,
):
    """
    Create a DataLoader for the local processed dataset layout.
    
    Convenience wrapper for get_dataloader() with CSV support.
    """
    labels_csv_path = os.path.join(processed_root, "labels.csv")
    
    return get_dataloader(
        data_dir=processed_root,
        batch_size=batch_size,
        target_size=target_size,
        normalize=normalize,
        num_workers=num_workers,
        shuffle=shuffle,
        labels_csv=labels_csv_path,
        pad_value=pad_value,
        transform=transform,
        strict=strict
    )


def get_processed_split_dataloaders(
    processed_root: str,
    batch_size: int = 32,
    target_size: int = 224,
    pad_value: float = 1.0,
    normalize: bool = True,
    num_workers: int = 4,
    split_ratio=(0.8, 0.1, 0.1),
    split_mode: str = 'stratified',
    images_per_class: int = 5,
    seed: int = 42,
    transform=None,
    strict: bool = False,
):
    """
    Create Train/Val/Test dataloaders for the local processed dataset layout.
    
    Convenience wrapper for get_split_dataloaders() with CSV support.
    """
    labels_csv_path = os.path.join(processed_root, "labels.csv")
    
    return get_split_dataloaders(
        data_dir=processed_root,
        batch_size=batch_size,
        target_size=target_size,
        split_ratio=split_ratio,
        num_workers=num_workers,
        seed=seed,
        labels_csv=labels_csv_path,
        pad_value=pad_value,
        normalize=normalize,
        transform=transform,
        strict=strict,
        split_mode=split_mode,
        images_per_class=images_per_class
    )


def collate_samples_to_arrays(data_dir, target_size=224, normalize=True, limit=None):
    """
    Load all images and convert to numpy arrays for backward compatibility.
    
    WARNING: Loads all images into memory. Only use for small datasets.
    For large datasets, use get_dataloader() instead.
    
    Parameters:
        data_dir: path to dataset directory
        target_size: image size
        normalize: whether to normalize
        limit: maximum number of samples to load
        
    Returns:
        tuple: (images_array, labels_array, paths_list)
    """
    dataset = FaceDataset(
        root_dir=data_dir,
        target_size=target_size,
        normalize=normalize
    )
    
    if limit is not None:
        dataset.samples = dataset.samples[:limit]
    
    images = []
    labels = []
    paths = []
    
    print(f"Loading {len(dataset.samples)} samples into memory...")
    
    for i in range(len(dataset)):
        img, label, path = dataset[i]
        images.append(img.numpy())
        labels.append(label)
        paths.append(path)
        
        if (i + 1) % 100 == 0:
            print(f"  Loaded {i + 1}/{len(dataset)} samples")
    
    images_array = np.array(images)
    labels_array = np.array(labels)
    
    print(f"Complete: {images_array.shape}")
    
    return images_array, labels_array, paths


def get_split_dataloaders(data_dir, batch_size=32, target_size=224, 
                          split_ratio=(0.8, 0.1, 0.1), 
                          num_workers=4, seed=42,
                          labels_csv=None, pad_value=1.0, normalize=True,
                          transform=None, strict=False,
                          split_mode='stratified', images_per_class=5):
    """
    Create Train/Val/Test dataloaders with flexible splitting strategies.
    
    Parameters:
        data_dir: path to dataset (or processed_root if labels_csv provided)
        batch_size: batch size
        target_size: image size
        split_ratio: tuple of (train_frac, val_frac, test_frac), sum must be 1.0
        num_workers: number of workers for data loading
        seed: random seed for reproducibility
        labels_csv: path to labels.csv for CSV-based loading (optional)
        pad_value: padding value (default: 1.0)
        normalize: apply ImageNet normalization (default: True)
        transform: additional transforms (optional)
        strict: strict mode for CSV loading (default: False)
        split_mode: 'stratified' or 'block'
            'stratified': sklearn stratified split (default)
            'block': first image per class -> val, rest -> train (no test)
        images_per_class: for block mode, number of images per class
        
    Returns:
        dict: {'train': loader, 'val': loader, 'test': loader} for stratified mode
              {'train': loader, 'val': loader} for block mode
    """
    # Create dataset
    if labels_csv is not None:
        training_data_dir = os.path.join(data_dir, "training_data")
        dataset = FaceDatasetFromCSV(
            training_data_dir=training_data_dir,
            labels_csv_path=labels_csv,
            target_size=target_size,
            pad_value=pad_value,
            normalize=normalize,
            transform=transform,
            strict=strict
        )
    else:
        dataset = FaceDataset(
            root_dir=data_dir,
            target_size=target_size,
            pad_value=pad_value,
            normalize=normalize,
            transform=transform
        )
    
    targets = [y for _, y, _ in dataset.samples]
    indices = np.arange(len(targets))
    
    # Block-based split (for EIB experiments)
    if split_mode == 'block':
        val_idx = []
        train_idx = []
        
        for i in range(len(indices)):
            if i % images_per_class == 0:
                val_idx.append(i)
            else:
                train_idx.append(i)
        
        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)
        
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        print("Data Split Summary (Block Mode):")
        print(f"  Train: {len(train_dataset)} images")
        print(f"  Val:   {len(val_dataset)} images")
        
        return {'train': train_loader, 'val': val_loader}
    
    # Stratified split
    else:
        assert abs(sum(split_ratio) - 1.0) < 1e-6, "Split ratios must sum to 1.0"
        
        # Check if stratification is possible
        unique_classes, class_counts = np.unique(targets, return_counts=True)
        min_class_count = class_counts.min()
        smallest_split = min(split_ratio)
        can_stratify = min_class_count >= 2 and (min_class_count * smallest_split) >= 2
        
        if not can_stratify:
            print(f"Warning: Dataset too small for stratified split. Using sequential split.")
            n_total = len(indices)
            n_train = int(n_total * split_ratio[0])
            n_val = int(n_total * split_ratio[1])
            
            np.random.seed(seed)
            shuffled_idx = np.random.permutation(indices)
            
            train_idx = shuffled_idx[:n_train]
            val_idx = shuffled_idx[n_train:n_train + n_val]
            test_idx = shuffled_idx[n_train + n_val:]
        else:
            # Stratified split: Train vs (Val + Test)
            train_idx, temp_idx = train_test_split(
                indices,
                test_size=(split_ratio[1] + split_ratio[2]),
                stratify=targets,
                random_state=seed
            )
            
            # Split Val vs Test
            relative_test_ratio = split_ratio[2] / (split_ratio[1] + split_ratio[2])
            temp_targets = [targets[i] for i in temp_idx]
            
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=relative_test_ratio,
                stratify=temp_targets,
                random_state=seed
            )
        
        # Create Subsets
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        test_dataset = Subset(dataset, test_idx)
        
        print("Data Split Summary:")
        print(f"  Train: {len(train_dataset)} images ({split_ratio[0]*100:.0f}%)")
        print(f"  Val:   {len(val_dataset)} images ({split_ratio[1]*100:.0f}%)")
        print(f"  Test:  {len(test_dataset)} images ({split_ratio[2]*100:.0f}%)")
        
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        return {'train': train_loader, 'val': val_loader, 'test': test_loader}
