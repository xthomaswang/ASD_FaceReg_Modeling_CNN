# src/analysis/rsa.py

"""
RSA (Representational Similarity Analysis) module

Complete pipeline for RSA analysis including:
- Feature extraction (PyTorch)
- Correlation computation
- Multi-run analysis workflows
- RSA visualization
"""

import numpy as np
import torch
import re
from collections import defaultdict, Counter

from .visualization import (
    visualize_correlation_heatmap, 
    compute_discriminability,
    plot_discriminability_comparison
)


# ============================================================================
# SECTION 1: Feature Extraction
# ============================================================================

def get_device():
    """Get available device (cuda or cpu)."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _find_layer(model, layer_name):
    """
    Resolve layer from model with support for multiple model structures.
    
    Returns: target layer module
    Raises: AttributeError if layer not found
    """
    # Helper: build search paths for a decoder sub-layer.
    # Covers: raw CORnet, DataParallel, CornetWithPathology (.base or .core_model)
    def _decoder_paths(attr):
        return [
            lambda m, a=attr: getattr(m.decoder, a),
            lambda m, a=attr: getattr(m.module.decoder, a),
            lambda m, a=attr: getattr(m.base.decoder, a),              # CornetWithPathology.base
            lambda m, a=attr: getattr(m.core_model.decoder, a),        # non-DataParallel wrapper
            lambda m, a=attr: getattr(m.core_model.module.decoder, a), # DataParallel wrapper
        ]

    # Handle special layer aliases
    if layer_name in ['decoder', 'flatten']:
        search_paths = _decoder_paths('flatten')
    elif layer_name in ['logits', 'linear']:
        search_paths = _decoder_paths('linear')
    elif layer_name in ['embed64', 'penultimate', 'penultimate_dense', 'fc64']:
        # Post-activation 64D like your Keras Dense(64, relu)
        # If penultimate_dim=0 (original architecture), fallback to flatten (512D)
        search_paths = _decoder_paths('penultimate_dense_relu') + _decoder_paths('flatten')
    elif layer_name in ['embed64_pre', 'penultimate_pre', 'penultimate_dense_pre']:
        # Pre-activation penultimate (before ReLU)
        search_paths = _decoder_paths('penultimate_dense') + _decoder_paths('flatten')
    else:
        # Generic layer search paths (try common model wrapper patterns)
        search_paths = [
            lambda m: getattr(m, layer_name),                           # Direct attribute
            lambda m: getattr(m.module, layer_name),                    # DataParallel wrapper
            lambda m: getattr(m.base, layer_name),                      # CornetWithPathology.base
            lambda m: getattr(m.core_model, layer_name),                # non-DataParallel wrapper
            lambda m: getattr(m.core_model.module, layer_name),         # DataParallel wrapper
        ]
    
    # Try each search path in priority order
    for search_fn in search_paths:
        try:
            target = search_fn(model)
            print(f"Found layer '{target}' in model.")
            return target
        except AttributeError:
            continue
    
    # If all paths fail, raise informative error
    raise AttributeError(
        f"Layer '{layer_name}' not found in model. "
        f"Model structure: {type(model).__name__}. "
        f"Available top-level attributes: {list(vars(model).keys())[:10]}"
    )


def extract_features_pytorch(model, data_loader, layer='IT', use_gap=True, device=None):
    """
    Extract features from PyTorch model (CORnet).
    
    Parameters:
        model: PyTorch model
        data_loader: PyTorch DataLoader
        layer: layer name to extract from
        use_gap: whether to apply Global Average Pooling
        device: target device (auto-detect if None)
        
    Returns:
        tuple: (features_array, labels_list)
    """
    if device is None:
        device = get_device()
    
    model.eval()
    model.to(device)
    
    all_features = []
    all_labels = []
    
    # Register hook for feature extraction
    features_holder = []
    
    def hook_fn(module, input, output):
        """Hook function to capture layer output."""
        out = output.detach()
        # Apply Global Average Pooling if requested and output is spatial (4D)
        # Skip GAP if layer is decoder/flatten/penultimate/logits (already 2D)
        skip_gap_layers = ['decoder', 'flatten', 'logits', 'linear', 
                          'embed64', 'penultimate', 'penultimate_dense', 'fc64', 
                          'embed64_pre', 'penultimate_pre', 'penultimate_dense_pre']
        if use_gap and len(out.shape) == 4 and layer not in skip_gap_layers:
            out = torch.mean(out, dim=[2, 3])
        features_holder.append(out.cpu().numpy())
    
    target_layer = _find_layer(model, layer)
    handle = target_layer.register_forward_hook(hook_fn)
    
    # Extract features
    with torch.no_grad():
        for images, labels, _ in data_loader:
            images = images.to(device)
            features_holder.clear()

            # Forward pass (triggers hook)
            model(images)
            
            if len(features_holder) == 0:
                raise RuntimeError(f"No features captured. Check layer='{layer}' hook registration.")

            batch_feats = features_holder[0]
            all_features.append(batch_feats)
            
            # Convert labels to list
            if isinstance(labels, torch.Tensor):
                all_labels.extend(labels.cpu().numpy())
            else:
                all_labels.extend(labels)
    
    # Remove hook
    handle.remove()
    
    # Concatenate all batches
    features_array = np.concatenate(all_features, axis=0)
    
    print(f"Extracted features: {features_array.shape}")
    
    return features_array, all_labels


def extract_features(model, data, layer='IT', **kwargs):
    """
    Extract features from a PyTorch model via a DataLoader.

    Parameters:
        model: PyTorch model
        data: DataLoader
        layer: layer name to extract from
        **kwargs: additional arguments

    Returns:
        tuple: (features array, labels)
    """
    if not hasattr(data, '__iter__'):
        raise ValueError("PyTorch models require DataLoader as input")
    return extract_features_pytorch(model, data, layer=layer, **kwargs)


# ============================================================================
# SECTION 2: Correlation Computation
# ============================================================================

def compute_image_image_pearson(
    features: np.ndarray,
    flatten: bool = True,
    per_image_center: bool = True,
    per_image_zscore: bool = False,
    eps: float = 1e-8,
    nan_to_zero: bool = True,
    dtype=np.float32,
):
    """
    Compute image-by-image Pearson correlation matrix from extracted features.
    This matches the semantic meaning of:
        np.corrcoef(X, rowvar=True)
    where X has shape (N_images, D_features).

    Parameters
    ----------
    features : np.ndarray
        Array of shape (N_images, ...) e.g. (N, C, H, W) or (N, D).
    flatten : bool
        If True, flatten features to (N, D) when features.ndim > 2.
    per_image_center : bool
        If True, subtract mean across feature dims for each image (row-wise centering).
        This is the standard step that makes Pearson equal to cosine on mean-centered vectors.
    per_image_zscore : bool
        If True, z-score each image across feature dims (row-wise). This makes Pearson
        less sensitive to per-image scale. Note Pearson already normalizes by std, but
        z-score helps numerical stability / explicitness.
    eps : float
        Small constant to avoid division by zero.
    nan_to_zero : bool
        Replace NaNs with 0 (can happen if an image vector has zero variance).
    dtype : np.dtype
        Output dtype.

    Returns
    -------
    sim : np.ndarray
        Pearson correlation matrix of shape (N_images, N_images).
    """
    X = np.asarray(features)
    
    # 1) Flatten to (N, D)
    if flatten and X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    elif X.ndim != 2:
        raise ValueError(f"features must be 2D or flattenable to 2D. Got shape={X.shape}")

    # 2) Row-wise centering: subtract mean across dims for each image
    if per_image_center:
        X = X - X.mean(axis=1, keepdims=True)

    # 3) Optional row-wise z-score (explicit standardization)
    if per_image_zscore:
        sd = X.std(axis=1, keepdims=True)
        X = X / (sd + eps)

    # 4) Compute Pearson correlation = normalized dot product on centered data
    #    Corr(i,j) = (Xi · Xj) / (||Xi|| * ||Xj||)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / (norms + eps)
    sim = Xn @ Xn.T  # (N, N)
    
    if nan_to_zero:
        sim = np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)
    
    return sim.astype(dtype)


def compute_correlation_pytorch(
    features,
    metric="pearson",
    flatten=True,
    per_image_center=True,
    per_image_zscore=False,
    eps=1e-8,
):
    """
    Compute image-by-image correlation matrix from extracted features.
    
    Parameters:
        features: numpy array of shape (N_images, ...)
                  If multi-dimensional, will be flattened
        metric: 'pearson' or 'cosine'
        flatten: if True, flatten features to (N, D) when ndim > 2
        per_image_center: if True, subtract mean across feature dims for each image (row-wise)
                         This is standard for Pearson correlation
        per_image_zscore: if True, z-score each image across feature dims
        eps: small value to avoid division by zero
        
    Returns:
        correlation_matrix: numpy array of shape (N_images, N_images)
    """
    metric = metric.lower().strip()
    
    if metric == "pearson":
        return compute_image_image_pearson(
            features,
            flatten=flatten,
            per_image_center=per_image_center,
            per_image_zscore=per_image_zscore,
            eps=eps,
        )
    
    elif metric == "cosine":
        # Pure cosine WITHOUT mean-centering (classic cosine similarity)
        X = np.asarray(features)
        if flatten and X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        norms = np.linalg.norm(X, axis=1, keepdims=True) + eps
        Xn = X / norms
        sim = Xn @ Xn.T
        sim = np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)
        return sim.astype(np.float32)
    
    else:
        raise ValueError("metric must be 'pearson' or 'cosine'")


# ============================================================================
# SECTION 3: Label Processing Utilities
# ============================================================================

def _smart_sort_key(person_id):
    """
    Create sort key that handles numeric components in strings.
    E.g., "Person_2" -> ("Person_", 2), "Person_10" -> ("Person_", 10)
    This ensures proper numeric ordering: Person_2 < Person_10
    """
    if isinstance(person_id, str):
        # Split on numbers: "Person_10" -> ["Person_", "10", ""]
        parts = re.split(r'(\d+)', str(person_id))
        # Convert numeric parts to int for proper sorting
        return tuple(int(p) if p.isdigit() else p for p in parts)
    else:
        # Numeric or other types: use as-is
        return person_id


def _extract_person_ids(labels):
    """
    Extract person identifiers from labels.
    Handles both "PersonName_0" format and numeric formats.
    
    Returns:
        list: person IDs (without image numbers)
    """
    if len(labels) == 0:
        return []
    
    # Handle both "PersonName_0" and numeric formats
    if isinstance(labels[0], str) and '_' in labels[0]:
        # Format: "PersonName_0" -> extract person part
        return [label.rsplit('_', 1)[0] for label in labels]
    else:
        # Numeric labels
        return list(labels)


def _sort_labels_and_matrices(labels, matrices):
    """
    Sort labels and corresponding matrices by person ID for clearer visualization.
    
    Parameters:
        labels: list of label strings
        matrices: list of correlation matrices
        
    Returns:
        tuple: (sorted_labels, sorted_person_ids, sorted_matrices, sort_indices)
    """
    labels_array = np.array(labels)
    person_ids = _extract_person_ids(labels)
    person_ids_array = np.array(person_ids)
    
    # Sort using smart key (handles numeric components)
    sort_indices = sorted(range(len(person_ids)), key=lambda i: _smart_sort_key(person_ids[i]))
    sort_indices = np.array(sort_indices)
    
    sorted_labels = labels_array[sort_indices].tolist()
    sorted_person_ids = person_ids_array[sort_indices].tolist()
    
    # Sort matrices (reorder both rows and columns)
    sorted_matrices = []
    for matrix in matrices:
        sorted_matrix = matrix[sort_indices, :][:, sort_indices]
        sorted_matrices.append(sorted_matrix)
    
    return sorted_labels, sorted_person_ids, sorted_matrices, sort_indices


def _create_display_labels(sorted_labels, sorted_person_ids):
    """
    Create display labels showing only first occurrence of each person.
    
    Parameters:
        sorted_labels: list of sorted full labels
        sorted_person_ids: list of sorted person IDs
        
    Returns:
        list: display labels (empty string for duplicate persons)
    """
    display_labels = []
    prev_person = None
    
    for i, person_id in enumerate(sorted_person_ids):
        if person_id != prev_person:
            # First image of this person - show the name
            if isinstance(sorted_labels[i], str) and '_' in sorted_labels[i]:
                display_labels.append(sorted_labels[i].rsplit('_', 1)[0])
            else:
                display_labels.append(str(sorted_labels[i]))
            prev_person = person_id
        else:
            # Not first image - use empty string
            display_labels.append('')
    
    return display_labels


def get_person_name_labels(loader, label_indices, add_image_number=True):
    """
    Convert label indices to person names with optional image numbers.
    
    Parameters:
        loader: DataLoader used for RSA
        label_indices: list of label indices from compute_rsa_matrix
        add_image_number: bool
            If True, return labels like ["PersonName_0", "PersonName_1", ...]
            If False, return labels like ["PersonName", "PersonName", ...]
        
    Returns:
        list: person name labels
    
    Usage:
        # For RSA heatmap visualization (need unique labels)
        labels = get_person_name_labels(loader, indices, add_image_number=True)
        
        # For compute_discriminability (need person names only)
        labels = get_person_name_labels(loader, indices, add_image_number=False)
    """
    # Get dataset (unwrap Subset if needed)
    dataset = loader.dataset
    if hasattr(dataset, 'dataset'):
        dataset = dataset.dataset
    
    # Get class names
    idx_to_name = {idx: name for name, idx in dataset.class_to_idx.items()}
    person_names = [idx_to_name[int(idx)] for idx in label_indices]
    
    # Return person names only (for discriminability calculation)
    if not add_image_number:
        return person_names
    
    # Add image numbers (for heatmap visualization)
    name_counter = Counter()
    detailed_labels = []
    for name in person_names:
        detailed_labels.append(f"{name}_{name_counter[name]}")
        name_counter[name] += 1
    
    return detailed_labels


# ============================================================================
# SECTION 4: RSA Workflows
# ============================================================================

def compute_rsa_matrix(
    model, loader, device='cuda', layer='decoder',
    per_image_center=True, per_image_zscore=False,
    metric="pearson" 
):
    """
    Extract features and compute RSA matrix for single run.
    
    Lightweight function for use inside training loops.
    
    Parameters:
        model: Trained PyTorch model
        loader: DataLoader (Test or Train set)
        device: 'cuda' or 'cpu'
        layer: Layer name to extract from
               - 'embed64' or 'penultimate_dense': penultimate layer (recommended for RSA)
                 * If penultimate_dim > 0: extracts penultimate_dim features (e.g., 64D)
                 * If penultimate_dim = 0: automatically falls back to 'flatten' (512D)
               - 'logits' or 'linear': final classification layer
               - 'decoder' or 'flatten': 512-dim features before penultimate
               - 'IT', 'V4', 'V2', 'V1': intermediate conv layers
        per_image_center: if True, subtract mean across feature dims for each image (standard for Pearson)
        per_image_zscore: if True, z-score each image across feature dims
        metric: 'pearson' or 'cosine'
        
    Returns:
        tuple: (correlation_matrix, labels_list)
    """
    features_concat, all_labels = extract_features_pytorch(
        model,
        loader,
        layer=layer,
        use_gap=True,
        device=device
    )

    corr_mat = compute_correlation_pytorch(
        features_concat,
        per_image_center=per_image_center,
        per_image_zscore=per_image_zscore,
        metric=metric
    )

    return corr_mat, all_labels


def aggregate_and_visualize_rsa(all_raw_matrices, all_labels, title_suffix="", full_label=False):
    """
    Aggregate raw matrices and visualize results using the same style as visualize_local_results.
    
    Automatically sorts labels and matrices by person ID for clearer visualization.
    Generates both sample-level and class-averaged plots.
    
    Parameters:
        all_raw_matrices: dict {condition_name: [matrix_run1, matrix_run2, ...]}
        all_labels: dict {condition_name: labels_list} OR list of labels (backward compatible)
                   - If dict: each condition has its own labels (RECOMMENDED)
                   - If list: same labels used for all conditions (old behavior)
        title_suffix: optional string for plot title
        full_label: bool, if False, only show first image label for each person (default: False)
        
    Returns:
        dict: Mean matrices for each condition
    """
    print(f"\nVISUALIZING MEAN RSA MATRICES {title_suffix}")
    print("="*60)
    
    # Handle backward compatibility: if all_labels is a list, convert to dict
    if isinstance(all_labels, list):
        print("WARNING: Using same labels for all conditions. Consider passing dict {condition: labels} instead.")
        labels_dict = {cond_name: all_labels for cond_name in all_raw_matrices.keys()}
    elif isinstance(all_labels, dict):
        labels_dict = all_labels
    else:
        raise ValueError("all_labels must be either a list or dict {condition_name: labels_list}")
    
    results = {}
    discriminability_summary = {}
    
    for cond_name, matrices in all_raw_matrices.items():
        # Get labels for this specific condition
        labels = labels_dict[cond_name]
        
        # Handle empty labels
        if len(labels) == 0:
            print(f"\n--- Condition: {cond_name} ---")
            print("WARNING: Empty labels, skipping this condition.")
            continue
        
        print(f"\n--- Condition: {cond_name} ---")
        print(f"Sorting {len(labels)} samples by person for clearer visualization")
        
        # Sort labels and matrices
        sorted_labels, sorted_person_ids, sorted_matrices, _ = _sort_labels_and_matrices(labels, matrices)
        
        print(f"Number of unique persons: {len(set(sorted_person_ids))}")
        print(f"Number of runs: {len(matrices)}")
        
        # Compute mean matrix across runs
        mean_matrix = np.mean(np.array(sorted_matrices), axis=0)
        print(f"Matrix shape: {mean_matrix.shape}")
        
        # Create display labels (only show person name on first image)
        display_labels = _create_display_labels(sorted_labels, sorted_person_ids)
        
        # Visualize using the same style as visualize_local_results
        # This will show both sample-level and class-averaged plots
        visualize_correlation_heatmap(
            correlation_matrix=mean_matrix,
            labels=display_labels,
            title=f"RSA Matrix: {cond_name} {title_suffix}\n(Averaged across {len(matrices)} runs)",
            show_class_averaged=True,  # Show both sample-level and class-averaged
            full_label=full_label
        )
        
        # Compute discriminability metrics (use person IDs for proper within/between calculation)
        print(f"\nDiscriminability Metrics for {cond_name}:")
        within, between, diff = compute_discriminability(mean_matrix, sorted_person_ids)
        discriminability_summary[cond_name] = (within, between, diff)
        print("-" * 40)
        
        results[cond_name] = mean_matrix
    
    # Print summary table
    if discriminability_summary:
        print("\n" + "="*60)
        print("DISCRIMINABILITY SUMMARY")
        print("="*60)
        header = f"{'Condition':<20} {'Within':>10} {'Between':>10} {'Diff':>10}"
        print(header)
        print("-" * 52)
        
        for cond_name, (within, between, diff) in discriminability_summary.items():
            within_str = f"{within:>9.4f}" if not np.isnan(within) else f"{'N/A':>10}"
            between_str = f"{between:>9.4f}" if not np.isnan(between) else f"{'N/A':>10}"
            diff_str = f"{diff:>9.4f}" if not np.isnan(diff) else f"{'N/A':>10}"
            print(f"{cond_name:<20} {within_str} {between_str} {diff_str}")
        
        print("-" * 52)
    
    print("\n" + "="*60)
    print("RSA aggregation and visualization complete!")
    
    return results


def run_multi_run_rsa(model_builder_func, loader, conditions, n_runs=10, device='cuda', layer='IT', metric="pearson"):
    """
    Execute full RSA pipeline across multiple runs for statistical robustness.
    
    Pipeline:
    1. Train/Build models N times per condition
    2. Extract features and compute correlation matrices
    3. Aggregate to find Mean Matrix
    4. Compute quantitative metrics
    
    Parameters:
        model_builder_func: Function that returns trained model for a condition
                           Signature: func(condition_dict) -> model
        loader: DataLoader for test set
        conditions: List of condition dicts with keys: 'name', 'alpha', 'noise', etc.
        n_runs: Number of iterations for statistical averaging
        device: 'cuda' or 'cpu'
        layer: Layer name to extract features from
        metric: 'pearson' or 'cosine'
    
    Returns:
        dict: Results for each condition containing mean matrices and labels
    """
    results = {}
    
    print(f"\nStarting RSA Analysis ({n_runs} runs per condition)...")
    print(f"Device: {device} | Layer: {layer}")
    print("="*60)
    
    for cond in conditions:
        print(f"\nProcessing Condition: {cond['name']}")
        matrices = []
        all_labels = None
        
        # Run N independent trials
        for run_idx in range(n_runs):
            print(f"  Run {run_idx+1}/{n_runs}...", end=" ")
            
            # Build/Train Model
            model = model_builder_func(cond)
            model.eval()
            model.to(device)
            
            # Extract Features
            all_feats = []
            current_labels = []
            
            with torch.no_grad():
                for imgs, lbls, _ in loader:
                    imgs = imgs.to(device)
                    # Extract with GAP
                    feats = extract_features_pytorch(model, [(imgs, lbls, [])], layer=layer, use_gap=True, device=device)[0]
                    all_feats.append(feats)
                    
                    # Convert labels to strings
                    if isinstance(lbls, torch.Tensor):
                        current_labels.extend([str(l.item()) for l in lbls])
                    else:
                        current_labels.extend([str(l) for l in lbls])
            
            # Save labels from first run
            if all_labels is None:
                all_labels = current_labels
            
            # Compute Correlation Matrix
            features_concat = np.concatenate(all_feats, axis=0)
            corr_mat = compute_correlation_pytorch(features_concat, metric=metric)
            matrices.append(corr_mat)
            
            print(f"Matrix shape: {corr_mat.shape}")
            
            # Cleanup
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Compute Mean Matrix
        mean_matrix = np.mean(np.array(matrices), axis=0)
        std_matrix = np.std(np.array(matrices), axis=0)
        
        results[cond['name']] = {
            'matrix': mean_matrix,
            'std_matrix': std_matrix,
            'labels': all_labels,
            'n_runs': n_runs
        }
        
        print(f"  Completed {cond['name']}: Mean matrix computed from {n_runs} runs")
    
    print("\n" + "="*60)
    print("All conditions completed!")
    
    return results


def visualize_rsa_results(results, title_suffix="", save_dir=None, full_label=False):
    """
    Visualize mean matrices and compute quantitative metrics using visualize_local_results style.
    
    Parameters:
        results: dict returned from run_multi_run_rsa()
        title_suffix: optional suffix for plot titles
        save_dir: optional directory to save figures (not yet implemented)
        full_label: bool, if False, only show first image label for each person
    """
    print(f"\nRSA Results Analysis {title_suffix}")
    print("="*60)
    
    discriminability_summary = {}
    
    for name, data in results.items():
        matrix = data['matrix']
        labels = data['labels']
        n_runs = data.get('n_runs', 'N/A')
        
        print(f"\n--- Condition: {name} ---")
        print(f"Matrix shape: {matrix.shape}")
        print(f"Number of runs: {n_runs}")
        
        # Extract person IDs for discriminability calculation
        person_ids = _extract_person_ids(labels)
        
        # Visualize using visualize_local_results style
        visualize_correlation_heatmap(
            correlation_matrix=matrix,
            labels=labels,
            title=f"RSA Matrix: {name} {title_suffix}\n(Averaged across {n_runs} runs)",
            show_class_averaged=True,  # Show both sample-level and class-averaged
            full_label=full_label
        )
        
        # Compute discriminability metrics
        print(f"\nDiscriminability Metrics:")
        within, between, diff = compute_discriminability(matrix, person_ids)
        discriminability_summary[name] = (within, between, diff)
        
        print("-" * 40)
    
    # Print summary table
    if discriminability_summary:
        print("\n" + "="*60)
        print("DISCRIMINABILITY SUMMARY")
        print("="*60)
        header = f"{'Condition':<20} {'Within':>10} {'Between':>10} {'Diff':>10}"
        print(header)
        print("-" * 52)
        
        for name, (within, between, diff) in discriminability_summary.items():
            within_str = f"{within:>9.4f}" if not np.isnan(within) else f"{'N/A':>10}"
            between_str = f"{between:>9.4f}" if not np.isnan(between) else f"{'N/A':>10}"
            diff_str = f"{diff:>9.4f}" if not np.isnan(diff) else f"{'N/A':>10}"
            print(f"{name:<20} {within_str} {between_str} {diff_str}")
        
        print("-" * 52)
    
    # Comparative Bar Plot
    if len(discriminability_summary) > 1:
        print(f"\nGenerating Comparative Discriminability Plot...")
        plot_discriminability_comparison(
            discriminability_summary, 
            metric='discriminability'
        )
    
    print("\n" + "="*60)
    print("Visualization complete!")
    
    return discriminability_summary


# ============================================================================
# SECTION 5: Optional Control Experiments
# ============================================================================

def run_object_control(model_builder_func, object_data_dir, conditions, n_runs=10, device='cuda', batch_size=64):
    """
    Optional: Run RSA analysis on object recognition as control experiment.
    
    Parameters:
        model_builder_func: Same model builder function as face experiment
        object_data_dir: Path to object dataset directory
        conditions: Same conditions as face experiment
        n_runs: Number of runs
        device: 'cuda' or 'cpu'
        batch_size: Batch size for loading
    
    Returns:
        dict: Results for object control experiment
    """
    import os
    from ..data import get_dataloader
    
    # Check if data exists
    if not os.path.exists(object_data_dir):
        print(f"WARNING: Skipping Object Control: Directory {object_data_dir} not found.")
        return None
    
    print(f"\nRunning Object Control RSA Analysis...")
    print(f"Data directory: {object_data_dir}")
    
    # Setup object data loader
    obj_loader = get_dataloader(
        data_dir=object_data_dir,
        batch_size=batch_size,
        target_size=224,
        normalize=True,
        shuffle=False
    )
    
    # Run multi-run RSA
    obj_results = run_multi_run_rsa(
        model_builder_func=model_builder_func,
        loader=obj_loader,
        conditions=conditions,
        n_runs=n_runs,
        device=device,
        layer='IT'
    )
    
    # Visualize results
    visualize_rsa_results(obj_results, title_suffix="(Objects)")
    
    return obj_results


def compare_face_vs_object(face_results, object_results):
    """
    Optional: Compare discriminability between face and object recognition.
    
    Parameters:
        face_results: Results dict from face RSA
        object_results: Results dict from object RSA
    
    Returns:
        Comparative analysis summary
    """
    print(f"\nFace vs Object Comparison")
    print("="*60)
    
    comparison = {}
    
    for cond_name in face_results.keys():
        if cond_name not in object_results:
            continue
        
        face_matrix = face_results[cond_name]['matrix']
        face_labels = face_results[cond_name]['labels']
        
        obj_matrix = object_results[cond_name]['matrix']
        obj_labels = object_results[cond_name]['labels']
        
        # Extract person IDs
        face_person_ids = _extract_person_ids(face_labels)
        obj_person_ids = _extract_person_ids(obj_labels)
        
        # Compute discriminability for both
        face_within, face_between, face_diff = compute_discriminability(face_matrix, face_person_ids)
        obj_within, obj_between, obj_diff = compute_discriminability(obj_matrix, obj_person_ids)
        
        comparison[cond_name] = {
            'face_diff': face_diff,
            'object_diff': obj_diff,
            'difference': face_diff - obj_diff
        }
        
        print(f"\n{cond_name}:")
        print(f"  Face Diff:   {face_diff:.4f}")
        print(f"  Object Diff: {obj_diff:.4f}")
        print(f"  Difference (Face - Obj): {face_diff - obj_diff:.4f}")
    
    print("\n" + "="*60)
    
    return comparison
