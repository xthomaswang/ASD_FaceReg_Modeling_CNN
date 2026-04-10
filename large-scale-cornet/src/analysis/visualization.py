# src/analysis/visualization.py

"""
Visualization functions for correlation matrices and training metrics
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as hc
from collections import Counter, defaultdict


# ==========================================
# Internal helpers (not part of the public API)
# ==========================================
def _set_seaborn_style(font_scale: float = 2.5) -> None:
    """Apply a consistent seaborn style across plots."""
    sns.set(font_scale=font_scale)


def _as_key_extractor(key_or_index):
    """
    Convert an integer index or a callable into a callable extractor.
    If key_or_index is an int, returns lambda k: k[key_or_index].
    """
    if isinstance(key_or_index, int):
        idx = key_or_index
        return lambda k: k[idx]
    return key_or_index


def _coerce_runs(value):
    """
    Normalize a metrics value into a list of runs.
    Expected formats:
      - list[runs], where each run is list/array
      - a single run (list/array)
    """
    if value is None:
        return []
    if isinstance(value, list):
        # Heuristic: a list of numbers is a single run; a list of lists is multiple runs.
        if value and not isinstance(value[0], (list, tuple, np.ndarray)):
            return [value]
        return value
    if isinstance(value, (tuple, np.ndarray)):
        return [list(value)]
    return []


def _final_mean_std(runs):
    """Compute (mean, std) for the final value across runs. Returns None if empty."""
    runs = _coerce_runs(runs)
    finals = [r[-1] for r in runs if r]
    if not finals:
        return None
    arr = np.asarray(finals, dtype=float)
    return float(arr.mean()), float(arr.std())


def _timeseries_mean_std(runs):
    """
    Compute (x_range, mean, std) across runs aligned to the shortest run length.
    Returns None if empty.
    """
    runs = _coerce_runs(runs)
    lengths = [len(r) for r in runs if r]
    if not lengths:
        return None
    min_len = min(lengths)
    if min_len <= 0:
        return None
    arr = np.asarray([r[:min_len] for r in runs if r], dtype=float)
    x = range(min_len)
    return x, arr.mean(axis=0), arr.std(axis=0)


# ==========================================
# Label Processing Helpers
# ==========================================
def _extract_person_names_from_labels(labels):
    """
    Extract person names from labels (remove image numbers).
    
    Parameters:
        labels: list of labels (e.g., ["Person_0", "Person_1", ...])
        
    Returns:
        list: person names without image numbers
    """
    if not labels:
        return []
    
    return [
        label.rsplit('_', 1)[0] if (isinstance(label, str) and '_' in label) else str(label)
        for label in labels
    ]


def _create_first_occurrence_labels(labels, person_names, max_chars=15):
    """
    Create labels showing only first occurrence of each person.
    
    Parameters:
        labels: list of full labels
        person_names: list of person names (parallel to labels)
        max_chars: maximum characters to show per label
        
    Returns:
        list: labels with empty strings for duplicate persons
    """
    display_labels = []
    prev_name = None
    
    for i, name in enumerate(person_names):
        if name != prev_name:
            # First occurrence - show the name
            display_labels.append(name[:max_chars])
            prev_name = name
        else:
            # Duplicate - use empty string
            display_labels.append('')
    
    return display_labels


def _create_class_midpoint_labels(person_names, max_chars=10):
    """
    Create labels at class midpoints for full_label=True mode.
    
    Parameters:
        person_names: list of person names (parallel to samples)
        max_chars: maximum characters to show per label
        
    Returns:
        tuple: (tick_locs, tick_labels)
    """
    # Find class boundaries
    boundaries = [0]
    for i in range(1, len(person_names)):
        if person_names[i] != person_names[i-1]:
            boundaries.append(i)
    boundaries.append(len(person_names))
    
    # Use midpoints of each class as tick locations
    tick_locs = [(boundaries[i] + boundaries[i+1]) // 2 for i in range(len(boundaries)-1)]
    tick_labels = [person_names[loc][:max_chars] for loc in tick_locs]
    
    # Limit to reasonable number of ticks
    max_ticks = 50
    if len(tick_locs) > max_ticks:
        step = len(tick_locs) // max_ticks
        tick_locs = tick_locs[::step]
        tick_labels = tick_labels[::step]
    
    return tick_locs, tick_labels


def _sort_matrix_by_person(correlation_matrix, labels):
    """
    Sort correlation matrix and labels by person name to group same person's images together.
    
    Parameters:
        correlation_matrix: 2D numpy array
        labels: list of labels
        
    Returns:
        tuple: (sorted_matrix, sorted_labels)
    """
    if labels is None or len(labels) == 0:
        return correlation_matrix, labels
    
    N = correlation_matrix.shape[0]
    
    print(f"[INFO] Person Name Sorting ... Started (N={N})")
    
    # Extract person names (without image numbers) - vectorized
    person_names = np.array(_extract_person_names_from_labels(labels))
    
    # Use numpy argsort for better performance on large arrays
    # First sort by person name, then by original index to maintain within-person order
    sort_idx = np.lexsort((np.arange(len(labels)), person_names))
    
    # Reorder matrix (both rows and columns) using advanced indexing
    print(f"[INFO] Reordering matrix ({N}x{N})...")
    sorted_matrix = correlation_matrix[np.ix_(sort_idx, sort_idx)]
    
    # Reorder labels
    sorted_labels = [labels[i] for i in sort_idx]
    
    print(f"[INFO] Person Name Sorting ... Completed.")
    
    return sorted_matrix, sorted_labels


# ==========================================
# Class-Averaged Matrix Computation
# ==========================================
def compute_class_averaged_matrix_flexible(full_matrix, person_names):
    """
    Compute class-averaged matrix with flexible handling of unequal images per class.
    
    This function manually computes the average correlation between each pair of classes,
    handling cases where different classes have different numbers of images.
    
    Parameters:
        full_matrix: 2D numpy array of shape (n_samples, n_samples)
        person_names: list of person names for each sample (parallel to matrix rows/cols)
        
    Returns:
        averaged_matrix: 2D numpy array of shape (n_classes, n_classes)
        unique_persons: list of unique person names (class labels)
    """
    unique_persons = list(dict.fromkeys(person_names))  # Preserve order
    n_classes = len(unique_persons)
    
    # Create averaged matrix
    averaged_matrix = np.zeros((n_classes, n_classes), dtype=np.float32)
    
    # For each pair of classes, compute average correlation
    for i, person_i in enumerate(unique_persons):
        # Get indices for person i
        indices_i = [idx for idx, name in enumerate(person_names) if name == person_i]
        
        for j, person_j in enumerate(unique_persons):
            # Get indices for person j
            indices_j = [idx for idx, name in enumerate(person_names) if name == person_j]
            
            # Extract submatrix for this class pair
            submatrix = full_matrix[np.ix_(indices_i, indices_j)]
            
            # Average all correlations in this block
            averaged_matrix[i, j] = submatrix.mean()
    
    return averaged_matrix, unique_persons


def compute_class_averaged_matrix(full_matrix, imgs_per_class):
    """
    Compress (N*M, N*M) matrix to (N, N) by averaging within-class blocks.
    
    Example: 10000x10000 -> 100x100 (if imgs_per_class=100)
    
    Parameters:
        full_matrix: 2D numpy array of shape (n_samples, n_samples)
        imgs_per_class: int, number of images per class/person
        
    Returns:
        averaged_matrix: 2D numpy array of shape (n_classes, n_classes)
    """
    n_samples = full_matrix.shape[0]
    n_classes = n_samples // imgs_per_class
    
    # Calculate how many samples will be used (might truncate if not perfectly divisible)
    usable_samples = n_classes * imgs_per_class
    
    if usable_samples < n_samples:
        print(f"[Warning] Truncating {n_samples - usable_samples} samples to fit {n_classes}×{imgs_per_class} grid")
        full_matrix = full_matrix[:usable_samples, :usable_samples]
    
    # Reshape to (n_classes, imgs_per_class, n_classes, imgs_per_class)
    # Then average over axis 1 and 3 to get class-level correlations
    reshaped = full_matrix.reshape(n_classes, imgs_per_class, n_classes, imgs_per_class)
    averaged_matrix = reshaped.mean(axis=(1, 3))
    
    return averaged_matrix


def _check_class_balance(labels):
    """
    Check if all classes have equal number of images.
    
    Parameters:
        labels: list of labels (should be in "PersonName_N" format)
        
    Returns:
        tuple: (is_balanced, imgs_per_class, unique_persons, person_counts)
               If is_balanced=False, imgs_per_class=-1
    """
    if not (isinstance(labels[0], str) and '_' in str(labels[0])):
        return False, -1, [], {}
    
    person_names = _extract_person_names_from_labels(labels)
    person_counts = Counter(person_names)
    unique_persons = list(dict.fromkeys(person_names))
    
    counts_values = list(person_counts.values())
    unique_counts = set(counts_values)
    
    is_balanced = len(unique_counts) == 1
    imgs_per_class = counts_values[0] if is_balanced else -1
    
    return is_balanced, imgs_per_class, unique_persons, person_counts


def _print_class_imbalance_report(labels, person_counts, unique_persons):
    """
    Print detailed report about class imbalance.
    
    Parameters:
        labels: list of labels
        person_counts: Counter object with person -> count mapping
        unique_persons: list of unique person names
    """
    counts_values = list(person_counts.values())
    min_count = min(counts_values)
    max_count = max(counts_values)
    unique_counts = set(counts_values)
    
    print(f"\n{'='*70}")
    print(f"❌ CLASS IMAGE IMBALANCE DETECTED - Skipping class-averaged plot")
    print(f"{'='*70}")
    print(f"[Summary]")
    print(f"  - Total samples: {len(labels)}")
    print(f"  - Total classes: {len(unique_persons)}")
    print(f"  - Images per class range: {min_count} to {max_count}")
    print(f"  - Unique counts: {sorted(unique_counts)}")
    
    # Show distribution
    count_dist = {count: sum(1 for c in counts_values if c == count) for count in unique_counts}
    print(f"\n[Distribution]")
    for count, num_classes in sorted(count_dist.items()):
        print(f"  - {num_classes} classes have {count} images")
    
    # Show ALL irregular classes
    print(f"\n[Irregular Classes - Full List]")
    irregular_classes = [(name, count) for name, count in sorted(person_counts.items()) if count != max_count]
    
    if len(irregular_classes) == 0:
        print("  No irregular classes (all have max count)")
    else:
        print(f"  Total irregular classes: {len(irregular_classes)}")
        print(f"  Expected: {max_count} images per class\n")
        
        # Group by count for easier reading
        by_count = defaultdict(list)
        for name, count in irregular_classes:
            by_count[count].append(name)
        
        for count in sorted(by_count.keys()):
            classes = by_count[count]
            print(f"  Classes with {count} images ({len(classes)} classes):")
            for name in classes:
                print(f"    - {name}")
    
    print(f"\n{'='*70}")
    print("Please ensure all classes have equal number of images for class-averaged RSA.")
    print(f"{'='*70}\n")


# ==========================================
# Sample-Level Matrix Visualization
# ==========================================
def _plot_sample_level_matrix_large(correlation_matrix, labels, title, full_label):
    """
    Plot sample-level matrix for large matrices using imshow.
    
    Parameters:
        correlation_matrix: 2D numpy array
        labels: list of labels
        title: plot title
        full_label: bool, whether to show full labels or only first occurrence
    """
    N = correlation_matrix.shape[0]
    fig_width, fig_height = 12, 10
    dpi = 150
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    _set_seaborn_style(font_scale=2.5)
    
    # Use imshow for high-performance rendering
    im = ax.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest', 
                   aspect='auto', vmin=-1, vmax=1, origin='upper')
    
    cbar = plt.colorbar(im, ax=ax, label='Correlation', shrink=0.8)
    cbar.ax.tick_params(labelsize=10)
    
    ax.set_title(title, fontsize=20)
    ax.set_xlabel('Sample Index', fontsize=14)
    ax.set_ylabel('Sample Index', fontsize=14)
    
    # Smart tick label handling for large matrices
    if labels is not None and isinstance(labels[0], str) and '_' in str(labels[0]):
        person_names = _extract_person_names_from_labels(labels)
        
        if full_label:
            # Show labels at class midpoints
            tick_locs, tick_labels = _create_class_midpoint_labels(person_names, max_chars=10)
        else:
            # Show only first image of each person (OPTIMIZED for large matrices)
            tick_locs, tick_labels = [], []
            prev_name = None
            for i, name in enumerate(person_names):
                if name != prev_name:
                    tick_locs.append(i)
                    tick_labels.append(name[:15])
                    prev_name = name
            
            # Still limit to max ticks if too many people
            max_ticks = 100
            if len(tick_locs) > max_ticks:
                step = len(tick_locs) // max_ticks
                tick_locs = tick_locs[::step]
                tick_labels = tick_labels[::step]
        
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=8)
        ax.set_yticks(tick_locs)
        ax.set_yticklabels(tick_labels, rotation=0, fontsize=8)
    
    # Invert y-axis to match seaborn style (diagonal from top-left to bottom-right)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.show()


def _plot_sample_level_matrix_small(correlation_matrix, labels, title, figsize, full_label):
    """
    Plot sample-level matrix for small matrices using seaborn heatmap.
    
    Parameters:
        correlation_matrix: 2D numpy array
        labels: list of labels
        title: plot title
        figsize: tuple, figure size
        full_label: bool, whether to show full labels or only first occurrence
    """
    plt.figure(figsize=figsize)
    _set_seaborn_style(font_scale=2.5)
    
    # Apply full_label logic
    if not full_label and labels is not None and len(labels) > 0:
        # Show only first image of each person
        person_names = _extract_person_names_from_labels(labels)
        tick_labels_x = _create_first_occurrence_labels(labels, person_names, max_chars=15)
        tick_labels_y = tick_labels_x.copy()
        
        ax = sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', 
                         xticklabels=tick_labels_x, yticklabels=tick_labels_y, 
                         cbar_kws={'label': 'Correlation'})
    else:
        # Show all labels (original behavior)
        ax = sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', 
                         xticklabels=labels, yticklabels=labels, 
                         cbar_kws={'label': 'Correlation'})
    
    ax.set_title(title, fontsize=20)
    ax.set_xlabel('Sample Index', fontsize=14)
    ax.set_ylabel('Sample Index', fontsize=14)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    ax.invert_yaxis()  # Flip y-axis so diagonal goes from top-left to bottom-right
    plt.tight_layout()
    plt.show()


# ==========================================
# Class-Averaged Matrix Visualization
# ==========================================
def _plot_class_averaged_matrix(avg_matrix, avg_labels, title):
    """
    Plot class-averaged correlation matrix.
    
    Parameters:
        avg_matrix: 2D numpy array of class-averaged correlations
        avg_labels: list of class labels
        title: plot title
    """
    n_classes = len(avg_labels)
    
    if n_classes <= 150:
        # Use seaborn for cleaner look with reasonable size
        fig_size = min(60, max(20, n_classes * 0.5))
        plt.figure(figsize=(fig_size, fig_size * 0.8))
        _set_seaborn_style(font_scale=2.0)
        
        # Control tick label frequency to avoid overcrowding
        if n_classes <= 50:
            # Show all labels for small number of classes
            x_labels = avg_labels
            y_labels = avg_labels
        else:
            # Show labels at regular intervals
            label_step = max(1, n_classes // 50)
            x_labels = [avg_labels[i] if i % label_step == 0 else '' for i in range(n_classes)]
            y_labels = [avg_labels[i] if i % label_step == 0 else '' for i in range(n_classes)]
        
        ax = sns.heatmap(avg_matrix, annot=False, cmap='coolwarm',
                         xticklabels=x_labels, yticklabels=y_labels,
                         cbar_kws={'label': 'Average Correlation'})
        ax.set_title(f"{title}\n[Class-Averaged: {n_classes} classes, Matrix: {avg_matrix.shape[0]}×{avg_matrix.shape[1]}]", 
                    fontsize=18)
        ax.set_xlabel('Person/Class', fontsize=14)
        ax.set_ylabel('Person/Class', fontsize=14)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        ax.invert_yaxis()
        
    else:
        # Use imshow for very large class-level matrices (>150 classes)
        fig, ax = plt.subplots(figsize=(14, 12), dpi=120)
        _set_seaborn_style(font_scale=2.0)
        
        im = ax.imshow(avg_matrix, cmap='coolwarm', interpolation='nearest',
                       aspect='auto', vmin=-1, vmax=1, origin='upper')
        cbar = plt.colorbar(im, ax=ax, label='Average Correlation', shrink=0.8)
        cbar.ax.tick_params(labelsize=10)
        
        ax.set_title(f"{title}\n[Class-Averaged: {n_classes} classes, Matrix: {avg_matrix.shape[0]}×{avg_matrix.shape[1]}]", 
                    fontsize=18)
        ax.set_xlabel('Person/Class', fontsize=14)
        ax.set_ylabel('Person/Class', fontsize=14)
        
        # Show subset of tick labels
        step = max(1, n_classes // 50)
        tick_locs = np.arange(0, n_classes, step)
        tick_labels = [avg_labels[i] for i in tick_locs]
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=8)
        ax.set_yticks(tick_locs)
        ax.set_yticklabels(tick_labels, rotation=0, fontsize=8)
        ax.invert_yaxis()
        
        print(f"[DEBUG] imshow rendering {avg_matrix.shape[0]}×{avg_matrix.shape[1]} matrix")
    
    plt.tight_layout()
    plt.show()


# ==========================================
# Optimized Framework-Agnostic Functions
# ==========================================
def visualize_training_comparisons(histories, figsize=(60, 48)):
    """
    Visualizes comparative training dynamics across multiple conditions.
    
    Generates a 2x2 grid separating Training Accuracy, Validation Accuracy,
    Training Loss, and Validation Loss into distinct plots for clarity.
    
    Parameters:
        histories: dict
            Dictionary where keys are condition names (str) and values are 
            history dictionaries containing 'train_acc', 'val_acc', 'train_loss', 'val_loss'.
        figsize: tuple
            Size of the entire figure (width, height).
    """
    _set_seaborn_style(font_scale=2.5)
    
    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Define plot configurations: (axis_idx, metric_key, title, ylabel)
    plot_configs = [
        (0, 'train_acc', 'Training Accuracy', 'Accuracy (%)'),
        (1, 'val_acc',   'Validation Accuracy', 'Accuracy (%)'),
        (2, 'train_loss', 'Training Loss', 'Loss'),
        (3, 'val_loss',   'Validation Loss', 'Loss')
    ]
    
    # Define a color palette to ensure consistency across subplots
    condition_names = list(histories.keys())
    colors = sns.color_palette("bright", n_colors=len(condition_names))
    color_map = dict(zip(condition_names, colors))

    for ax_idx, metric, title, ylabel in plot_configs:
        ax = axes[ax_idx]
        has_data = False
        
        for name, history in histories.items():
            # Check if metric exists in history
            if metric in history and history[metric]:
                data = history[metric]
                ax.plot(data, label=name, color=color_map[name], linewidth=2.5, alpha=0.85)
                has_data = True
        
        # Styling the subplot
        ax.set_title(title, fontsize=16, fontweight='bold', pad=12)
        ax.set_xlabel('Epochs', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        
        if has_data:
            ax.legend(fontsize=10, loc='best')
            ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()


def visualize_correlation_heatmap(correlation_matrix, labels, title="Correlation Matrix", figsize=(120, 96), 
                                  show_class_averaged=True, full_label=False):
    """
    Universal correlation matrix visualization with performance optimization.
    
    Generates two separate plots:
    1. Sample-level correlation matrix (always shown)
    2. Class-averaged correlation matrix (optional, for large matrices with structured labels)
    
    This function is framework-agnostic and works with any extracted features.
    
    Parameters:
        correlation_matrix: 2D numpy array of shape (N, N) containing correlation coefficients
        labels: list of strings, one label per sample (e.g., person names or image IDs)
        title: string, title for the plot
        figsize: tuple, figure size in inches (default: (120, 96))
        show_class_averaged: bool, whether to also show class-averaged plot for large matrices
        full_label: bool, if False, only show the first image label for each person (default: False)
    
    Output:
        Displays heatmap plot(s)
    """
    N = correlation_matrix.shape[0]
    
    # Sort matrix and labels by person name to group same person's images together
    correlation_matrix, labels = _sort_matrix_by_person(correlation_matrix, labels)
    
    is_huge = N > 500
    
    # ========== Plot 1: Full Sample-Level Matrix (ALWAYS SHOWN) ==========
    print(f"\n[Plot 1/2] Sample-Level Correlation Matrix ({N}×{N})")
    print("-" * 70)
    
    if is_huge:
        _plot_sample_level_matrix_large(correlation_matrix, labels, title, full_label)
    else:
        _plot_sample_level_matrix_small(correlation_matrix, labels, title, figsize, full_label)
    
    print(f"[Plot 1/2] Sample-level visualization complete!")
    
    # ========== Plot 2: Class-Averaged Matrix (OPTIONAL, SEPARATE PLOT) ==========
    if not show_class_averaged:
        return None
    
    if not is_huge:
        print(f"\n[Info] Matrix too small ({N} samples), skipping class-averaged plot.")
        return None
    
    if labels is None:
        print(f"\n[Info] No labels provided, skipping class-averaged plot.")
        return None
    
    print(f"\n[Plot 2/2] Class-Averaged Correlation Matrix")
    print("-" * 70)
    
    # Check class balance
    is_balanced, imgs_per_class, unique_persons, person_counts = _check_class_balance(labels)
    
    # Debug info
    person_names = _extract_person_names_from_labels(labels)
    print(f"[DEBUG] Total samples: {len(labels)}")
    print(f"[DEBUG] Unique classes: {len(unique_persons) if unique_persons else 'N/A'}")
    print(f"[DEBUG] First 5 labels: {labels[:5]}")
    print(f"[DEBUG] First 5 person_names: {person_names[:5]}")
    
    if not is_balanced:
        _print_class_imbalance_report(labels, person_counts, unique_persons)
        return None
    
    # EQUAL: Proceed with class-averaged plot
    print(f"\n[Class-Averaged] ✓ All classes have {imgs_per_class} images each")
    print(f"[Class-Averaged] Compressing {N}x{N} matrix to {len(unique_persons)}x{len(unique_persons)}...")
    
    # Compute class-averaged matrix using fast reshape method
    avg_matrix = compute_class_averaged_matrix(correlation_matrix, imgs_per_class)
    avg_labels = unique_persons
    
    # Plot the class-averaged matrix
    print(f"[DEBUG] avg_matrix shape after averaging: {avg_matrix.shape}")
    print(f"[DEBUG] Expected shape: ({len(avg_labels)}, {len(avg_labels)})")
    
    _plot_class_averaged_matrix(avg_matrix, avg_labels, title)
    
    print(f"[Plot 2/2] Class-averaged visualization complete!")
    
    return None


def compute_discriminability(correlation_matrix, labels):
    """
    Compute within-class vs between-class correlation using upper triangle pairs.
    
    This implementation uses only upper triangle (i < j) to avoid double counting
    and explicitly excludes the diagonal. More robust than full-matrix masking.
    
    Parameters:
        correlation_matrix: 2D numpy array of shape (N, N)
        labels: list or array of labels for each sample (must be person names, NOT unique IDs)
                CORRECT: ["George_Bush", "George_Bush", "Colin_Powell", ...]
                WRONG: ["George_Bush_0", "George_Bush_1", "Colin_Powell_0", ...]
        
    Returns:
        tuple: (within_score, between_score, discriminability_diff)
    
    Note:
        Use get_person_name_labels(loader, indices, add_image_number=False) 
        to generate correct labels.
    """
    labels = np.asarray(labels)
    N = correlation_matrix.shape[0]
    
    assert correlation_matrix.shape == (N, N), f"Expected square matrix, got {correlation_matrix.shape}"
    assert labels.shape[0] == N, f"Labels length {labels.shape[0]} != matrix size {N}"
    
    # Upper triangle indices (exclude diagonal)
    iu = np.triu_indices(N, k=1)
    
    # Determine same/different class for each upper triangle pair
    same_class = labels[iu[0]] == labels[iu[1]]
    diff_class = ~same_class
    
    # Extract correlation values
    within_vals = correlation_matrix[iu][same_class]
    between_vals = correlation_matrix[iu][diff_class]
    
    # Compute means (handle empty arrays gracefully)
    within_score = float(np.nanmean(within_vals)) if within_vals.size > 0 else float('nan')
    between_score = float(np.nanmean(between_vals)) if between_vals.size > 0 else float('nan')
    diff = within_score - between_score
    
    print(f"Within-class Correlation: {within_score:.4f} (n={within_vals.size} pairs)")
    print(f"Between-class Correlation: {between_score:.4f} (n={between_vals.size} pairs)")
    print(f"Diff (Within - Between): {diff:.4f}")
    
    return within_score, between_score, diff


def plot_discriminability_comparison(results_dict, metric='discriminability'):
    """
    Compare discriminability across different conditions (e.g., control vs pathology).
    
    Parameters:
        results_dict: dict mapping condition names to (within, between, discriminability) tuples
        metric: which metric to plot ('discriminability', 'within', or 'between')
    
    Output:
        Bar plot comparing conditions
    """
    _set_seaborn_style(font_scale=2.5)
    
    conditions = list(results_dict.keys())
    
    if metric == 'discriminability':
        values = [results_dict[c][2] for c in conditions]
        ylabel = 'Diff (Within - Between)'
    elif metric == 'within':
        values = [results_dict[c][0] for c in conditions]
        ylabel = 'Within-class Correlation'
    elif metric == 'between':
        values = [results_dict[c][1] for c in conditions]
        ylabel = 'Between-class Correlation'
    else:
        raise ValueError("metric must be 'discriminability', 'within', or 'between'")
    
    plt.figure(figsize=(60, 30))
    plt.bar(conditions, values, color=['green', 'orange', 'red', 'darkred'][:len(conditions)])
    plt.ylabel(ylabel, fontsize=14)
    plt.xlabel('Condition', fontsize=14)
    plt.title(f'{ylabel} Across Conditions', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==========================================
# Generic Plotting Functions
# ==========================================
def plot_grouped_bar_comparison(
    metrics_dict,
    group_by_key,
    group_values,
    x_axis_key,
    group_label='Group',
    x_label='X Parameter',
    y_label='Performance',
    title='Performance Comparison',
    figsize_per_group=(6, 5),
    y_limits=None,
    color='steelblue',
    alpha=0.85
):
    """
    Generic function: Plot grouped bar charts comparing final performance across conditions.
    
    Parameters:
        metrics_dict: dict
            Keys are tuples of any structure, values are lists of metrics
        group_by_key: callable or int
            If callable: function to extract grouping value from key, e.g., lambda k: k[0]
            If int: use the nth element of key for grouping
        group_values: list
            List of group values to display (preserves order)
        x_axis_key: callable or int
            If callable: function to extract x-axis value from key
            If int: use the nth element of key for x-axis
        group_label: str
            Label for the grouping parameter (used in subplot titles)
        x_label: str
            Label for x-axis
        y_label: str
            Label for y-axis
        title: str
            Overall plot title
        figsize_per_group: tuple
            Size of each subplot (width, height)
        y_limits: tuple or None
            Y-axis range, e.g., (0, 1.05)
        color: str
            Bar chart color
        alpha: float
            Transparency level
    
    Example:
        plot_grouped_bar_comparison(
            train_accuracies,
            group_by_key=lambda k: k[0],
            group_values=[0.005, 0.05, 0.5],
            x_axis_key=lambda k: k[3],
            group_label='Positive Slope',
            x_label='Threshold',
            y_label='Mean Final Accuracy',
            title='Training Accuracy Comparison',
            y_limits=(0, 1.05)
        )
    """
    import pandas as pd

    group_by_key = _as_key_extractor(group_by_key)
    x_axis_key = _as_key_extractor(x_axis_key)

    records = []
    for key, value in metrics_dict.items():
        try:
            stats = _final_mean_std(value)
            if stats is None:
                continue
            mean, std = stats
            records.append({
                'group': group_by_key(key),
                'x_value': x_axis_key(key),
                'mean': mean,
                'std': std
            })
        except Exception:
            continue

    if not records:
        print("No data available for plotting.")
        return

    df = pd.DataFrame(records)
    
    # Create subplots
    n_groups = len(group_values)
    fig, axes = plt.subplots(1, n_groups, figsize=(figsize_per_group[0]*n_groups, figsize_per_group[1]), squeeze=False)
    axes = axes.flatten()
    
    for idx, group_val in enumerate(group_values):
        ax = axes[idx]
        group_df = df[df['group'] == group_val].sort_values('x_value')
        
        if group_df.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{group_label} = {group_val}')
            continue
        
        # Plot bar chart
        x_positions = np.arange(len(group_df))
        ax.bar(x_positions, group_df['mean'], yerr=group_df['std'],
               capsize=5, color=color, alpha=alpha)
        
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(f'{group_label} = {group_val}', fontsize=13)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(group_df['x_value'], rotation=45)
        
        if y_limits:
            ax.set_ylim(y_limits)
        
        ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    fig.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_grouped_training_curves(
    metrics_dict,
    group_by_key,
    group_values,
    curve_by_key,
    group_label='Group',
    curve_label='Curve',
    x_label='Epochs',
    y_label='Metric',
    title='Training Curves Comparison',
    figsize_per_group=(7, 5),
    layout='single',
    y_limits=None,
    colormap='viridis',
    linewidth=2,
    alpha_fill=0.2,
    legend_fontsize=9,
    group_colormap='tab10',
    linestyles=('-', '--', '-.', ':')
):
    """
    Generic function: Plot grouped training curves for comparing multiple conditions.
    
    Parameters:
        metrics_dict: dict
            Keys are tuples of any structure, values are lists of time-series metrics
        group_by_key: callable or int
            Extract grouping value from key (used to create subplots)
        group_values: list
            List of group values to display
        curve_by_key: callable or int
            Extract curve classification value from key (different colors within same subplot)
        group_label: str
            Label for grouping parameter
        curve_label: str
            Label for curve classification (used in legend)
        x_label: str
            Label for x-axis
        y_label: str
            Label for y-axis
        title: str
            Overall plot title
        figsize_per_group: tuple
            Size of each subplot
        layout: str
            'single': plot all group values on a single axes (shared x/y)
            'subplots': one subplot per group value
        y_limits: tuple or None
            Y-axis range
        colormap: str
            Name of colormap
        linewidth: float
            Width of lines
        alpha_fill: float
            Transparency of standard deviation region
        legend_fontsize: int
            Font size for legend
        group_colormap: str
            Colormap used to assign colors to groups when layout='single'
        linestyles: tuple[str, ...]
            Line style cycle used to distinguish curve values when layout='single'
    
    Example:
        plot_grouped_training_curves(
            train_accuracies,
            group_by_key=0,
            group_values=[0.005, 0.05, 0.5],
            curve_by_key=3,
            group_label='Positive Slope',
            curve_label='Threshold',
            y_label='Accuracy',
            title='Training Accuracy Over Time'
        )
    """
    import matplotlib

    group_by_key = _as_key_extractor(group_by_key)
    curve_by_key = _as_key_extractor(curve_by_key)

    def _iter_group_data(target_group_val):
        """Return {curve_val: list[runs]} for a given group value, merging compatible entries."""
        group_data = {}
        for key, value in metrics_dict.items():
            try:
                if group_by_key(key) != target_group_val:
                    continue
                curve_val = curve_by_key(key)
                runs = _coerce_runs(value)
                if not runs:
                    continue
                if curve_val in group_data:
                    group_data[curve_val].extend(runs)
                else:
                    group_data[curve_val] = list(runs)
            except Exception:
                continue
        return group_data

    layout = (layout or 'subplots').lower()

    if layout == 'single':
        fig = plt.figure(figsize=(figsize_per_group[0], figsize_per_group[1]))
        ax = plt.gca()

        # Color by group, linestyle by curve value
        group_cmap = matplotlib.colormaps.get_cmap(group_colormap)
        group_colors = [group_cmap(i) for i in np.linspace(0, 1, max(len(group_values), 2))]

        for g_i, group_val in enumerate(group_values):
            group_data = _iter_group_data(group_val)
            if not group_data:
                continue

            curve_values = sorted(group_data.keys())
            for c_i, curve_val in enumerate(curve_values):
                runs = group_data[curve_val]
                if not runs:
                    continue
                stats = _timeseries_mean_std(runs)
                if stats is None:
                    continue
                epochs_range, means, stds = stats

                color = group_colors[g_i % len(group_colors)]
                linestyle = linestyles[c_i % len(linestyles)] if linestyles else '-'
                label = f'{group_label}={group_val}, {curve_label}={curve_val}'

                ax.plot(
                    epochs_range,
                    means,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    label=label
                )
                ax.fill_between(
                    epochs_range,
                    means - stds,
                    means + stds,
                    alpha=alpha_fill,
                    color=color
                )

        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.grid(True, linestyle='--', alpha=0.6)
        if y_limits:
            ax.set_ylim(y_limits)

        ax.legend(fontsize=legend_fontsize, loc='best')
        plt.tight_layout()
        plt.show()
        return

    # Default behavior: one subplot per group value
    n_groups = len(group_values)
    fig, axes = plt.subplots(
        1,
        n_groups,
        figsize=(figsize_per_group[0] * n_groups, figsize_per_group[1]),
        squeeze=False
    )
    axes = axes.flatten()

    for idx, group_val in enumerate(group_values):
        ax = axes[idx]

        group_data = _iter_group_data(group_val)
        if not group_data:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{group_label} = {group_val}')
            continue

        curve_values = sorted(group_data.keys())
        cmap = matplotlib.colormaps.get_cmap(colormap)
        colors = [cmap(i) for i in np.linspace(0, 1, len(curve_values))]

        for i, curve_val in enumerate(curve_values):
            runs = group_data[curve_val]
            if not runs:
                continue

            stats = _timeseries_mean_std(runs)
            if stats is None:
                continue
            epochs_range, means, stds = stats

            ax.plot(
                epochs_range,
                means,
                color=colors[i],
                linewidth=linewidth,
                label=f'{curve_label} = {curve_val}'
            )
            ax.fill_between(
                epochs_range,
                means - stds,
                means + stds,
                alpha=alpha_fill,
                color=colors[i]
            )

        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(f'{group_label} = {group_val}', fontsize=13)
        ax.legend(fontsize=legend_fontsize, loc='best')
        ax.grid(True, linestyle='--', alpha=0.6)
        if y_limits:
            ax.set_ylim(y_limits)

    fig.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


def visualize_local_results(
    run_dir,
    conditions=None,
    show_training_curves=True,
    show_rsa_matrices=True,
    splits=["train", "test"],
    run_idx=0,
    full_label=False
):
    """
    Visualize saved experimental results from local directory without retraining.
    
    This function reads pre-saved training histories and RSA matrices from disk
    and generates publication-ready plots using existing visualization functions.
    
    Parameters:
        run_dir: str
            Path to the run directory containing saved results.
            Example: "/path/to/results/EIB/cornet/100_1_20_5_2026-01-16_15-00-00/run_0"
        
        conditions: list[str], optional
            List of condition names to visualize (e.g., ["Balanced", "Excitated", "Inhibitated"]).
            If None, auto-detects from available matrix files.
        
        show_training_curves: bool, default=True
            Whether to plot training/validation accuracy and loss curves.
        
        show_rsa_matrices: bool, default=True
            Whether to plot RSA correlation matrices.
        
        splits: list[str], default=["train", "test"]
            Which data splits to visualize for RSA. Can be ["train"], ["test"], or ["train", "test"].
        
        run_idx: int, default=0
            Which run to visualize (if multiple runs exist, shows matrices from this run index).
        
        full_label: bool, default=False
            If False, only show the first image label for each person (more efficient for large matrices).
            If True, show labels at class midpoints.
    
    Returns:
        dict: Summary statistics including final accuracies and discriminability scores.
    """
    import json
    import os
    from pathlib import Path
    
    # Convert single string to list for backward compatibility
    if isinstance(splits, str):
        splits = [splits]
    
    run_path = Path(run_dir)
    
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    print(f"Loading results from: {run_dir}")
    print("=" * 70)
    
    summary = {}
    
    # ==========================================
    # 1. Load and Visualize Training Histories
    # ==========================================
    if show_training_curves:
        history_file = run_path / "all_raw_histories.json"
        
        if history_file.exists():
            with open(history_file, 'r') as f:
                all_histories = json.load(f)
            
            # Convert to format expected by visualize_training_comparisons
            processed_histories = {}
            for cond_name, runs in all_histories.items():
                if len(runs) > 0:
                    if len(runs) == 1:
                        processed_histories[cond_name] = runs[0]
                    else:
                        # Average metrics across runs
                        avg_hist = {}
                        metrics = ["train_acc", "val_acc", "train_loss", "val_loss"]
                        for metric in metrics:
                            vals = [np.array(h.get(metric, []), dtype=float) for h in runs if h.get(metric)]
                            if vals:
                                avg_hist[metric] = np.mean(np.stack(vals, axis=0), axis=0).tolist()
                        
                        # Average final test accuracy across runs
                        test_accs = [h.get("final_test_acc") for h in runs if "final_test_acc" in h]
                        if test_accs:
                            avg_hist["final_test_acc"] = np.mean(test_accs)
                        
                        processed_histories[cond_name] = avg_hist
            
            print("\n[1/2] Training Dynamics")
            print("-" * 70)
            visualize_training_comparisons(processed_histories)
            
            # Store final accuracies in summary
            for cond_name, hist in processed_histories.items():
                train_acc = hist.get("train_acc", [])
                val_acc = hist.get("val_acc", [])
                test_acc = hist.get("final_test_acc", None)
                
                if cond_name not in summary:
                    summary[cond_name] = {}
                if train_acc:
                    summary[cond_name]["final_train_acc"] = train_acc[-1]
                if val_acc:
                    summary[cond_name]["final_val_acc"] = val_acc[-1]
                if test_acc is not None:
                    summary[cond_name]["final_test_acc"] = test_acc
        else:
            print(f"Warning: Training history file not found: {history_file}")
    
    # ==========================================
    # 2. Load and Visualize RSA Matrices
    # ==========================================
    if show_rsa_matrices:
        for split in splits:
            # Load labels for this split
            labels_file = run_path / f"{split}_labels_by_condition.json"
            
            if not labels_file.exists():
                print(f"\nWarning: Labels file not found: {labels_file}")
                print(f"Skipping {split.upper()} RSA visualization.")
                continue
            
            with open(labels_file, 'r') as f:
                labels_dict = json.load(f)
            
            # Auto-detect conditions if not provided (only once)
            if conditions is None and split == splits[0]:
                pattern = f"*_{split}_matrices.npz"
                matrix_files = list(run_path.glob(pattern))
                conditions = [f.stem.replace(f"_{split}_matrices", "") for f in matrix_files]
                print(f"Auto-detected conditions: {conditions}")
            
            print(f"\n[RSA Matrices: {split.upper()} set]")
            print("=" * 70)
            
            # Visualize each condition for this split
            for cond_name in conditions:
                matrix_file = run_path / f"{cond_name}_{split}_matrices.npz"
                
                if not matrix_file.exists():
                    print(f"Warning: Matrix file not found: {matrix_file}")
                    continue
                
                print(f"\nCondition: {cond_name}")
                print("-" * 70)
                
                # Load matrix from npz
                data = np.load(matrix_file)
                
                # Get matrix for specified run_idx
                key = f"arr_{run_idx}"
                if key not in data.files:
                    print(f"Warning: Run {run_idx} not found in {matrix_file.name}. Available: {data.files}")
                    if len(data.files) > 0:
                        key = data.files[0]
                        print(f"Using {key} instead.")
                    else:
                        continue
                
                matrix = data[key]
                
                # Get labels for this condition
                labels = labels_dict.get(cond_name, [])
                
                if len(labels) == 0:
                    print(f"Warning: No labels found for condition '{cond_name}'")
                    continue
                
                # Verify shape consistency
                if matrix.shape[0] != len(labels):
                    print(f"Warning: Shape mismatch for {cond_name}: matrix={matrix.shape}, labels={len(labels)}")
                    labels = labels[:matrix.shape[0]]
                
                # Visualize using existing function
                visualize_correlation_heatmap(
                    correlation_matrix=matrix,
                    labels=labels,
                    title=f"RSA Matrix: {cond_name} ({split.upper()})",
                    show_class_averaged=True,
                    full_label=full_label
                )
                
                # Compute discriminability
                person_labels = _extract_person_names_from_labels(labels)
                within, between, diff = compute_discriminability(matrix, person_labels)
                
                # Store in summary
                if cond_name not in summary:
                    summary[cond_name] = {}
                summary[cond_name][f"{split}_discriminability"] = {
                    "within": within,
                    "between": between,
                    "diff": diff
                }
    
    # ==========================================
    # 3. Print Summary Table
    # ==========================================
    print("\n" + "=" * 80)
    print("SUMMARY - MODEL PERFORMANCE")
    print("=" * 80)
    
    if summary:
        # Table 1: Accuracy Summary
        print("\n[Table 1] Accuracy Summary")
        print("-" * 80)
        
        conditions_list = sorted(summary.keys())
        
        header = f"{'Condition':<20} {'Train Acc':>12} {'Val Acc':>12} {'Test Acc':>12}"
        print(header)
        print("-" * 80)
        
        for cond_name in conditions_list:
            stats = summary[cond_name]
            train_acc = stats.get("final_train_acc", float('nan'))
            val_acc = stats.get("final_val_acc", float('nan'))
            test_acc = stats.get("final_test_acc", float('nan'))
            
            train_str = f"{train_acc:>11.2f}%" if not np.isnan(train_acc) else f"{'N/A':>12}"
            val_str   = f"{val_acc:>11.2f}%"   if not np.isnan(val_acc)   else f"{'N/A':>12}"
            test_str  = f"{test_acc:>11.2f}%"  if not np.isnan(test_acc)  else f"{'N/A':>12}"
            
            print(f"{cond_name:<20} {train_str} {val_str} {test_str}")
        
        print("-" * 80)
        
        # Table 2: Discriminability Summary
        has_discriminability = any(
            any(k.endswith('_discriminability') for k in stats.keys()) 
            for stats in summary.values()
        )
        
        if has_discriminability:
            print("\n[Table 2] RSA Discriminability Summary")
            print("-" * 80)
            
            # Collect all splits
            all_splits = set()
            for stats in summary.values():
                for key in stats.keys():
                    if key.endswith('_discriminability'):
                        split = key.replace('_discriminability', '')
                        all_splits.add(split)
            all_splits = sorted(all_splits)
            
            # Print for each split
            for split in all_splits:
                print(f"\n{split.upper()} Set:")
                header = f"{'Condition':<20} {'Within':>10} {'Between':>10} {'Diff':>10}"
                print(header)
                print("-" * 52)
                
                for cond_name in conditions_list:
                    stats = summary[cond_name]
                    disc_key = f"{split}_discriminability"
                    
                    if disc_key in stats and isinstance(stats[disc_key], dict):
                        disc = stats[disc_key]
                        within = disc.get('within', float('nan'))
                        between = disc.get('between', float('nan'))
                        diff = disc.get('diff', float('nan'))
                        
                        within_str = f"{within:>9.4f}" if not np.isnan(within) else f"{'N/A':>10}"
                        between_str = f"{between:>9.4f}" if not np.isnan(between) else f"{'N/A':>10}"
                        diff_str = f"{diff:>9.4f}" if not np.isnan(diff) else f"{'N/A':>10}"
                        
                        print(f"{cond_name:<20} {within_str} {between_str} {diff_str}")
                
                print("-" * 52)
        
        # Detailed View
        print("\n[Detailed View]")
        print("-" * 80)
        for cond_name, stats in summary.items():
            print(f"\n{cond_name}:")
            for key, val in stats.items():
                if isinstance(val, dict):
                    print(f"  {key}:")
                    for subkey, subval in val.items():
                        print(f"    {subkey}: {subval:.4f}")
                else:
                    print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
    
    print("\n" + "=" * 80)
    
    return summary


# ==========================================
# Legacy Functions (Backward Compatibility)
# ==========================================

def visualize_correlation_matrix(correlation_matrix, person_names):
    """
    Generate a heatmap visualization of the image correlation matrix with group labels.

    Inputs:
      - correlation_matrix: A 2D NumPy array of shape (N, N) containing correlation coefficients.
      - person_names: A list of strings representing the names of persons. It is assumed each person has a fixed number of images (e.g., 5 images per person).
    
    Output:
      - Displays a heatmap plot with annotated correlation values and labeled axes.
    """
    num_people = len(person_names)
    num_images_per_person = 5
    
    # Create custom axis labels
    labels = [''] * correlation_matrix.shape[0]
    for i, name in enumerate(person_names):
        index = i * num_images_per_person
        labels[index] = name

    _set_seaborn_style(font_scale=2.5)
    
    plt.figure(figsize=(120, 96))
    
    ax = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f",
                     xticklabels=labels, yticklabels=labels)
    ax.set_title('Correlation Matrix Heatmap', fontsize=50)
    ax.set_xlabel('Image Index', fontsize=40)
    ax.set_ylabel('Image Index', fontsize=40)
    plt.gca().invert_yaxis()
    plt.show()


def visualize_correlation_dendrogram(correlation_matrix):
    """
    Create and display a hierarchical clustering dendrogram based on the image correlation matrix.

    Inputs:
      - correlation_matrix: A 2D NumPy array of shape (N, N) representing the correlation coefficients between images.
    
    Output:
      - Displays a dendrogram plot representing the hierarchical clustering of images.
    """
    from scipy.spatial.distance import squareform

    distance_matrix = 1.0 - np.asarray(correlation_matrix, dtype=float)
    condensed = squareform(distance_matrix, checks=False)
    linkage_matrix = hc.linkage(condensed, method='average')
    
    plt.figure(figsize=(30, 28))
    hc.dendrogram(linkage_matrix)
    plt.title('Hierarchical Clustering Dendrogram of Face Images')
    plt.xlabel('Index of Image')
    plt.ylabel('Distance')
    plt.show()


def visualize_correlation_matrixByPerson(correlation_matrix, person_names):
    """
    Compute and visualize the average correlation coefficient for each person by averaging the correlations among a subset of images (excluding the first test image) per person.

    Inputs:
      - correlation_matrix: A 2D NumPy array of shape (N, N) containing the correlation coefficients for all images.
      - person_names: A list of strings representing the names of persons. It is assumed each person has a fixed number of images (e.g., 5 images per person), where the first image is used for testing and excluded from the averaging.
    
    Outputs:
      - Prints the mean within-person and out-of-person correlation values.
      - Displays a heatmap plot of the computed person-to-person average correlation matrix.
    """
    num_people = len(person_names)
    num_images_per_person = 5
    person_correlation_matrix = np.zeros((num_people, num_people), dtype=np.float32)

    for i in range(num_people):
        for j in range(num_people):
            start_i = i * num_images_per_person + 1
            end_i = start_i + num_images_per_person - 2

            start_j = j * num_images_per_person + 1
            end_j = start_j + num_images_per_person - 2

            block = correlation_matrix[start_i:end_i+1, start_j:end_j+1]
            person_correlation_matrix[i, j] = np.mean(block)

    # Calculate and print averages
    meanTempWithIn = 0.0
    meanTempOut = 0.0
    for i in range(num_people):
        meanTempWithIn += person_correlation_matrix[i, i]
        for j in range(i+1, num_people):
            meanTempOut += person_correlation_matrix[i, j]

    meanCorWithIn = meanTempWithIn / num_people
    meanCorOut = meanTempOut / (num_people * 5)

    print("Mean within-person correlation =", meanCorWithIn)
    print("Mean out-of-person correlation =", meanCorOut)

    plt.figure(figsize=(60, 48))
    sns.heatmap(person_correlation_matrix, annot=True, cmap='coolwarm',
                xticklabels=person_names,
                yticklabels=person_names)
    plt.title('Average Correlation Coefficient Per Person (Excluding Test Image)')
    plt.xlabel('Person')
    plt.ylabel('Person')
    plt.gca().invert_yaxis()
    plt.show()


def visualize_c(correlation_matrix, person_names):
    """
    Execute a comprehensive visualization of the correlation matrix by generating a heatmap, a dendrogram, and an aggregated person-to-person block average heatmap.

    Inputs:
      - correlation_matrix: A 2D NumPy array containing image correlation coefficients.
      - person_names: A list of strings representing the names of persons, with the assumption of a fixed number of images per person.
    
    Output:
      - Displays three visualizations: a detailed heatmap, a hierarchical clustering dendrogram, and an aggregated person correlation heatmap.
    """
    visualize_correlation_matrix(correlation_matrix, person_names)
    visualize_correlation_dendrogram(correlation_matrix)
    visualize_correlation_matrixByPerson(correlation_matrix, person_names)


def plot_metrics_acc(metrics_dict, epochs_num, title, ylabel):
    """
    Plot training metrics with error bars for each slope category over a specified number of epochs.

    Inputs:
      - metrics_dict: A dictionary where keys represent slope values (or identifiers) and values are lists or arrays of metric measurements collected over epochs.
      - epochs_num: An integer indicating the total number of training epochs.
      - title: A string representing the title of the plot.
      - ylabel: A string representing the label for the y-axis.
    
    Output:
      - Displays a line plot showing the mean metric values with error bars representing the standard deviation across epochs, differentiated by slope.
    """
    plt.figure(figsize=(60, 30))
    step = int(epochs_num / 10)
    error_bar_positions = np.arange(step, epochs_num+1, step)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    color_index = 0
    
    for slope, m_list in metrics_dict.items():
        arr = np.array(m_list).T
        m_means = arr.mean(axis=1)
        m_stds = arr.std(axis=1)
        current_color = colors[color_index % len(colors)]
        
        plt.plot(range(len(arr)), m_means, label=f'Slope {slope}', color=current_color)
        for i in error_bar_positions:
            if i == 0:
                continue
            plt.errorbar(i, m_means[i-1], yerr=m_stds[i-1], fmt='.', capsize=5, color=current_color)
        color_index += 1

    plt.title(title)
    plt.xlabel('Training Time (Epochs)')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def visualized_cor_mat(
    all_cor_data,
    positive_slope,
    negative_slope,
    noise_level,
    label,
    threshold=None
):
    """
    NEW VERSION: Selects data for a specific parameter combination from the
    aggregated correlation dictionary, computes the mean matrix, and visualizes it.
    
    Parameters:
        all_cor_data: dict, correlation matrices keyed by (pos_slope, neg_slope, noise_level) 
                      or (pos_slope, neg_slope, noise_level, threshold)
        positive_slope: float, positive slope value
        negative_slope: float, negative slope value
        noise_level: float, noise level value
        label: list, labels for visualization
        threshold: float or None, threshold value (optional, for newer experiments)
    """
    # Construct the key to look up in the dictionary
    if threshold is not None:
        key = (positive_slope, negative_slope, noise_level, threshold)
    else:
        key = (positive_slope, negative_slope, noise_level)
    
    print(f"--- Visualizing mean correlation matrix for key: {key} ---")
    
    matrices = all_cor_data.get(key)
    
    if not matrices:
        print(f"ERROR: No correlation data found for key {key}.")
        return

    # Calculate the mean correlation matrix across all iterations
    mean_matrix = np.mean(np.array(matrices), axis=0)
    
    print(f"Aggregated {len(matrices)} runs. Mean matrix shape: {mean_matrix.shape}")
    
    try:
        visualize_c(mean_matrix, label)
    except Exception as e:
        print(f"Warning: failed to visualize correlation matrix. {e}")
