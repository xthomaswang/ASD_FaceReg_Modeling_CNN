# src/analysis/__init__.py

"""
Analysis module: RSA analysis and visualization
"""

# RSA analysis (features + correlation + workflows)
from .rsa import (
    extract_features_pytorch,
    extract_features,
    compute_correlation_pytorch,
    compute_rsa_matrix,
    get_person_name_labels,
    aggregate_and_visualize_rsa,
    run_multi_run_rsa,
    visualize_rsa_results,
)

# Visualization
from .visualization import (
    visualize_correlation_heatmap,
    compute_discriminability,
    plot_discriminability_comparison,
    visualize_training_comparisons,
    visualize_local_results,
    visualize_correlation_matrix,
    visualize_correlation_dendrogram,
    visualize_correlation_matrixByPerson,
    visualize_c,
    plot_metrics_acc,
    visualized_cor_mat,
    plot_grouped_bar_comparison,
    plot_grouped_training_curves
)

__all__ = [
    # RSA - Feature extraction
    'extract_features',
    'extract_features_pytorch',
    # RSA - Correlation
    'compute_correlation_pytorch',
    # RSA - Workflows
    'compute_rsa_matrix',
    'get_person_name_labels',
    'aggregate_and_visualize_rsa',
    'run_multi_run_rsa',
    'visualize_rsa_results',
    # Visualization
    'visualize_correlation_heatmap',
    'compute_discriminability',
    'plot_discriminability_comparison',
    'visualize_training_comparisons',
    'visualize_local_results',
    'visualize_correlation_matrix',
    'visualize_correlation_dendrogram',
    'visualize_correlation_matrixByPerson',
    'visualize_c',
    'plot_metrics_acc',
    'visualized_cor_mat',
    'plot_grouped_bar_comparison',
    'plot_grouped_training_curves',
]
