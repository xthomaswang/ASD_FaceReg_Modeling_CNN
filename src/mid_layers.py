# src/mid_layers.py

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from preprocessing import preprocess_image  # Use your existing preprocessing function

def build_activation_model(model, layer_names):
    """
    Create an activation model that outputs activations for the specified layers.
    
    Parameters:
      - model: A trained Keras model.
      - layer_names: List of layer names whose outputs will be extracted.
      
    Returns:
      - A new Keras Model with the same inputs as `model` but with outputs as the activations of the specified layers.
    """
    outputs = [model.get_layer(name).output for name in layer_names]
    return Model(inputs=model.input, outputs=outputs)

def compute_inactive_percentage(activation, threshold=1e-3):
    """
    Compute the percentage of inactive neurons in an activation array.
    
    Parameters:
      - activation: Numpy array output from a model layer.
      - threshold: A threshold below which a neuron's absolute value is considered inactive.
      
    Returns:
      - Percentage of neurons with absolute values less than the threshold.
    """
    total_neurons = np.prod(activation.shape)
    inactive_neurons = np.sum(np.abs(activation) < threshold)
    return (inactive_neurons / total_neurons) * 100

def generate_avg_results_by_model(model_dict, layer_names, img_dir, target_size, pad_value=1):
    """
    Generate a dictionary mapping model keys to the average inactive percentages computed over a directory of images.
    
    Parameters:
      - model_dict: Dictionary with keys (e.g., (slope, noise)) and values as lists of trained models.
      - layer_names: List of layer names to analyze (e.g., ['conv_block1_relu', 'conv_block2_relu', 'conv_block3_relu']).
      - img_dir: Directory containing images for analysis.
      - target_size: Target size (int) for resizing images.
      - pad_value: Padding value for preprocessing.
    
    Returns:
      - avg_results_by_model: Dictionary mapping each model key to a dictionary of layer inactive percentages.
    """
    avg_results_by_model = {}
    for key, models in model_dict.items():
        # Use the first model from the list for this key.
        model = models[0]
        analyzer = MidLayerAnalyzer(model, layer_names, threshold=1e-3)
        avg_results = analyzer.analyze_directory(img_dir, target_size, pad_value)
        avg_results_by_model[key] = avg_results
        print(f"Generated results for model key {key}: {avg_results}")
    return avg_results_by_model

class MidLayerAnalyzer:
    """
    A class to analyze and visualize mid-layer activations and inactive neuron percentages.
    """
    def __init__(self, model, layer_names, threshold=1e-3):
        self.model = model
        self.layer_names = layer_names
        self.threshold = threshold
        self.activation_model = build_activation_model(model, layer_names)
    
    def analyze_single_image(self, image_array):
        """
        Analyze a single preprocessed image to compute the inactive neuron percentage for each specified layer.
        
        Parameters:
          - image_array: Preprocessed image with shape (1, H, W, C).
          
        Returns:
          - A dictionary mapping each layer name to its inactive neuron percentage.
        """
        activations = self.activation_model.predict(image_array)
        results = {}
        for i, layer in enumerate(self.layer_names):
            results[layer] = compute_inactive_percentage(activations[i], self.threshold)
        return results

    def analyze_directory(self, img_dir, target_size, pad_value=1):
        """
        Analyze all images in a directory (recursively) and compute the average inactive neuron percentage for each specified layer.
        
        Parameters:
          - img_dir: Directory containing images.
          - target_size: Target size (int) for resizing each image.
          - pad_value: Padding value added around each image.
          
        Returns:
          - A dictionary mapping each layer name to the average inactive neuron percentage across all images.
        """
        results = {layer: [] for layer in self.layer_names}
        
        for root, _, files in os.walk(img_dir):
            for file in files:
                if file.lower().endswith(('png', 'jpg', 'jpeg')):
                    img_path = os.path.join(root, file)
                    # Use existing preprocess_image function
                    img_arr = preprocess_image(img_path, target_size, pad_value)
                    img_arr = np.expand_dims(img_arr, axis=0)
                    activations = self.activation_model.predict(img_arr)
                    for i, layer in enumerate(self.layer_names):
                        perc = compute_inactive_percentage(activations[i], self.threshold)
                        results[layer].append(perc)
        
        avg_results = {layer: np.mean(results[layer]) if results[layer] else 0.0 for layer in self.layer_names}
        return avg_results

    def save_results(self, results, epoch, base_dir="../res/EIB"):
        """
        Save the analysis results to a CSV file under a designated folder.
        
        Parameters:
          - results: Dictionary with inactive neuron percentages.
          - epoch: Current epoch number (used to define the save folder).
          - base_dir: Base directory to save results.
        """
        save_dir = os.path.join(base_dir, str(epoch), "mid_layer")
        os.makedirs(save_dir, exist_ok=True)
        df = pd.DataFrame(list(results.items()), columns=["Layer", "Inactive_Percentage"])
        save_path = os.path.join(save_dir, f"mid_layer_results_epoch_{epoch}.csv")
        df.to_csv(save_path, index=False)
        print(f"Saved mid-layer analysis results to: {save_path}")

    def visualize_results(self, results):
        """
        Visualize the average inactive neuron percentages as a bar chart.
        
        Parameters:
          - results: Dictionary mapping layer names to average inactive percentages.
        """
        layers = list(results.keys())
        percentages = list(results.values())
        plt.figure(figsize=(10, 6))
        plt.bar(layers, percentages, color='skyblue')
        plt.title("Average Inactive Neuron Percentage by Layer")
        plt.xlabel("Layer")
        plt.ylabel("Inactive Percentage")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def visualize_activations(self, image_array, cmap='viridis'):
        """
        Visualize activation maps for each specified layer using a preprocessed single image.
        Displays each layer's feature maps in a grid.
        
        Parameters:
          - image_array: Preprocessed image with shape (1, H, W, C).
          - cmap: Colormap for visualization (default 'viridis' for color).
        """
        activations = self.activation_model.predict(image_array)
        for i, layer in enumerate(self.layer_names):
            activation = np.squeeze(activations[i], axis=0)  # Remove batch dimension
            num_channels = activation.shape[-1]
            cols = 8
            rows = math.ceil(num_channels / cols)
            plt.figure(figsize=(16, 2 * rows))
            plt.suptitle(f"Layer: {layer} - {num_channels} feature maps", fontsize=14)
            for c in range(num_channels):
                plt.subplot(rows, cols, c + 1)
                plt.imshow(activation[..., c], cmap=cmap)
                plt.axis("off")
            plt.tight_layout()
            plt.show()

    def visualize_activations_from_path(self, image_path, target_size, pad_value=1, cmap='viridis'):
        """
        Load and preprocess an image using the existing preprocessing function, then visualize its activations.
        
        Parameters:
          - image_path: Path to the image file.
          - target_size: Target size (int) for resizing the image.
          - pad_value: Padding value for image preprocessing.
          - cmap: Colormap for visualization.
        """
        img_arr = preprocess_image(image_path, target_size, pad_value)
        img_arr = np.expand_dims(img_arr, axis=0)
        self.visualize_activations(img_arr, cmap=cmap)

def visualize_activations_across_models(model_dict, layer_names, image_path, target_size, pad_value=1, cmap='viridis'):
    """
    For each key in model_dict, take the first model and extract activations for the specified layer.
    Then, for each layer, arrange the activation maps from all models in a composite figure:
      - Rows: one for each model (e.g., different slope values)
      - Columns: each channel of the activation map
    This function produces one figure per layer.
    
    Parameters:
      - model_dict: Dictionary with keys (e.g., slope values) and values as lists of trained models.
      - layer_names: List of layer names to visualize (e.g., for conv and relu layers).
      - image_path: File path to the image to be used for activation extraction.
      - target_size: Target size (int) for resizing the image.
      - pad_value: Padding value for preprocessing.
      - cmap: Colormap for visualization (default 'viridis' for color).
    """
    # Preprocess the image using the existing function
    img_arr = preprocess_image(image_path, target_size, pad_value)
    img_arr = np.expand_dims(img_arr, axis=0)
    
    # For each layer in layer_names, create a composite figure
    for layer in layer_names:
        # List to store activation maps for each model for this layer
        activations_list = []
        model_keys = []
        for key in sorted(model_dict.keys()):
            # Get the first model for this key
            model = model_dict[key][0]
            # Build a temporary activation model for the specific layer
            temp_act_model = build_activation_model(model, [layer])
            act = temp_act_model.predict(img_arr)  # shape: (1, H, W, channels)
            act = np.squeeze(act, axis=0)  # shape: (H, W, channels)
            activations_list.append(act)
            model_keys.append(str(key))
        
        # Assume all activations have the same shape and channel number
        num_models = len(activations_list)
        num_channels = activations_list[0].shape[-1]
        
        # Create a figure with rows = number of models and columns = number of channels
        fig, axes = plt.subplots(num_models, num_channels, figsize=(num_channels*1.5, num_models*1.5))
        fig.suptitle(f"Layer: {layer}", fontsize=16)
        
        # If only one row, axes may not be a 2D array
        if num_models == 1:
            axes = np.expand_dims(axes, axis=0)
            
        for i, act in enumerate(activations_list):
            for j in range(num_channels):
                ax = axes[i, j] if num_models > 1 else axes[j]
                ax.imshow(act[..., j], cmap=cmap)
                ax.axis("off")
                if i == 0:
                    ax.set_title(f"Ch {j+1}", fontsize=8)
            # Label each row with the corresponding key
            axes[i, 0].set_ylabel(model_keys[i], fontsize=10, rotation=0, labelpad=40)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

def visualize_grouped_inactive_percentages(avg_results_dict, relu_layers, group_by="both"):
    """
    Visualize average inactive percentages for ReLU layers across models in a grouped bar chart.
    
    Parameters:
      - avg_results_dict: Dictionary mapping model keys (e.g., (slope, noise)) to dictionaries of layer inactive percentages.
      - relu_layers: List of ReLU layer names to include 
        (e.g., ['conv_block1_relu', 'conv_block2_relu', 'conv_block3_relu']).
      - group_by: Specifies which part of the key to use for grouping. Options:
            "slope" - use only the slope (key[0])
            "noise" - use only the noise (key[1])
            "both"  - use both values formatted as "slope: X, noise: Y"
    
    This function produces a grouped bar chart where each group corresponds to a key as determined by group_by.
    """
    # Build the list of labels for the x-axis from the keys
    x_labels = []
    sorted_keys = sorted(avg_results_dict.keys())
    for key in sorted_keys:
        if group_by == "slope":
            label = str(key[0])
        elif group_by == "noise":
            label = str(key[1])
        elif group_by == "both":
            label = f"slope: {key[0]}, noise: {key[1]}"
        else:
            label = str(key)
        x_labels.append(label)
    
    # Prepare data: each row corresponds to a key (as determined by group_by)
    rows = []
    for key in sorted_keys:
        if group_by == "slope":
            label = str(key[0])
        elif group_by == "noise":
            label = str(key[1])
        elif group_by == "both":
            label = f"slope: {key[0]}, noise: {key[1]}"
        else:
            label = str(key)
        row = {"Group": label}
        for layer in relu_layers:
            row[layer] = avg_results_dict[key].get(layer, 0.0)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.set_index("Group")
    
    # Plot grouped bar chart
    ax = df.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Average Inactive Percentage for ReLU Layers")
    ax.set_xlabel("Group")
    ax.set_ylabel("Inactive Percentage")
    plt.xticks(rotation=0)
    plt.legend(title="ReLU Layer")
    plt.tight_layout()
    plt.show()

