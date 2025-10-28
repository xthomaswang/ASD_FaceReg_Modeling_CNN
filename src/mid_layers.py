# src/mid_layers.py (Corrected and Optimized)

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from preprocessing import preprocess_image # Assuming this is in your src directory

def build_activation_model(model, layer_names):
    """
    Creates a Keras Model that extracts activations from a list of specified layers.
    """
    # FIX: Corrected typo from `model.layer` to `model.layers`
    try:
        outputs = [model.get_layer(name).output for name in layer_names]
        return Model(inputs=model.layers[0].input, outputs=outputs)
    except Exception as e:
        print(f"Error building activation model. Please check if model is built and layer names are correct: {layer_names}")
        raise e

def compute_inactive_percentage(activation, threshold=1e-3):
    """
    Computes the percentage of near-zero ('inactive') neurons in an activation map.
    """
    if activation is None or activation.size == 0:
        return 0.0
    total_neurons = np.prod(activation.shape)
    inactive_neurons = np.sum(np.abs(activation) < threshold)
    return (inactive_neurons / total_neurons) * 100

class MidLayerAnalyzer:
    """
    A class to analyze and visualize mid-layer activations of a Keras model.
    """
    def __init__(self, model, layer_names, threshold=1e-3):
        self.model = model
        self.layer_names = layer_names
        self.threshold = threshold
        # This will now work correctly with the fixed build_activation_model function.
        self.activation_model = build_activation_model(model, layer_names)
    
    def analyze_directory(self, img_dir, target_size, pad_value=1):
        """
        OPTIMIZED: Analyzes all images in a directory using efficient batch prediction.
        """
        all_image_paths = []
        for root, _, files in os.walk(img_dir):
            for file in files:
                if file.lower().endswith(('png', 'jpg', 'jpeg')):
                    all_image_paths.append(os.path.join(root, file))
        
        if not all_image_paths:
            print(f"Warning: No images found in directory {img_dir}")
            return {layer: 0.0 for layer in self.layer_names}

        # Preprocess all images and create a single batch
        image_batch = np.array([preprocess_image(p, target_size, pad_value) for p in all_image_paths])
        
        # Get all activations with a single predict call
        all_activations = self.activation_model.predict(image_batch)
        
        # If there's only one layer, activations won't be in a list, so we wrap it
        if len(self.layer_names) == 1:
            all_activations = [all_activations]
            
        # Calculate inactive percentages
        results = {layer: [] for layer in self.layer_names}
        for i, layer_name in enumerate(self.layer_names):
            layer_activations = all_activations[i] # This is a batch of activations for one layer
            for single_image_activation in layer_activations:
                perc = compute_inactive_percentage(single_image_activation, self.threshold)
                results[layer_name].append(perc)
        
        # Calculate the average across all images for each layer
        avg_results = {layer: np.mean(res) if res else 0.0 for layer, res in results.items()}
        return avg_results

    def visualize_activations_from_path(self, image_path, target_size, pad_value=1, cmap='viridis'):
        """
        Loads, preprocesses, and visualizes activations for a single image.
        """
        img_arr = preprocess_image(image_path, target_size, pad_value)
        img_arr_batch = np.expand_dims(img_arr, axis=0) # Add batch dimension
        
        activations = self.activation_model.predict(img_arr_batch)
        if len(self.layer_names) == 1:
            activations = [activations]
            
        for i, layer_name in enumerate(self.layer_names):
            activation = activations[i][0] # Get the first (and only) activation from the batch
            
            if activation.ndim != 3:
                print(f"Layer '{layer_name}' is not a 2D convolutional layer. Skipping visualization.")
                continue

            num_channels = activation.shape[-1]
            cols = min(num_channels, 8)
            rows = math.ceil(num_channels / cols)
            
            plt.figure(figsize=(cols * 2, rows * 2))
            plt.suptitle(f"Layer: {layer_name} - {num_channels} feature maps", fontsize=16)
            for c in range(num_channels):
                plt.subplot(rows, cols, c + 1)
                plt.imshow(activation[..., c], cmap=cmap)
                plt.axis("off")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

def generate_avg_results_by_model(model_dict, layer_names, img_dir, target_size, pad_value=1):
    """
    Generates a dictionary mapping model keys to their average inactive neuron percentages.
    """
    avg_results_by_model = {}
    for key, models in model_dict.items():
        if not models:
            continue
        model = models[0] # Use the first model from the list for this key
        analyzer = MidLayerAnalyzer(model, layer_names)
        avg_results = analyzer.analyze_directory(img_dir, target_size, pad_value)
        avg_results_by_model[key] = avg_results
        print(f"Generated results for model key {key}: {avg_results}")
    return avg_results_by_model

def visualize_activations_across_models(model_dict, layer_names, image_path, target_size, pad_value=1, cmap='viridis'):
    """
    For each specified layer, creates a composite figure showing that layer's activations
    from a single image across multiple models.
    """
    img_arr = preprocess_image(image_path, target_size, pad_value)
    img_arr_batch = np.expand_dims(img_arr, axis=0)
    
    for layer_name in layer_names:
        activations_list = []
        model_keys = []
        
        for key in sorted(model_dict.keys()):
            if not model_dict[key]: continue
            model = model_dict[key][0]
            
            try:
                temp_act_model = build_activation_model(model, [layer_name])
                act = temp_act_model.predict(img_arr_batch)
                activations_list.append(act[0]) # Squeeze out batch dim
                model_keys.append(str(key))
            except Exception as e:
                print(f"Could not get activation for layer '{layer_name}' in model '{key}'. Skipping. Error: {e}")
                continue
        
        if not activations_list or activations_list[0].ndim != 3:
            print(f"No valid activations found for layer '{layer_name}', or it is not a 2D layer. Skipping visualization.")
            continue

        num_models = len(activations_list)
        num_channels = activations_list[0].shape[-1]
        
        fig, axes = plt.subplots(num_models, num_channels, figsize=(num_channels * 1.5, num_models * 1.8), squeeze=False)
        fig.suptitle(f"Activations for Layer: {layer_name}", fontsize=16)
        
        for i, (act, key_label) in enumerate(zip(activations_list, model_keys)):
            axes[i, 0].set_ylabel(key_label, rotation=0, ha='right', va='center', fontsize=10, labelpad=40)
            for j in range(num_channels):
                ax = axes[i, j]
                ax.imshow(act[..., j], cmap=cmap)
                ax.axis("off")
                if i == 0:
                    ax.set_title(f"Channel {j+1}", fontsize=8)
                    
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

def visualize_grouped_inactive_percentages(avg_results_dict, relu_layers, group_by="slope"):
    """
    Visualizes average inactive percentages for specified layers in a grouped bar chart.
    """
    # SIMPLIFIED: Prepare data directly for DataFrame creation
    records = []
    for key, layer_results in avg_results_dict.items():
        if group_by == "slope":
            group_label = f"Slope: {key[0]}"
        elif group_by == "noise":
            group_label = f"Noise: {key[1]}"
        else:
            group_label = str(key)
        
        record = {"Group": group_label}
        for layer in relu_layers:
            record[layer] = layer_results.get(layer, 0.0)
        records.append(record)
    
    if not records:
        print("No results to plot.")
        return

    df = pd.DataFrame.from_records(records).set_index("Group")
    
    # Plot grouped bar chart
    ax = df.plot(kind="bar", figsize=(12, 7), width=0.8)
    plt.title("Average Inactive Neuron Percentage for ReLU Layers", fontsize=16)
    plt.xlabel("Model Group", fontsize=12)
    plt.ylabel("Inactive Percentage (%)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="ReLU Layer", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()