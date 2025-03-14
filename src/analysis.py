# src/analysis.py

"""
Module: analysis
Description: Contains functions for:
- Calculating pairwise correlations across images.
- Visualizing correlation matrices using heatmaps, dendrograms, and aggregated person-based blocks.
"""

import os
import numpy as np
import pandas as pd
import concurrent.futures
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as hc
import tensorflow as tf
import json
from tensorflow.keras.models import Model

############################
# 1) Pairwise Image Correlation
############################

def cor_calculator(img1, img2, model, layer_name):
    """
    Compute the Pearson correlation coefficient between two images based on feature representations extracted from a specified layer of a model.

    Inputs:
      - img1: A preprocessed image array.
      - img2: A preprocessed image array.
      - model: A compiled/trained tf.keras.Model.
      - layer_name: A string representing the name of the layer from which to extract features.
    
    Output:
      - A float representing the Pearson correlation coefficient between the flattened feature representations of img1 and img2.
    """
    # Create an intermediate model that outputs the specified layer's output
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    
    # Obtain and flatten the output features for each image
    output1 = intermediate_layer_model.predict(np.expand_dims(img1, axis=0)).flatten()
    output2 = intermediate_layer_model.predict(np.expand_dims(img2, axis=0)).flatten()
    
    # Compute and return the Pearson correlation coefficient between the two feature vectors
    ans = np.corrcoef(output1, output2)[0, 1]
    return ans

def compute_correlations(image_data, model, layer_name):
    """
    Compute a symmetric correlation matrix for a set of images using feature representations extracted from a specific layer of a model.

    Inputs:
      - image_data: A dictionary where keys are image file paths and values are preprocessed image arrays.
      - model: A compiled/trained tf.keras.Model.
      - layer_name: A string representing the name of the layer from which to extract image features.
    
    Output:
      - A NumPy array of shape (N, N) containing the Pearson correlation coefficients between every pair of images.
    """
    # Sort images to ensure consistent matrix indices
    all_images = list(sorted(image_data.keys()))
    num_images = len(all_images)

    # Generate list of image pairs for which the correlation will be computed (only upper triangular pairs)
    pairs = [(all_images[i], all_images[j]) 
             for i in range(num_images) for j in range(i, num_images)]

    def compute_correlation(pair):
        img_path1, img_path2 = pair
        return cor_calculator(image_data[img_path1],
                              image_data[img_path2],
                              model, layer_name)

    # Use a ThreadPoolExecutor to compute correlations in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(compute_correlation, pairs))

    # Fill in a symmetric correlation matrix using the computed results
    correlation_matrix = np.zeros((num_images, num_images), dtype=np.float32)
    k = 0
    for i in range(num_images):
        for j in range(i, num_images):
            correlation_matrix[i, j] = results[k]
            correlation_matrix[j, i] = results[k]
            k += 1

    return correlation_matrix

############################
# 2) Visualizations
############################

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
    num_images_per_person = 5  # Assumption: each person has 5 images
    
    # Create custom axis labels: label the first image of each person's group with their name
    labels = [''] * correlation_matrix.shape[0]
    for i, name in enumerate(person_names):
        index = i * num_images_per_person
        labels[index] = name

    # Configure seaborn font scale for better visibility
    sns.set(font_scale=2.5)
    
    # Set the figure size for the heatmap
    plt.figure(figsize=(120, 96))
    
    # Plot the heatmap with annotations and custom labels
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
    # Convert the correlation matrix to a distance matrix
    distance_matrix = 1 - correlation_matrix
    linkage_matrix = hc.linkage(distance_matrix, method='average')
    
    # Set the figure size for the dendrogram
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
            # Define indices for person i and person j, skipping the first (test) image
            start_i = i * num_images_per_person + 1
            end_i = start_i + num_images_per_person - 2

            start_j = j * num_images_per_person + 1
            end_j = start_j + num_images_per_person - 2

            block = correlation_matrix[start_i:end_i+1, start_j:end_j+1]
            person_correlation_matrix[i, j] = np.mean(block)

    # Calculate and print the average correlation for within-person and out-of-person comparisons
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

    # Plot the person-to-person correlation matrix as a heatmap
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

def visualized_cor_mat(switch, ep, iteration, label, slope=0.05, noise=0):
    """
    Load a JSON file containing correlation matrices, compute the mean matrix for each key, and visualize the matrix corresponding to a given slope value.

    Inputs:
      - switch: 1 represents reading EIB data, 0 represents reading IN data.
      - ep: An identifier (e.g., an integer) representing the epoch, used to construct the JSON file path.
      - iteration: An integer representing the iteration number; the JSON file name is derived using iteration-1.
      - label: A string label used for the visualization.
      - slope: A float (default 0.05) indicating the key in the JSON data for which the mean correlation matrix is to be visualized.
    
    Output:
      - Displays a heatmap visualization of the mean correlation matrix corresponding to the provided slope value.
    
    Raises:
      - FileNotFoundError: If the specified JSON file does not exist.
    """

    if switch:
      json_file_path = f"../res/EIB/{ep}/cor_output_{ep}_{iteration - 1}.json"
    else:
      json_file_path = f"../res/IN/{ep}/cor_output_{ep}_{iteration - 1}.json"


    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"Error: {json_file_path} does not exist.")
    
    # Load the JSON file containing correlation data
    with open(json_file_path, 'r', encoding='utf-8') as file:
        cor_dict = json.load(file)
    
    mean_data = {}
    for key, arrays in cor_dict.items():
        # Convert list of arrays into a NumPy array for averaging
        arrays_np = np.array(arrays)
        # Compute the mean correlation matrix for the given key
        mean_array = np.mean(arrays_np, axis=0)
        mean_data[float(key)] = mean_array  # Convert key to float for consistency

    # Visualize the correlation matrix corresponding to the specified slope value
    visualize_c(mean_data[slope], label)
