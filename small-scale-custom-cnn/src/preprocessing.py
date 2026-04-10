# src/preprocessing.py

"""
Module: preprocessing
Description: Provides functions for image loading, normalization, and augmentation.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import keras.utils as image

def load_image(img_path):
    """
    Load an image from disk in RGB format.

    Inputs:
      - img_path (str): The file path of the image to load.
    
    Output:
      - A PIL.Image object in RGB mode.
    """
    return Image.open(img_path).convert('RGB')

def preprocess_image(image_path, target_size, pad_value=1):
    """
    Load and preprocess an image.

    Description:
      1) Loads the image resized to (target_size x target_size).
      2) Converts the image to a NumPy array and normalizes its pixel values to the [0,1] range.
      3) Pads the image by 1 pixel on each side with a constant value of pad_value.
    
    Inputs:
      - image_path (str): The file path of the image to preprocess.
      - target_size (int): The target width and height to which the image is resized.
      - pad_value (int or float, optional): The constant value used for padding (default is 1).
    
    Output:
      - A NumPy array representing the padded image.
    """
    # Load the image with the specified target size
    img = image.load_img(image_path, target_size=(target_size, target_size))
    # Convert the image to a NumPy array
    img_array = image.img_to_array(img)
    # Normalize pixel values to the range [0, 1]
    img_array /= 255.0
    # Pad the image by 1 pixel on all sides with the specified pad_value
    padded_img = np.pad(img_array, ((1, 1), (1, 1), (0, 0)),
                        mode='constant', constant_values=pad_value)
    return padded_img

def load_images_and_process(folder_path, target_size):
    """
    Load and preprocess all images from a given folder structure.

    Description:
      Iterates through all subdirectories in folder_path, loads each image, preprocesses it using the 
      preprocess_image function, and stores the result in a dictionary.
    
    Inputs:
      - folder_path (str): The path to the main folder containing subfolders for each class or person.
      - target_size (int): The target size for resizing images before processing.
    
    Output:
      - image_data (dict): A dictionary mapping each image file path to its preprocessed image array.
    """
    image_data = {}
    for person_name in sorted(os.listdir(folder_path)):
        person_path = os.path.join(folder_path, person_name)
        for image_name in sorted(os.listdir(person_path)):
            image_path = os.path.join(person_path, image_name)
            processed_image = preprocess_image(image_path, target_size)
            image_data[image_path] = processed_image
    return image_data

def load_dataset(folder_path, target_size=(128, 128)):
    """
    Load a structured dataset of images and corresponding labels.

    Description:
      Reads images from a folder structure where each subfolder corresponds to a class (person).
      It processes each image, creates a DataFrame mapping image filenames to one-hot encoded labels,
      and constructs a dictionary of image paths to preprocessed image arrays along with a NumPy array of labels.
    
    Inputs:
      - folder_path (str): The root directory containing subfolders for each class.
      - target_size (tuple, optional): A tuple indicating the desired image dimensions (width, height).
                                        Default is (128, 128).
    
    Outputs:
      - df (pd.DataFrame): A DataFrame with columns 'img_id' and one column per class, indicating one-hot labels.
      - images_data (dict): A dictionary mapping image file paths to their corresponding preprocessed image arrays.
      - labels_arr (np.ndarray): A NumPy array of one-hot encoded labels with shape (N, num_classes).
    """
    # Load and preprocess all images
    full_image_data = load_images_and_process(folder_path, target_size[0])

    # Retrieve class names from subdirectory names, excluding hidden folders
    person_names = sorted([
        d for d in os.listdir(folder_path)
        if not d.startswith('.') and os.path.isdir(os.path.join(folder_path, d))
    ])

    # Prepare DataFrame columns and lists to store rows and labels
    columns = ['img_id'] + person_names
    rows = []
    images_data = {}  # Dictionary to store images that match the labels
    labels_list = []

    for person_name in person_names:
        person_index = person_names.index(person_name)
        person_folder = os.path.join(folder_path, person_name)
        
        for img_file in sorted(os.listdir(person_folder)):
            if img_file.startswith('.'):
                continue

            img_path = os.path.join(person_folder, img_file)

            # Only add image if it was successfully loaded and processed
            if img_path not in full_image_data:
                continue

            images_data[img_path] = full_image_data[img_path]

            # Create a row for the DataFrame: first element is the image filename,
            # followed by one-hot encoding for each class
            row = [img_file] + [1 if name == person_name else 0 for name in person_names]
            rows.append(row)

            # Create a one-hot encoded label vector for the current image
            label_vec = np.zeros(len(person_names), dtype=np.float32)
            label_vec[person_index] = 1.0
            labels_list.append(label_vec)

    # Create a DataFrame and a NumPy array of labels from the collected data
    df = pd.DataFrame(rows, columns=columns)
    labels_arr = np.array(labels_list, dtype=np.float32)

    return df, images_data, labels_arr
