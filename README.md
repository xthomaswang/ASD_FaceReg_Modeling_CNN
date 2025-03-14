# ASD_FaceReg_Modeling_CNN Research Project 
Xijing Wang, Emily Rios, Dr. Lang Chen in Santa Clara University in LCCN Lab

This project demonstrates a convolutional neural network (CNN) approach with two special variants:
- **EIB (E/I Imbalance) Model**: A custom activation function simulating excitatory and inhibitory imbalances.
- **IN (Internal Noise) Model**: A Gaussian noise layer inserted into the network.

## Project Structure


### Modules

1. **preprocessing.py**  
   - `load_image(image_path)`: Loads an image from the specified path.  
   - `resize_image(image, width=None, height=None)`: Resizes an image while maintaining aspect ratio.  
   - `normalize_image(image)`: Normalizes pixel values to the range [0, 1].  
   - `augment_image(image)`: Performs random image augmentations (optional).

2. **models.py**  
   - `build_base_cnn(input_shape, num_classes)`: Builds a simple CNN baseline.  
   - `build_EIB_cnn(input_shape, num_classes, e_coeff=1.0, i_coeff=1.0)`: Builds a CNN with a custom E/I imbalance activation function.  
   - `build_IN_cnn(input_shape, num_classes, noise_std=0.1)`: Builds a CNN model containing a Gaussian noise layer.  
   - `train_model(model, train_data, train_labels, ...)`: Trains the given model.  
   - `evaluate_model(model, test_data, test_labels)`: Evaluates the model’s accuracy on test data.

3. **analysis.py**  
   - `compute_correlation_matrix(features)`: Computes a correlation matrix for extracted features.  
   - `extract_intermediate_features(model, layer_name, data)`: Extracts features from a specified intermediate layer.  
   - `compute_pearson_correlation(vec1, vec2)`: Computes the Pearson correlation coefficient for two vectors.

4. **utils.py**  
   - `is_google_colab()`: Detects if the environment is Google Colab.  
   - `install_missing_packages()`: Installs required dependencies automatically if in Colab.

### How to Use

1. **Install Dependencies**  
   - Create a virtual environment and install packages:
     ```bash
     pip install -r requirements.txt
     ```
2. **Run in Local Environment**  
   - Use Python or Jupyter Notebook to run scripts or notebooks in the `notebooks/` directory.
3. **Run in Google Colab**  
   - Upload the repository or clone it.
   - Optionally call `install_missing_packages()` from `utils.py` if some packages are missing.
4. **Training & Evaluation**  
   - Import the relevant model-building function, e.g., `build_EIB_cnn` or `build_IN_cnn`.
   - Load and preprocess images.
   - Call `train_model(...)` to train and `evaluate_model(...)` to evaluate.
5. **Feature Analysis**  
   - Use `extract_intermediate_features(...)` to get intermediate layer outputs.
   - Compute correlation with `compute_correlation_matrix(...)`.

Feel free to modify or extend these modules to fit your research needs.
