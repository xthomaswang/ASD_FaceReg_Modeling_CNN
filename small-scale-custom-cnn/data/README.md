# Old Dataset Notes

This directory contains the small cropped dataset used in the early version of the project.

## Included

- `training_data/`: 50 cropped images across 10 identities
- `labels.csv`: one-hot labels used by the TensorFlow/Keras training code
- `cropped_training_data_manifest.csv`: file-level manifest for the cropped images included in this repository
- `image_sources.csv`: provenance links for the original manually collected web images

## Not included

- uncropped source images
- any larger external face datasets

Only the cropped training sample is redistributed here.

`image_sources.csv` refers to the manually collected source images before cropping and format conversion, so those source filenames do not always match the final cropped `.png` filenames one-to-one.
