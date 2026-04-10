# Old Pipeline

This directory preserves the legacy TensorFlow/Keras implementation used for the original 50-image experiments.

## Contents

- `src/`: simple custom CNN, preprocessing, intermediate-layer extraction, and utility code
- `data/`: cropped face images, one-hot labels, and provenance CSV files
- `notebooks/`: original notebooks used to run the early experiments
- `results/`: saved JSON outputs from the early E/I-balance and internal-noise runs
- `analysis/`: R and Python scripts for downstream result analysis
- `docs/`: legacy project image assets

## Data included here

- 10 identities
- 5 cropped face images per identity
- 50 cropped images total
- `labels.csv` for class labels
- `cropped_training_data_manifest.csv` for included image files
- `image_sources.csv` for provenance of the manually collected source images

The uncropped raw source images are not included in the repository.

## Run the legacy code

```bash
pip install -r requirements.txt
```

Main implementation files:

- `src/models.py`
- `src/preprocessing.py`
- `src/analysis.py`
- `src/mid_layers.py`

This branch of the project is kept mostly as an archival baseline so the original TensorFlow workflow remains reproducible and easy to compare against the later `large-scale-cornet/` pipeline.
