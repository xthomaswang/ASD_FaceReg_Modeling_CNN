# Supplementary CORnet and Large-Scale Experiments

This directory contains the supplementary materials associated with the later large-scale CORnet-based experiments in the ASD face-recognition study.

Within the repository as a whole:

- `small-scale-custom-cnn/` preserves the original 50-image customized-model pipeline
- `large-scale-cornet/` (this directory) preserves the later supplementary CORnet experiments, dataset manifests, lightweight results, and associated statistical materials

Accordingly, `large-scale-cornet/` should be read as a supplementary experiment package rather than the original project codebase.

## Model architecture

The core model is **CORnet-Z** (a biologically inspired feedforward CNN mapping onto ventral visual stream areas V1 → V2 → V4 → IT), used in two modes:

### 1. Frozen pathology injection (`CornetWithPathology`)

- Loads a pretrained (ImageNet) CORnet-Z/S/RT and **freezes all weights**
- Injects E/I imbalance and internal noise via **forward hooks** on each area's nonlinearity:
  - `slope` > 1.0 simulates excitation-dominated (E > I) processing
  - `slope` < 1.0 simulates inhibition-dominated (I > E) processing
  - `noise_std` > 0 adds Gaussian noise to neural activity
- No retraining — this probes how a "healthy" pretrained visual system responds under simulated pathology

### 2. Trainable CORnet with E/I activation (`build_cornet_for_training`)

- Loads pretrained CORnet-Z and **replaces all ReLU layers** with `EIRectifiedLinear(alpha, noise_std)`:
  - `y = alpha * ReLU(x) + Gaussian_noise`
- Rebuilds the decoder head: `512 → 64 (ReLU + Dropout) → num_classes`
- Supports full fine-tuning or backbone-frozen training
- Training loop (`train_cornet`) includes batch augmentation, weight decay, and train/val/test evaluation

### Custom activation layer

`EIRectifiedLinear` (PyTorch `nn.Module`) implements gain-modulated ReLU with optional noise injection, parameterized by `alpha` (E/I gain) and `noise_std`.

## Directory structure

```
new/
├── README.md
├── STRUCTURE.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── models/
│   │   ├── cornet.py          # CORnet wrapper, training, feature extraction
│   │   └── custom_layers.py   # EIRectifiedLinear (PyTorch)
│   ├── data/
│   │   ├── loader.py          # Dataset download and preparation
│   │   └── preprocessing.py   # PyTorch Dataset/DataLoader for face images
│   ├── analysis/
│   │   ├── rsa.py             # RSA feature extraction, correlation, multi-run workflows
│   │   └── visualization.py   # Heatmaps, discriminability plots, training curves
│   └── utils/
│       ├── helpers.py         # Device management, environment detection
│       └── io.py              # JSON serialization utilities
├── notebooks/
│   └── CORnet.ipynb           # Supplementary large-scale CORnet experiment notebook
├── data/                      # Dataset manifests only (no raw images)
│   ├── vggface2_identities.csv
│   ├── vggface2_subset_manifest.csv
│   ├── lfw_identities.csv
│   └── lfw_subset_manifest.csv
├── results/EIB/cornet/        # Lightweight JSON results from supplementary CORnet runs
└── rcode/                     # R statistical analyses and supplementary output figures
```

## Scope of this directory

This directory focuses on the supplementary CORnet workflow. Legacy components from the earlier customized-model pipeline are not part of this directory and are instead represented separately in `small-scale-custom-cnn/`.

The present `large-scale-cornet/` package therefore centers on:

- CORnet-based modeling and feature extraction
- large-scale VGGFace2/LFW manifest documentation
- lightweight result archives for supplementary CORnet experiments
- R-based statistical summaries linked to the later-stage analyses

Examples of earlier legacy components that are not part of this supplementary directory include:

- **TF/Keras CNN baseline** (`src/models/cnn.py`): A 3-block Sequential CNN (Conv2D → BatchNorm → MaxPool → Dropout × 3 → Dense(128) → Dense(64) → Softmax) that used `CustomActivation` (a Keras `Layer`) for E/I simulation. This was the earlier baseline before the CORnet approach.
- **Keras data loaders** (`src/data/keras/`): NumPy/Keras-based image loading and preprocessing (`load_image`, `preprocess_image`, `load_dataset`) using `keras.utils.image`.
- **Mid-layer analysis** (`src/analysis/mid_layer.py`): Keras-specific intermediate-layer activation extraction and inactive-neuron analysis (`MidLayerAnalyzer`, `build_activation_model`).
- **TensorFlow paths in RSA** (`extract_features_tensorflow`, `compute_correlations`): TF/Keras feature extraction and correlation functions that were part of the dual-framework RSA module.
- **Legacy CNN results** (`results/EIB/1000/`, `results/EIB/500/`): Training metrics (accuracy, loss JSON files) from the old Keras CNN runs.
- **Keras `CustomActivation` layer**: The TF/Keras version of the E/I activation in `custom_layers.py` (replaced entirely by PyTorch `EIRectifiedLinear`).

## Data policy

No raw VGGFace2 or LFW image files are redistributed. The `data/` directory contains only identity lists and file-reference manifests.

## Quick start

```bash
pip install -r requirements.txt
pip install git+https://github.com/dicarlolab/CORnet.git
```

Entry points:
- `notebooks/CORnet.ipynb` — supplementary large-scale CORnet experiment notebook
- `src/models/cornet.py` — model definitions and training
