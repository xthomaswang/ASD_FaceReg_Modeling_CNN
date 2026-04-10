# CORnet Pipeline Structure

```text
new/
├── README.md
├── STRUCTURE.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cornet.py          # CORnet wrapper, training, feature extraction
│   │   └── custom_layers.py   # EIRectifiedLinear (PyTorch)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py          # Dataset download and preparation
│   │   └── preprocessing.py   # PyTorch Dataset/DataLoader for face images
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── rsa.py             # RSA feature extraction, correlation, multi-run workflows
│   │   └── visualization.py   # Heatmaps, discriminability plots, training curves
│   └── utils/
│       ├── __init__.py
│       ├── helpers.py         # Device management, environment detection
│       └── io.py              # JSON serialization utilities
├── notebooks/
│   └── CORnet.ipynb           # Main experiment notebook
├── data/
│   ├── README.md
│   ├── vggface2_identities.csv
│   ├── vggface2_subset_manifest.csv
│   ├── lfw_identities.csv
│   └── lfw_subset_manifest.csv
├── results/
│   └── EIB/cornet/            # Lightweight JSON results from CORnet runs
└── rcode/
    ├── ACC_data_EIB.R         # Accuracy analysis
    ├── Repres_data_EIB.R      # Representational analysis
    └── output/                # Statistical output figures and tables
```
