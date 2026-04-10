# A Neurocomputational Basis of Face Recognition Changes in ASD

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Preprint](https://img.shields.io/badge/bioRxiv-10.1101%2F2025.04.02.646903-b31b1b)](https://doi.org/10.1101/2025.04.02.646903)

Official code repository for:

> **A neurocomputational basis of face recognition changes in ASD: E/I balance, internal noise, and weak neural representations**
>
> Xijing Wang<sup>1,2</sup>, Emily Rios<sup>2</sup>, and Lang Chen<sup>2,3,*</sup>
>
> <sup>1</sup>Mathematics and Computer Science, <sup>2</sup>Neuroscience Program, <sup>3</sup>Psychology, Santa Clara University
>
> <sup>*</sup>Correspondence: Lang Chen (lchen4@scu.edu)
>
> *Communications Biology* (accepted)
>
> Preprint: [https://doi.org/10.1101/2025.04.02.646903](https://doi.org/10.1101/2025.04.02.646903)

## Overview

This project investigates how Excitatory/Inhibitory (E/I) imbalance and internal neural noise affect face recognition in a biologically grounded computational model of the ventral visual stream. We use **CORnet-Z** — a feedforward CNN whose layers map onto cortical areas V1, V2, V4, and IT — and systematically manipulate the E/I gain ratio across these layers to simulate conditions associated with Autism Spectrum Disorder (ASD).

Key findings:

- Increased excitation (E > I) degrades face identity representations more than inhibition-dominated conditions
- The effect is reflected in both classification accuracy and representational similarity structure (RSA)
- Results are consistent across multiple random initializations and dataset scales

## Repository Structure

```
├── small-scale-custom-cnn/   # Original 50-image pipeline (TensorFlow/Keras baseline)
├── large-scale-cornet/       # CORnet-based large-scale experiments (PyTorch)
│   ├── src/                  # Modular source code (models, data, analysis, utils)
│   ├── notebooks/            # CORnet experiment notebook
│   ├── results/              # Lightweight JSON results
│   ├── rcode/                # R statistical analyses and output figures
│   └── data/                 # Dataset manifests (no raw images)
├── docs/                     # GitHub Pages site assets
├── LICENSE
└── CITATION.cff
```

- **`small-scale-custom-cnn/`** — original customized-CNN pipeline (10 identities, 50 images, TensorFlow/Keras)
- **`large-scale-cornet/`** — CORnet-based experiments with VGGFace2 (100 identities, 10,000 images, PyTorch) and LFW evaluation

See [`small-scale-custom-cnn/README.md`](small-scale-custom-cnn/README.md) and [`large-scale-cornet/README.md`](large-scale-cornet/README.md) for detailed documentation of each pipeline.

## Requirements

Python 3.8+ with PyTorch. For the CORnet pipeline (`large-scale-cornet/`):

```bash
pip install -r large-scale-cornet/requirements.txt
pip install git+https://github.com/dicarlolab/CORnet.git
```

The legacy pipeline (`small-scale-custom-cnn/`) additionally requires TensorFlow/Keras; see that directory for details.

## Reproducing Results

1. **Dataset preparation** — VGGFace2 and LFW images are not redistributed due to licensing. Download them from the official sources and use the manifests in `large-scale-cornet/data/` to reconstruct the subsets used in the paper. The 50-image sample for the small-scale pipeline is included directly.

2. **CORnet experiments** — Open `large-scale-cornet/notebooks/CORnet.ipynb` (designed for Google Colab with GPU). The notebook runs the full pipeline: dataset creation, model training under E/I conditions, RSA extraction, and visualization.

3. **Statistical analysis** — R scripts in `large-scale-cornet/rcode/` reproduce the ANOVA and accuracy analyses. Pre-computed outputs are in `large-scale-cornet/rcode/output/`.

## Data Availability

| Dataset | Included | Notes |
|---------|----------|-------|
| 50-image cropped sample | Yes (`small-scale-custom-cnn/data/`) | Original small-scale experiment |
| VGGFace2 subset manifest | Yes (`large-scale-cornet/data/`) | 100 identities, 10,000 file references |
| LFW subset manifest | Yes (`large-scale-cornet/data/`) | 50 identities, 500 file references |
| Raw VGGFace2 images | No | Download from [VGGFace2](https://github.com/ox-vgg/vgg_face2) |
| Raw LFW images | No | Download from [LFW](http://vis-www.cs.umass.edu/lfw/) |

## Citation

If you use this code or find our work useful, please cite:

```bibtex
@article{wang2025cnnFaceASD,
  title   = {A neurocomputational basis of face recognition changes in {ASD}:
             {E/I} balance, internal noise, and weak neural representations},
  author  = {Wang, Xijing and Rios, Emily and Chen, Lang},
  journal = {Communications Biology},
  year    = {2025},
  doi     = {10.1101/2025.04.02.646903},
  note    = {Accepted; preprint available on bioRxiv}
}
```

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions about the code or data, please [open an issue](../../issues) on this repository or contact Lang Chen at lchen4@scu.edu.
