# Dataset Notes

This directory documents the datasets used by the large-scale CORnet pipeline without redistributing copyrighted image data.

## VGGFace2 subset

- Source dataset: VGGFace2
- Subset used: 100 identities x 100 images each (10,000 total)
- Official project page: <https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/>
- Manifest files:
  - `vggface2_identities.csv` — the 100 selected identity IDs and names
  - `vggface2_subset_manifest.csv` — the 10,000 image filenames used in the balanced subset

These are the identities and images used in the analyses reported in the paper and supplementary materials.

## LFW files (exploratory only)

This directory also contains `lfw_identities.csv` and `lfw_subset_manifest.csv` (50 identities, 500 images). LFW was used only for exploratory internal testing and is not part of the reported analyses presented in the paper or supplementary materials. Raw LFW images are not redistributed in this repository. Users interested in this dataset should obtain it from the official [LFW source](http://vis-www.cs.umass.edu/lfw/) and follow the corresponding dataset terms and citation requirements.

## Redistribution policy

The repository does not include raw VGGFace2 or LFW image files. Only manifest CSVs are included so the exact identities and filenames used in the experiments remain documented.
