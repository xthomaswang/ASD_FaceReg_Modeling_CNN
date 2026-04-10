# New Dataset Notes

This directory documents the datasets used by the later-stage pipeline without redistributing copyrighted image data.

## VGGFace2 subset used in this project

- source dataset: VGGFace2
- subset used here: 100 identities x 100 images each
- total listed images: 10,000
- official project page: <https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/>
- manifest files:
  - `vggface2_identities.csv`
  - `vggface2_subset_manifest.csv`

`vggface2_identities.csv` records the 100 selected identity IDs and their names from the official `identity_meta.csv` metadata. `vggface2_subset_manifest.csv` records which image filenames from those identities were used in the balanced 10,000-image subset.

## LFW subset used for additional testing

- source dataset: Labeled Faces in the Wild (LFW)
- subset used here: 50 identities x 10 images each
- total listed images: 500
- official project page: <http://vis-www.cs.umass.edu/lfw/>
- manifest files:
  - `lfw_identities.csv`
  - `lfw_subset_manifest.csv`

This LFW subset was used as a smaller benchmark/evaluation set relative to the VGGFace2 experiments.

## Redistribution policy

The repository does not include raw VGGFace2 or LFW image files. Only dataset descriptions and manifest CSVs are included so the exact identities and image filenames used in the experiments remain documented.
