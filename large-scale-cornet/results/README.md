# Results

This directory preserves lightweight outputs from the CORnet E/I imbalance runs.

Only small result files are kept in the public repository:

- training histories
- RSA label orderings
- run-level summaries

Large artifacts such as `.pt`, `.npy`, and `.npz` files are intentionally omitted.

## Included

- `EIB/cornet/`: CORnet experiment outputs across older pilot sweeps and later VGGFace2 runs

## File Types

Within each `run_x/` directory:

- `all_raw_histories.json`: training / validation / test metrics for all three conditions in that run
- `train_labels_by_condition.json`: ordered training-set labels used for RSA in that run
- `test_labels_by_condition.json`: ordered test-set labels used for RSA in that run

Within each condition subfolder:

- `Balanced/history_runX.json`
- `Balanced/rsa_train_labels_runX.json`
- `Balanced/rsa_test_labels_runX.json`
- and the same pattern for `Excitated/` and `Inhibitated/`

Older aggregate-only exports may also include:

- `averaged_histories.json`

## Current Data Categories

The result folders currently included in this repository fall into four groups.

### 1. Alpha sweep pilot runs

These are the older small-scale pilot experiments:

- `.5_1_2_10L_100E_10P_50I`
- `.2_1_5_10L_100E_10P_50I`

What they test:

- different E/I alpha triplets
  - `[0.5, 1.0, 2.0]`
  - `[0.2, 1.0, 5.0]`

What stays fixed:

- `10` runs
- `100` epochs
- `10` identities
- `50` images per identity
- according to the notebook, `penultimate_dim=64`
- according to the notebook, `dropout=0.5`

Interpretation:

- this is primarily an alpha-condition sweep

### 2. Decoder / dropout pilot sweeps on VGGFace2

These are the later VGGFace2 pilot runs used to compare decoder width and dropout:

- `100_1_100_100_2026-01-18_01-11-35_128_0.2`
- `80_1_100_100_2026-01-17_22-47-41_256_0.1`
- `80_1_100_100_2026-01-17_23-02-53_256_0.2`

What they test:

- decoder width
  - `128`
  - `256`
- dropout
  - `0.1`
  - `0.2`
- epoch budget
  - `80`
  - `100`

What stays fixed:

- `1` run per folder
- `100` identities
- `100` images per identity
- shared VGGFace2 loaders from the notebook
- notebook sections `8.4a` to `8.5` use `SAFE_BATCH_SIZE = 2048`

Interpretation:

- this is mainly a decoder/dropout tuning stage

Note:

- notebook section `8.4a` also mentions `decoder=64`, `dropout=0.0`, `1 run x 100 epochs`
- the present repository snapshot focuses on the result sets retained for lightweight public release

### 3. Selected final VGGFace2 configuration

- `100_20_100_100_2026-02-07_03-11-07`

What it represents:

- the selected CORnet configuration after the pilot sweeps
- `20` independent runs
- `100` epochs
- `100` identities
- `100` images per identity
- notebook section `8.5` identifies this setting as `decoder=128`, `dropout=0.2`

What stays fixed:

- same VGGFace2 setup as the later pilot sweeps
- same shared notebook loaders
- same `SAFE_BATCH_SIZE = 2048` notebook setup

Interpretation:

- this is the main final experiment batch, not another hyperparameter sweep

### 4. Aggregate-only legacy export

- `100_0`

What it contains:

- only `averaged_histories.json`

Interpretation:

- legacy aggregate summary export

## About Batch Size

Batch size is documented in the notebook workflow rather than encoded directly in the retained result folder names.

For the result groups preserved here:

- the later decoder/dropout sweeps and the final selected-config run reuse notebook loaders built with `SAFE_BATCH_SIZE = 2048`
- the notebook also contains a separate rerun with `BATCH_SIZE=512`, `penultimate_dim=64`, and `dropout=0.5`

Accordingly, the public result archive is most naturally organized by:

- alpha-condition set
- decoder width
- dropout
- epoch count
- run count

## Folder Name Guide

There are two main naming styles in this directory.

### 1. Newer VGGFace2 experiment batches

General pattern:

- `epochs_runs_n_people_imgs_per_person_timestamp`
- `epochs_runs_n_people_imgs_per_person_timestamp_decoder_dropout`

Meaning:

- first number: training epochs
- second number: number of independent runs
- third number: number of identities (`N_PEOPLE`)
- fourth number: images per identity (`IMGS_PER_PERSON`)
- timestamp: experiment creation time
- optional final suffixes:
  - penultimate decoder width
  - decoder dropout

Important note:

- `batch size` is not encoded in these folder names
- in the notebook sections that generated the `8.4a` through `8.5` sweeps, the shared loaders are created with `SAFE_BATCH_SIZE = 2048`
- this means decoder width and dropout may appear in the folder name, but batch size is inherited from the notebook setup rather than embedded in the directory name

Examples:

- `80_1_100_100_2026-01-17_22-47-41_256_0.1`
  - 80 epochs
  - 1 run
  - 100 identities
  - 100 images per identity
  - timestamp `2026-01-17_22-47-41`
  - decoder width `256`
  - dropout `0.1`
  - corresponds to notebook section `8.4c`

- `80_1_100_100_2026-01-17_23-02-53_256_0.2`
  - 80 epochs
  - 1 run
  - 100 identities
  - 100 images per identity
  - timestamp `2026-01-17_23-02-53`
  - decoder width `256`
  - dropout `0.2`
  - corresponds to notebook section `8.4d`

- `100_1_100_100_2026-01-18_01-11-35_128_0.2`
  - 100 epochs
  - 1 run
  - 100 identities
  - 100 images per identity
  - timestamp `2026-01-18_01-11-35`
  - decoder width `128`
  - dropout `0.2`
  - corresponds to notebook section `8.4b`

- `100_20_100_100_2026-02-07_03-11-07`
  - 100 epochs
  - 20 runs
  - 100 identities
  - 100 images per identity
  - timestamp `2026-02-07_03-11-07`
  - according to the notebook, this selected batch used `decoder=128` and `dropout=0.2`
  - those two settings are not encoded in the folder name itself
  - corresponds to notebook section `8.5`

### 2. Older hidden pilot batches

Examples:

- `.5_1_2_10L_100E_10P_50I`
- `.2_1_5_10L_100E_10P_50I`

These folders begin with `.` because the first alpha value was written without a leading zero, so they may appear hidden in some file browsers.

Confirmed or strongly inferred meaning:

- leading triplet:
  - `.5_1_2` = alpha conditions `[0.5, 1.0, 2.0]`
  - `.2_1_5` = alpha conditions `[0.2, 1.0, 5.0]`
- `100E` = 100 epochs
- `10P` = 10 identities / people
- `50I` = 50 images per identity
- `10L` is inferred to mean 10 loops or 10 runs
  - this is based on the presence of `run_0` through `run_9`

These older batches appear to be small LFW-style pilot experiments rather than the later VGGFace2 runs:

- labels are person names such as `Colin_Powell`
- there are 10 identities
- each identity contributes 50 images total
- the saved train RSA labels show 40 images per identity, consistent with an 80/10/10 split

### 3. Legacy aggregate-only folder

- `100_0`

This folder only contains `averaged_histories.json`.

This folder is retained as a legacy aggregate summary export.

## Conditions

Within each run, files are grouped into the three experimental conditions:

- `Inhibitated`: lower alpha / under-excitation condition
- `Balanced`: baseline alpha = 1.0 condition
- `Excitated`: higher alpha / over-excitation condition

The spellings `Inhibitated` and `Excitated` are kept exactly as they appear in the original saved outputs.

## Note

The goal of this directory is reproducible lightweight reporting, not artifact hosting. Checkpoints, intermediate RSA arrays, and matrix dumps remain excluded from the public repository.
