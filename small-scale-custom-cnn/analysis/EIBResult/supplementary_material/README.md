# EIB Results — Supplementary Material

## Folder Structure Documentation

This folder contains supplementary material analysis results for EIB (Effective Information Bottleneck) experiments, primarily analyzing training data across 1000 epochs under different conditions.

### Top-level Folder Structure

```
supplementary_material/
├── ACC_data_0.R                    # Main analysis script
├── csv_1000_categorical/           # Training Accuracy with categorical cross-entropy loss 
├── csv_1000_negslope/              # Negative slope analysis results
├── csv_1000_negths/                # Negative threshold analysis results
└── csv_1000_posths/                # Positive threshold analysis results
```

---

## File Naming Convention

### Prefix Explanations
- `csv_1000_*`: 1000-epoch training data
- `categorical`: Categorical cross-entropy loss function (baseline)
- `negslope`: Compares negative slope parameters
- `negths`: Compares negative threshold parameters
- `posths`: Compares positive threshold parameters

---

## Detailed Folder Descriptions

### 1. ACC_data_0.R
**Main analysis script** for statistical analysis and visualization
- Set `targeted_folder` to specify which folder to analyze
- Set `compare_index` to choose comparison factor (1=negative_slope, 2=noise, 3=threshold)

### 2. csv_1000_categorical/
**Baseline comparison using categorical cross-entropy loss function**
- Compares positive slopes: [0.005, 0.05, 0.5]
- Original experimental setting with standard categorical loss

### 3. csv_1000_negslope/
**Compares negative slope parameters**
- Negative slope values: [0.005, 0.05, 0.5]
- Grouped by positive slope, compares effect of negative slope

### 4. csv_1000_negths/
**Compares negative threshold parameters**
- Threshold values: [-5, -0.5, 0]
- Grouped by positive slope, compares effect of negative threshold

### 5. csv_1000_posths/
**Compares positive threshold parameters**
- Contains 18 experiments (0-17)
- Analysis results not yet generated

---

## Data File Format Descriptions

### trainedACC_1000_*.csv
**Training accuracy data files** containing:
- `slope`: Slope parameter tuple in format like "(0.005, 0, 0)"
- `epoch_1, epoch_2, ..., epoch_1000`: Training accuracy for each epoch
- Each file corresponds to one experimental condition


#### EIB_*_tests.txt
Statistical test results containing:
- ANOVA test results
- Tukey HSD multiple comparison results
- Independent test results for each epoch

---

## Analysis Workflow

1. **Data Preprocessing**: Load data from CSV files and parse slope parameters
2. **Factor Construction**: Build outer and inner comparison factors based on analysis objectives
3. **Data Reshaping**: Convert wide-format data to long format for analysis
4. **Statistical Analysis**: Perform repeated measures ANOVA and multiple comparisons
5. **Visualization**: Generate line plots and box plots to display results
6. **Results Saving**: Save statistical results and charts to analysis directory

---
## Important Notes

- All analyses are based on 1000-epoch training data
- Data is sampled every 10 epochs for analysis
- Statistical tests are performed at epochs 100, 300, 500
- Charts use consistent color schemes for easy comparison 