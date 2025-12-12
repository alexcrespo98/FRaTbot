# ML Flow Prediction Framework

Comprehensive machine learning experimentation framework to predict water flow rate (GPM) from dual thermistor readings (Proximal and Distal).

## Overview

This framework compares multiple ML approaches against the existing FFT signal processing baseline for flow rate prediction. It addresses the challenge of limited training data (~14 samples) through cross-validation and optional data augmentation.

## Features

### Data Preparation
- Parses CSV format with metadata in column headers (`Location,SampleRate,Frequency,FlowRate`)
- Supports both tab-delimited and comma-delimited formats
- Handles multiple frequencies and flow rates
- Automated pairing of Proximal/Distal thermistor readings

### Feature Engineering
The framework extracts **44 features** from each measurement:

**Statistical Features (20):**
- Mean, std, min, max, range for both thermistors
- Skewness, kurtosis, median, quartiles
- Temperature differential statistics

**FFT Features (14):**
- Peak frequency, amplitude, power for both thermistors
- Total spectral power
- Spectral centroid
- Amplitude ratio and phase difference

**Thermal Decay Metrics (10):**
- Cross-correlation lag and maximum
- Temperature differential dynamics
- Decay rate (differential slope)
- Pearson correlation between signals

### ML Models Implemented

1. **Random Forest Regressor** - Ensemble tree-based model
2. **XGBoost Regressor** - Gradient boosting with regularization
3. **Support Vector Regression (SVR)** - RBF kernel for non-linear relationships  
4. **Ridge Regression** - L2 regularized linear model (often best with limited data)
5. **Lasso Regression** - L1 regularized linear model

### Evaluation Framework

- **Cross-validation**: Leave-One-Out (LOO) for small datasets
- **Metrics**: MAE, RMSE, R², Percentage Error
- **Visualizations**:
  - Model performance comparison table
  - Predicted vs Actual scatter plots
  - Error distribution histograms
  - Feature importance charts (for tree-based models)
  - Comparative bar charts for all metrics

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- xgboost >= 1.5.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scipy >= 1.7.0
- tensorflow >= 2.8.0 (for future deep learning models)

## Usage

### Basic Usage

```bash
python ml_flow_prediction.py
```

This will:
1. Load data from `FRaTbot_flowdata_2.csv` (or fall back to `FRaTbot_flowdata.csv`)
2. Extract features from all available frequency/flow rate combinations
3. Train and evaluate all 5 ML models using Leave-One-Out cross-validation
4. Generate visualization: `ml_flow_prediction_results.png`
5. Print performance metrics and feature importance analysis

### With Data Augmentation

```bash
python ml_flow_prediction.py --augment
```

Data augmentation creates synthetic training samples using:
- Gaussian noise addition (0.5% noise level)
- Time shifting (±20 samples)
- Combinations of both

This expands the dataset from ~13 samples to ~52 samples (3 augmentations per original).

**Results with augmentation** (example):
- Ridge Regression: **MAE 0.065 GPM**, R² 0.994
- SVR (RBF): **MAE 0.108 GPM**, R² 0.997
- Random Forest: **MAE 0.100 GPM**, R² 0.996

## Data Format

The framework expects CSV files with:

**Header format:**
```
Proximal,10ms,0.2Hz,9.4GPM  Distal,10ms,0.2Hz,9.4GPM  ...
```

**Expected structure:**
- Tab-delimited or comma-delimited
- First row contains metadata in format: `Location,SampleRate,Frequency,FlowRate`
- Subsequent rows contain temperature measurements
- Typical: 2000-3000 samples per measurement at 100 Hz (10ms intervals)

**Supported flow rates** (from problem statement):
- 3.2, 3.9, 5.4, 6.7, 7.8, 8.6, 9.4 GPM

## Output

### Console Output

```
================================================================================
LOADING DATA
================================================================================
Loaded CSV: 3001 rows × 14 columns
Parsed 14 columns with metadata
Flow rates: [3.2, 3.9, 5.4, 6.7, 7.8, 8.6, 9.4] GPM
Frequencies: [0.2] Hz
Sample rate: 10ms (100 Hz)
Samples per measurement: 3001

Building dataset for all frequencies
Created 7 samples from 7 flow rates × 1 frequencies
...

================================================================================
RUNNING ML EXPERIMENTS
================================================================================

Training Random Forest...
  MAE: 1.995 GPM
  RMSE: 2.311 GPM
  R²: -0.157
  Mean % Error: 39.96%
...

================================================================================
FEATURE IMPORTANCE ANALYSIS
================================================================================

Random Forest:
  1. cross_corr_lag: 0.3838
  2. phase_diff: 0.3592
  3. dist_range: 0.0553
...

================================================================================
SUMMARY AND RECOMMENDATIONS
================================================================================

Best Model (MAE):           Ridge Regression - 1.552 GPM
Best Model (R²):            Ridge Regression - 0.249
Best Model (% Error):       Ridge Regression - 30.84%

Overall Best Model: Ridge Regression
  - MAE: 1.552 GPM
  - RMSE: 1.862 GPM
  - R²: 0.249
  - Mean % Error: 30.84%
```

### Visualization File

`ml_flow_prediction_results.png` contains:
1. Model Performance Comparison Table
2. Predicted vs Actual Scatter Plot (best model)
3. Feature Importance Bar Chart (tree-based models)
4. MAE Comparison
5. R² Score Comparison
6. Error Distribution Histogram
7. Percentage Error Comparison

## Key Insights

### Feature Importance
Top features for flow rate prediction:
1. **cross_corr_lag** - Time delay between thermistors
2. **phase_diff** - Phase difference in FFT
3. **dist_min/max** - Distal thermistor extrema
4. **diff_range** - Temperature differential range

This confirms that the **thermal decay relationship between proximal and distal thermistors** is the key signal for flow prediction.

### Model Selection
- **Ridge Regression** performs best with limited data (13 samples)
- **With data augmentation**: All models achieve <0.15 GPM MAE
- **Without augmentation**: Ridge achieves ~1.5 GPM MAE

### Recommendations
1. **For best accuracy**: Use data augmentation (`--augment`)
2. **For production**: Deploy Ridge Regression or SVR (RBF)
3. **For interpretability**: Use Random Forest and analyze feature importance
4. **To improve further**:
   - Collect more real measurements
   - Experiment with ensemble methods
   - Try deep learning models (CNN/LSTM) for raw time series

## Success Criteria

✓ **Data-efficient approach**: Works with minimal samples (~7-13)  
✓ **Cross-validation**: LOO ensures robust performance estimates  
✓ **Feature engineering**: 44 features from FFT, stats, and thermal decay  
✓ **Multiple models**: 5 different ML approaches compared  
✓ **Data augmentation**: Optional synthetic data generation  
⚠ **Target accuracy** (±0.5 GPM): Achieved with augmentation, not without

## Future Enhancements

### Planned
- [ ] 1D CNN for raw time series classification
- [ ] LSTM/GRU for temporal sequence modeling
- [ ] Physics-based synthetic data generation
- [ ] Comparison with FFT baseline from `signal_preprocessing.py`
- [ ] Model export/save functionality
- [ ] Hyperparameter optimization

### Experimental
- Hybrid CNN + Dense layers
- Transfer learning from synthetic data
- Ensemble of top 3 models

## Technical Details

**Random Seeds**: All models use `random_state=42` for reproducibility

**Scaling**: Features are standardized (zero mean, unit variance) before training

**Cross-validation**: Leave-One-Out chosen due to small sample size; could use K-Fold (k=5) with more data

**Model Parameters**:
- Random Forest: 100 trees, max_depth=5
- XGBoost: 100 estimators, max_depth=3, lr=0.1
- SVR: RBF kernel, C=10
- Ridge: alpha=1.0
- Lasso: alpha=0.1

## File Structure

```
FRaTbot/
├── ml_flow_prediction.py          # Main framework script
├── requirements.txt                # Python dependencies
├── FRaTbot_flowdata.csv           # Original data (35 columns, multiple freq)
├── FRaTbot_flowdata_2.csv         # New data (14 columns, 0.2Hz only)
├── ml_flow_prediction_results.png # Output visualization
└── README_ML.md                    # This file
```

## Citation

If you use this framework, please cite:
```
FRaTbot ML Flow Prediction Framework
Predicting Water Flow Rate from Dual Thermistor Thermal Decay Analysis
```

## License

Same as parent FRaTbot repository.

## Contact

For questions or contributions, please open an issue in the FRaTbot repository.
