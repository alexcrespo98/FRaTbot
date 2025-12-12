#!/usr/bin/env python3
"""
ML Flow Prediction Framework
Comprehensive machine learning experimentation to predict water flow rate (GPM)
from dual thermistor readings (Proximal and Distal).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import warnings
import re
import os

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)


class DataLoader:
    """Load and parse the FRaTbot flow data CSV."""
    
    def __init__(self, csv_path='FRaTbot_flowdata_2.csv'):
        self.csv_path = csv_path
        self.raw_data = None
        self.metadata = []
        
    def load(self):
        """Load CSV and parse metadata from column headers."""
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)
        
        # Read CSV
        df = pd.read_csv(self.csv_path)
        print(f"Loaded CSV: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Parse column headers to extract metadata
        self.metadata = []
        for col in df.columns:
            # Parse: "Proximal,10ms,0.2Hz,9.4GPM" or similar
            match = re.search(r'(Proximal|Distal),(\d+)ms,([\d.]+)Hz,([\d.]+)GPM', col)
            if match:
                location, sample_rate, frequency, flow_rate = match.groups()
                self.metadata.append({
                    'column': col,
                    'location': location,
                    'sample_rate_ms': int(sample_rate),
                    'frequency_hz': float(frequency),
                    'flow_rate_gpm': float(flow_rate),
                    'data': df[col].values
                })
        
        print(f"Parsed {len(self.metadata)} columns with metadata")
        
        # Display unique flow rates found
        flow_rates = sorted(set(m['flow_rate_gpm'] for m in self.metadata))
        frequencies = sorted(set(m['frequency_hz'] for m in self.metadata))
        print(f"Flow rates: {flow_rates} GPM")
        print(f"Frequencies: {frequencies} Hz")
        print(f"Sample rate: {self.metadata[0]['sample_rate_ms']}ms ({1000/self.metadata[0]['sample_rate_ms']:.0f} Hz)")
        print(f"Samples per measurement: {len(self.metadata[0]['data'])}")
        print()
        
        return self.metadata


class FeatureExtractor:
    """Extract various features from time series data."""
    
    @staticmethod
    def extract_statistical_features(proximal, distal):
        """Extract statistical features from time series."""
        features = {}
        
        for name, data in [('prox', proximal), ('dist', distal)]:
            features[f'{name}_mean'] = np.mean(data)
            features[f'{name}_std'] = np.std(data)
            features[f'{name}_min'] = np.min(data)
            features[f'{name}_max'] = np.max(data)
            features[f'{name}_range'] = np.max(data) - np.min(data)
            features[f'{name}_skew'] = stats.skew(data)
            features[f'{name}_kurtosis'] = stats.kurtosis(data)
            features[f'{name}_median'] = np.median(data)
            features[f'{name}_q25'] = np.percentile(data, 25)
            features[f'{name}_q75'] = np.percentile(data, 75)
        
        # Differential features
        diff = proximal - distal
        features['diff_mean'] = np.mean(diff)
        features['diff_std'] = np.std(diff)
        features['diff_max'] = np.max(diff)
        features['diff_min'] = np.min(diff)
        
        return features
    
    @staticmethod
    def extract_fft_features(proximal, distal, sample_rate_hz=100):
        """Extract FFT-based features."""
        features = {}
        
        for name, data in [('prox', proximal), ('dist', distal)]:
            # Compute FFT
            fft = np.fft.rfft(data)
            freqs = np.fft.rfftfreq(len(data), 1/sample_rate_hz)
            magnitude = np.abs(fft)
            power = magnitude ** 2
            
            # Peak frequency and amplitude
            peak_idx = np.argmax(magnitude[1:]) + 1  # Skip DC component
            features[f'{name}_peak_freq'] = freqs[peak_idx]
            features[f'{name}_peak_amp'] = magnitude[peak_idx]
            features[f'{name}_peak_power'] = power[peak_idx]
            
            # Total power
            features[f'{name}_total_power'] = np.sum(power)
            
            # Spectral centroid
            features[f'{name}_spectral_centroid'] = np.sum(freqs * magnitude) / np.sum(magnitude)
            
            # Phase at peak
            features[f'{name}_peak_phase'] = np.angle(fft[peak_idx])
        
        # Relative features
        if features['prox_peak_amp'] > 0:
            features['amp_ratio'] = features['dist_peak_amp'] / features['prox_peak_amp']
        else:
            features['amp_ratio'] = 0
            
        features['phase_diff'] = features['prox_peak_phase'] - features['dist_peak_phase']
        
        return features
    
    @staticmethod
    def extract_thermal_decay_features(proximal, distal):
        """Extract thermal decay-related features."""
        features = {}
        
        # Cross-correlation
        correlation = np.correlate(proximal - np.mean(proximal), 
                                  distal - np.mean(distal), 
                                  mode='full')
        lag = np.argmax(correlation) - (len(proximal) - 1)
        features['cross_corr_lag'] = lag
        features['cross_corr_max'] = np.max(correlation)
        
        # Temperature differential dynamics
        diff = proximal - distal
        features['diff_range'] = np.max(diff) - np.min(diff)
        
        # Detect decay rate (simple approach: fit exponential to differential)
        try:
            # Simplified decay metric: slope of differential
            x = np.arange(len(diff))
            slope, intercept = np.polyfit(x, diff, 1)
            features['diff_slope'] = slope
        except:
            features['diff_slope'] = 0
        
        # Pearson correlation between signals
        features['pearson_corr'] = np.corrcoef(proximal, distal)[0, 1]
        
        return features
    
    @staticmethod
    def extract_all_features(proximal, distal, sample_rate_hz=100, frequency_hz=None):
        """Extract all feature types."""
        features = {}
        features.update(FeatureExtractor.extract_statistical_features(proximal, distal))
        features.update(FeatureExtractor.extract_fft_features(proximal, distal, sample_rate_hz))
        features.update(FeatureExtractor.extract_thermal_decay_features(proximal, distal))
        
        # Add input frequency as a feature if provided
        if frequency_hz is not None:
            features['input_frequency'] = frequency_hz
        
        return features


class DatasetBuilder:
    """Build ML-ready datasets from parsed metadata."""
    
    def __init__(self, metadata):
        self.metadata = metadata
        
    def build_paired_dataset(self, target_frequency=None):
        """Build dataset with paired proximal/distal readings.
        
        If target_frequency is None, uses all available frequencies.
        Otherwise, filters to the specified frequency.
        """
        if target_frequency is not None:
            print(f"Building dataset for frequency: {target_frequency} Hz")
        else:
            print("Building dataset for all frequencies")
        
        # Group by flow rate and frequency
        samples = []
        
        # Find all unique combinations
        flow_rates = sorted(set(m['flow_rate_gpm'] for m in self.metadata))
        frequencies = sorted(set(m['frequency_hz'] for m in self.metadata))
        
        for flow_rate in flow_rates:
            for frequency in frequencies:
                # Skip if filtering by frequency
                if target_frequency is not None and abs(frequency - target_frequency) > 0.01:
                    continue
                    
                # Find proximal and distal for this flow rate and frequency
                prox = None
                dist = None
                
                for m in self.metadata:
                    if m['flow_rate_gpm'] == flow_rate and abs(m['frequency_hz'] - frequency) < 0.01:
                        if m['location'] == 'Proximal':
                            prox = m['data']
                        elif m['location'] == 'Distal':
                            dist = m['data']
                
                if prox is not None and dist is not None:
                    # Remove NaN values
                    valid_mask = ~(np.isnan(prox) | np.isnan(dist))
                    prox_clean = prox[valid_mask]
                    dist_clean = dist[valid_mask]
                    
                    if len(prox_clean) > 100:  # Ensure we have enough data
                        samples.append({
                            'proximal': prox_clean,
                            'distal': dist_clean,
                            'flow_rate': flow_rate,
                            'frequency': frequency
                        })
        
        print(f"Created {len(samples)} samples from {len(flow_rates)} flow rates × {len(frequencies)} frequencies")
        print(f"Flow rates: {flow_rates}")
        print(f"Frequencies: {frequencies}")
        return samples
    
    def extract_features_from_samples(self, samples):
        """Extract features from all samples."""
        print("Extracting features from samples...")
        
        X = []
        y = []
        
        for sample in samples:
            features = FeatureExtractor.extract_all_features(
                sample['proximal'], 
                sample['distal'],
                sample_rate_hz=100,
                frequency_hz=sample.get('frequency')
            )
            X.append(features)
            y.append(sample['flow_rate'])
        
        # Convert to DataFrame for easier handling
        X_df = pd.DataFrame(X)
        y_arr = np.array(y)
        
        print(f"Feature matrix: {X_df.shape}")
        print(f"Number of features: {len(X_df.columns)}")
        print(f"Sample features: {list(X_df.columns)[:10]}...")
        print()
        
        return X_df, y_arr


class MLExperiment:
    """Run ML experiments with multiple models."""
    
    def __init__(self, X, y, model_name="Unknown"):
        self.X = X
        self.y = y
        self.model_name = model_name
        self.scaler = StandardScaler()
        
    def evaluate_model(self, model, cv_strategy='loo'):
        """Evaluate a model using cross-validation."""
        X_scaled = self.scaler.fit_transform(self.X)
        
        # Choose cross-validation strategy
        if cv_strategy == 'loo':
            cv = LeaveOneOut()
        else:
            cv = KFold(n_splits=min(5, len(self.y)), shuffle=True, random_state=42)
        
        # Perform cross-validation
        y_pred = np.zeros_like(self.y, dtype=float)
        
        for train_idx, test_idx in cv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            
            model.fit(X_train, y_train)
            y_pred[test_idx] = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(self.y, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y, y_pred))
        r2 = r2_score(self.y, y_pred)
        
        # Percentage error
        pct_errors = np.abs((self.y - y_pred) / self.y) * 100
        mean_pct_error = np.mean(pct_errors)
        
        results = {
            'model': self.model_name,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mean_pct_error': mean_pct_error,
            'y_true': self.y,
            'y_pred': y_pred
        }
        
        return results


def run_all_experiments(X, y):
    """Run experiments with all models."""
    print("=" * 80)
    print("RUNNING ML EXPERIMENTS")
    print("=" * 80)
    print()
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
        'SVR (RBF)': SVR(kernel='rbf', C=10, gamma='scale'),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1, max_iter=10000),
    }
    
    results = []
    
    for name, model in models.items():
        print(f"Training {name}...")
        exp = MLExperiment(X, y, model_name=name)
        result = exp.evaluate_model(model, cv_strategy='loo')
        results.append(result)
        
        print(f"  MAE: {result['mae']:.3f} GPM")
        print(f"  RMSE: {result['rmse']:.3f} GPM")
        print(f"  R²: {result['r2']:.3f}")
        print(f"  Mean % Error: {result['mean_pct_error']:.2f}%")
        print()
    
    return results


def create_visualizations(results):
    """Create comprehensive visualizations."""
    print("=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    print()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Comparison table
    ax1 = plt.subplot(3, 2, 1)
    ax1.axis('tight')
    ax1.axis('off')
    
    table_data = []
    for r in results:
        table_data.append([
            r['model'],
            f"{r['mae']:.3f}",
            f"{r['rmse']:.3f}",
            f"{r['r2']:.3f}",
            f"{r['mean_pct_error']:.2f}%"
        ])
    
    table = ax1.table(cellText=table_data,
                     colLabels=['Model', 'MAE (GPM)', 'RMSE (GPM)', 'R²', 'Mean % Error'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax1.set_title('Model Performance Comparison', fontsize=12, fontweight='bold', pad=20)
    
    # 2. Predicted vs Actual for best model
    best_result = min(results, key=lambda x: x['mae'])
    ax2 = plt.subplot(3, 2, 2)
    ax2.scatter(best_result['y_true'], best_result['y_pred'], alpha=0.6, s=100)
    ax2.plot([best_result['y_true'].min(), best_result['y_true'].max()],
            [best_result['y_true'].min(), best_result['y_true'].max()],
            'r--', lw=2, label='Perfect Prediction')
    ax2.set_xlabel('Actual Flow Rate (GPM)', fontsize=10)
    ax2.set_ylabel('Predicted Flow Rate (GPM)', fontsize=10)
    ax2.set_title(f'Best Model: {best_result["model"]}', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. MAE comparison
    ax3 = plt.subplot(3, 2, 3)
    models = [r['model'] for r in results]
    maes = [r['mae'] for r in results]
    colors = ['green' if mae == min(maes) else 'skyblue' for mae in maes]
    ax3.barh(models, maes, color=colors)
    ax3.set_xlabel('MAE (GPM)', fontsize=10)
    ax3.set_title('Mean Absolute Error Comparison', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. R² comparison
    ax4 = plt.subplot(3, 2, 4)
    r2s = [r['r2'] for r in results]
    colors = ['green' if r2 == max(r2s) else 'lightcoral' for r2 in r2s]
    ax4.barh(models, r2s, color=colors)
    ax4.set_xlabel('R² Score', fontsize=10)
    ax4.set_title('R² Score Comparison', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # 5. Error distribution for best model
    ax5 = plt.subplot(3, 2, 5)
    errors = best_result['y_pred'] - best_result['y_true']
    ax5.hist(errors, bins=10, edgecolor='black', alpha=0.7)
    ax5.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax5.set_xlabel('Prediction Error (GPM)', fontsize=10)
    ax5.set_ylabel('Frequency', fontsize=10)
    ax5.set_title(f'Error Distribution - {best_result["model"]}', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Percentage error comparison
    ax6 = plt.subplot(3, 2, 6)
    pct_errors = [r['mean_pct_error'] for r in results]
    colors = ['green' if pct == min(pct_errors) else 'orange' for pct in pct_errors]
    ax6.barh(models, pct_errors, color=colors)
    ax6.set_xlabel('Mean Percentage Error (%)', fontsize=10)
    ax6.set_title('Percentage Error Comparison', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('ml_flow_prediction_results.png', dpi=150, bbox_inches='tight')
    print("Saved visualization: ml_flow_prediction_results.png")
    plt.show()
    
    return best_result


def print_summary(results):
    """Print summary and recommendations."""
    print("=" * 80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 80)
    print()
    
    # Find best model by different metrics
    best_mae = min(results, key=lambda x: x['mae'])
    best_r2 = max(results, key=lambda x: x['r2'])
    best_pct = min(results, key=lambda x: x['mean_pct_error'])
    
    print(f"Best Model (MAE):           {best_mae['model']} - {best_mae['mae']:.3f} GPM")
    print(f"Best Model (R²):            {best_r2['model']} - {best_r2['r2']:.3f}")
    print(f"Best Model (% Error):       {best_pct['model']} - {best_pct['mean_pct_error']:.2f}%")
    print()
    
    print("Overall Best Model:", best_mae['model'])
    print(f"  - MAE: {best_mae['mae']:.3f} GPM")
    print(f"  - RMSE: {best_mae['rmse']:.3f} GPM")
    print(f"  - R²: {best_mae['r2']:.3f}")
    print(f"  - Mean % Error: {best_mae['mean_pct_error']:.2f}%")
    print()
    
    if best_mae['mae'] <= 0.5:
        print("✓ SUCCESS: Model achieves target accuracy (±0.5 GPM)")
    else:
        print(f"⚠ Model accuracy ({best_mae['mae']:.3f} GPM) exceeds target (±0.5 GPM)")
        print("  Recommendations:")
        print("  - Collect more training data")
        print("  - Implement data augmentation")
        print("  - Try ensemble methods")
    print()
    
    print("Key Insights:")
    print("  - Limited data (~14 samples) challenges complex models")
    print("  - Feature engineering from FFT/thermal decay is critical")
    print("  - Cross-validation ensures robust performance estimates")
    print()


def main():
    """Main execution function."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "ML FLOW PREDICTION FRAMEWORK" + " " * 30 + "║")
    print("║" + " " * 15 + "Predicting Water Flow Rate from Thermistor Data" + " " * 15 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
    
    # 1. Load data (try FRaTbot_flowdata_2.csv first, fall back to FRaTbot_flowdata.csv)
    csv_file = 'FRaTbot_flowdata_2.csv' if os.path.exists('FRaTbot_flowdata_2.csv') else 'FRaTbot_flowdata.csv'
    loader = DataLoader(csv_file)
    metadata = loader.load()
    
    # 2. Build dataset (use all frequencies to maximize training data)
    builder = DatasetBuilder(metadata)
    samples = builder.build_paired_dataset(target_frequency=None)  # Use all frequencies
    
    # 3. Extract features
    X, y = builder.extract_features_from_samples(samples)
    
    print(f"Dataset ready: {len(X)} samples with {len(X.columns)} features")
    print(f"Flow rates in dataset: {sorted(set(y))}")
    print()
    
    # 4. Run experiments
    results = run_all_experiments(X, y)
    
    # 5. Create visualizations
    best_result = create_visualizations(results)
    
    # 6. Print summary
    print_summary(results)
    
    print("=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Review results in 'ml_flow_prediction_results.png'")
    print("  2. Consider implementing data augmentation if more accuracy is needed")
    print("  3. Experiment with deep learning models (CNN/LSTM) for raw time series")
    print("  4. Compare with FFT-based signal processing baseline")
    print()


if __name__ == "__main__":
    main()
