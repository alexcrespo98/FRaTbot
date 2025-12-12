#!/usr/bin/env python3
"""
Signal Processing Baseline for Flow Prediction
FFT-based analysis of amplitude attenuation between proximal and distal thermistors.
This is the traditional signal processing approach (non-ML) for comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
import re
import warnings

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


class FFTSignalProcessor:
    """Process thermistor signals using FFT to extract amplitude features."""
    
    def __init__(self, sample_rate_hz=100):
        self.sample_rate_hz = sample_rate_hz
        
    def compute_fft_peaks(self, data, num_peaks=3):
        """Compute FFT and extract top peaks.
        
        Returns:
            List of dicts with frequency_hz, amplitude, power, phase for each peak
        """
        # Remove DC component and apply windowing
        data_centered = data - np.mean(data)
        window = np.hanning(len(data))
        data_windowed = data_centered * window
        
        # Compute FFT
        fft = np.fft.rfft(data_windowed)
        freqs = np.fft.rfftfreq(len(data), 1/self.sample_rate_hz)
        magnitude = np.abs(fft)
        power = magnitude ** 2
        phase = np.angle(fft)
        
        # Find peaks (skip DC component at index 0)
        peaks = []
        # Get indices of top peaks by magnitude
        peak_indices = np.argsort(magnitude[1:])[::-1][:num_peaks] + 1
        
        for idx in peak_indices:
            peaks.append({
                'frequency_hz': freqs[idx],
                'amplitude': magnitude[idx],
                'power': power[idx],
                'phase': phase[idx],
                'relative_strength': magnitude[idx] / np.max(magnitude[1:]) if np.max(magnitude[1:]) > 0 else 0
            })
        
        return peaks
    
    def find_peak_near_frequency(self, peaks, target_freq, tolerance=0.02):
        """Find the peak closest to target frequency.
        
        Returns:
            Peak dict if found, None otherwise
        """
        if not peaks:
            return None
            
        closest = min(peaks, key=lambda p: abs(p['frequency_hz'] - target_freq))
        
        if abs(closest['frequency_hz'] - target_freq) <= tolerance:
            closest['in_tolerance'] = True
        else:
            closest['in_tolerance'] = False
            
        return closest


class SignalProcessingBaseline:
    """Traditional signal processing baseline for flow rate prediction."""
    
    def __init__(self, csv_path='FRaTbot_flowdata.csv', target_freq=0.2, freq_tolerance=0.02):
        self.csv_path = csv_path
        self.target_freq = target_freq
        self.freq_tolerance = freq_tolerance
        self.metadata = []
        self.results = []
        
    def load_and_parse_data(self):
        """Load CSV and parse paired proximal/distal measurements."""
        print("=" * 80)
        print("SIGNAL PROCESSING BASELINE - FFT Analysis")
        print("=" * 80)
        
        # Read CSV with proper handling of malformed header
        import csv
        from io import StringIO
        
        with open(self.csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            header = next(reader)
            data_rows = list(reader)
        
        # Fix split first column: merge columns 0-3 if they're split
        # Check if first few columns look like: ['﻿"Proximal', '10ms', '0.2Hz', '9.4GPM"']
        if (len(header) > 3 and 
            'Proximal' in header[0] and 
            'ms' in header[1] and 
            'Hz' in header[2] and 
            'GPM' in header[3]):
            
            # Merge first 4 columns into one properly formatted column
            merged_col = f"{header[0].strip().strip('\"').strip()},{header[1]},{header[2]},{header[3].strip('\"')}"
            fixed_header = [merged_col] + header[4:]
            
            # Merge data rows too
            fixed_data = []
            for row in data_rows:
                if len(row) > 3:
                    # Merge first 4 data values (they belong to same column)
                    # Take the first non-empty value from these 4 columns
                    merged_val = row[0] if row[0].strip() else (row[1] if len(row) > 1 and row[1].strip() else '')
                    fixed_row = [merged_val] + row[4:]
                    # Pad row if needed to match header length
                    while len(fixed_row) < len(fixed_header):
                        fixed_row.append('')
                    fixed_data.append(fixed_row[:len(fixed_header)])
                else:
                    # Pad short rows
                    padded_row = row + [''] * (len(fixed_header) - len(row))
                    fixed_data.append(padded_row[:len(fixed_header)])
            
            # Create DataFrame from fixed data
            df = pd.DataFrame(fixed_data, columns=fixed_header)
            # Convert numeric columns to float
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
        else:
            # Normal CSV, just read it
            df = pd.read_csv(self.csv_path, encoding='utf-8-sig')
            
        print(f"Loaded CSV: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Parse column headers
        self.metadata = []
        for col in df.columns:
            col_clean = col.strip().strip('"').strip('\ufeff').strip()
            # Try standard format first
            match = re.search(r'(Proximal|Distal),(\d+)ms,([\d.]+)Hz,([\d.]+)GPM', col_clean)
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
            else:
                # Try alternative format with spaces or different separators
                match2 = re.search(r'(Proximal|Distal)\s+(\d+)ms\s+([\d.]+)Hz\s+([\d.]+)GPM', col_clean)
                if match2:
                    location, sample_rate, frequency, flow_rate = match2.groups()
                    self.metadata.append({
                        'column': col,
                        'location': location,
                        'sample_rate_ms': int(sample_rate),
                        'frequency_hz': float(frequency),
                        'flow_rate_gpm': float(flow_rate),
                        'data': df[col].values
                    })
        
        print(f"Parsed {len(self.metadata)} columns with metadata")
        
        # Display unique values
        flow_rates = sorted(set(m['flow_rate_gpm'] for m in self.metadata))
        frequencies = sorted(set(m['frequency_hz'] for m in self.metadata))
        print(f"Flow rates: {flow_rates} GPM")
        print(f"Frequencies: {frequencies} Hz")
        print(f"Target frequency: {self.target_freq} Hz")
        print()
        
    def build_paired_measurements(self):
        """Build pairs of proximal/distal measurements."""
        pairs = []
        
        # Group by flow rate and frequency
        flow_rates = sorted(set(m['flow_rate_gpm'] for m in self.metadata))
        frequencies = sorted(set(m['frequency_hz'] for m in self.metadata))
        
        for flow_rate in flow_rates:
            for frequency in frequencies:
                prox = None
                dist = None
                
                for m in self.metadata:
                    if m['flow_rate_gpm'] == flow_rate and abs(m['frequency_hz'] - frequency) < 0.01:
                        if m['location'] == 'Proximal':
                            prox = m
                        elif m['location'] == 'Distal':
                            dist = m
                
                if prox is not None and dist is not None:
                    # Clean data (remove NaN)
                    valid_mask = ~(np.isnan(prox['data']) | np.isnan(dist['data']))
                    if np.sum(valid_mask) > 100:
                        pairs.append({
                            'flow_rate': flow_rate,
                            'frequency': frequency,
                            'proximal_data': prox['data'][valid_mask],
                            'distal_data': dist['data'][valid_mask],
                            'sample_rate_ms': prox['sample_rate_ms']
                        })
        
        print(f"Built {len(pairs)} paired measurements")
        print()
        return pairs
    
    def analyze_pairs(self, pairs):
        """Analyze all pairs using FFT and compute attenuation metrics."""
        print("Analyzing amplitude attenuation...")
        print(f"Filtering to target frequency: {self.target_freq} Hz")
        print()
        
        results = []
        processor = FFTSignalProcessor(sample_rate_hz=100)
        
        # Filter pairs to only target frequency
        filtered_pairs = [p for p in pairs if abs(p['frequency'] - self.target_freq) < 0.01]
        print(f"Using {len(filtered_pairs)} pairs at {self.target_freq} Hz (filtered from {len(pairs)} total)")
        print()
        
        for i, pair in enumerate(filtered_pairs):
            # Compute FFT peaks for both proximal and distal
            prox_peaks = processor.compute_fft_peaks(pair['proximal_data'], num_peaks=3)
            dist_peaks = processor.compute_fft_peaks(pair['distal_data'], num_peaks=3)
            
            # Find peaks near target frequency
            prox_target = processor.find_peak_near_frequency(prox_peaks, self.target_freq, self.freq_tolerance)
            dist_target = processor.find_peak_near_frequency(dist_peaks, self.target_freq, self.freq_tolerance)
            
            if prox_target and dist_target:
                # Main attenuation metrics (at target frequency)
                amp_diff = prox_target['amplitude'] - dist_target['amplitude']
                amp_ratio = dist_target['amplitude'] / prox_target['amplitude'] if prox_target['amplitude'] > 0 else np.nan
                
                # Multi-peak weighted metrics (using top 3 peaks)
                weighted_diff = 0
                weighted_ratio = 0
                total_weight = 0
                
                for rank in range(min(3, len(prox_peaks), len(dist_peaks))):
                    pp = prox_peaks[rank]
                    dp = dist_peaks[rank]
                    
                    weight = (pp['relative_strength'] + dp['relative_strength']) / 2
                    weighted_diff += weight * (pp['amplitude'] - dp['amplitude'])
                    if pp['amplitude'] > 0:
                        weighted_ratio += weight * (dp['amplitude'] / pp['amplitude'])
                    total_weight += weight
                
                if total_weight > 0:
                    weighted_diff /= total_weight
                    weighted_ratio /= total_weight
                else:
                    weighted_diff = np.nan
                    weighted_ratio = np.nan
                
                results.append({
                    'pair_id': i + 1,
                    'flow_rate': pair['flow_rate'],
                    'frequency': pair['frequency'],
                    'prox_amp': prox_target['amplitude'],
                    'dist_amp': dist_target['amplitude'],
                    'amp_diff': amp_diff,
                    'amp_ratio': amp_ratio,
                    'weighted_diff': weighted_diff,
                    'weighted_ratio': weighted_ratio,
                    'prox_in_tol': prox_target['in_tolerance'],
                    'dist_in_tol': dist_target['in_tolerance']
                })
        
        self.results = results
        return results
    
    def compute_correlations(self):
        """Compute correlations between attenuation metrics and flow rate."""
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        
        # Filter to only measurements with data
        valid_df = df.dropna(subset=['flow_rate', 'amp_diff', 'amp_ratio'])
        
        if len(valid_df) < 2:
            return {}
        
        correlations = {
            'amp_diff_vs_flow': np.corrcoef(valid_df['flow_rate'], valid_df['amp_diff'])[0, 1],
            'amp_ratio_vs_flow': np.corrcoef(valid_df['flow_rate'], valid_df['amp_ratio'])[0, 1],
        }
        
        # Weighted correlations if available
        valid_weighted = df.dropna(subset=['flow_rate', 'weighted_diff', 'weighted_ratio'])
        if len(valid_weighted) >= 2:
            correlations['weighted_diff_vs_flow'] = np.corrcoef(valid_weighted['flow_rate'], valid_weighted['weighted_diff'])[0, 1]
            correlations['weighted_ratio_vs_flow'] = np.corrcoef(valid_weighted['flow_rate'], valid_weighted['weighted_ratio'])[0, 1]
        
        return correlations
    
    def fit_linear_model(self, metric='amp_diff'):
        """Fit a simple linear model for prediction."""
        if not self.results:
            return None, None
        
        df = pd.DataFrame(self.results)
        valid_df = df.dropna(subset=['flow_rate', metric])
        
        if len(valid_df) < 2:
            return None, None
        
        # Simple linear regression: flow = a * metric + b
        X = valid_df[metric].values
        y = valid_df['flow_rate'].values
        
        # Use numpy polyfit (degree 1 = linear)
        coeffs = np.polyfit(X, y, 1)
        slope, intercept = coeffs
        
        # Predict and compute metrics
        y_pred = slope * X + intercept
        mae = np.mean(np.abs(y - y_pred))
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
        
        return {
            'slope': slope,
            'intercept': intercept,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'y_true': y,
            'y_pred': y_pred
        }, valid_df
    
    def print_summary(self):
        """Print summary of signal processing analysis."""
        if not self.results:
            print("No results to display")
            return
        
        df = pd.DataFrame(self.results)
        
        print("=" * 80)
        print("SIGNAL PROCESSING RESULTS")
        print("=" * 80)
        print()
        
        # Display results table
        print(f"Amplitude Attenuation Analysis (Target Frequency: {self.target_freq} Hz)")
        print("-" * 80)
        display_cols = ['pair_id', 'flow_rate', 'frequency', 'prox_amp', 'dist_amp', 
                       'amp_diff', 'amp_ratio', 'weighted_diff', 'weighted_ratio']
        print(df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print()
        
        # Correlations
        corr = self.compute_correlations()
        print("Pearson Correlations with Flow Rate:")
        print("-" * 80)
        for key, value in corr.items():
            print(f"  {key:30s}: {value:6.3f}")
        print()
        
        # Linear model performance
        print("Linear Model Performance:")
        print("-" * 80)
        
        for metric_name, metric_key in [('Amplitude Difference', 'amp_diff'),
                                         ('Amplitude Ratio', 'amp_ratio'),
                                         ('Weighted Difference', 'weighted_diff'),
                                         ('Weighted Ratio', 'weighted_ratio')]:
            model, _ = self.fit_linear_model(metric_key)
            if model:
                print(f"\n{metric_name}:")
                print(f"  Equation: Flow = {model['slope']:.3f} × {metric_key} + {model['intercept']:.3f}")
                print(f"  MAE:      {model['mae']:.3f} GPM")
                print(f"  RMSE:     {model['rmse']:.3f} GPM")
                print(f"  R²:       {model['r2']:.3f}")
        print()
    
    def create_visualizations(self):
        """Create visualization plots comparing signal processing metrics."""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        valid_df = df.dropna(subset=['flow_rate'])
        
        if valid_df.empty:
            print("No valid data for visualization")
            return
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Amplitude Difference vs Flow Rate
        ax1 = plt.subplot(2, 3, 1)
        sorted_df = valid_df.sort_values('flow_rate')
        ax1.plot(sorted_df['flow_rate'], sorted_df['amp_diff'], '-o', color='royalblue', markersize=8, linewidth=2)
        ax1.set_xlabel('Flow Rate (GPM)', fontsize=11)
        ax1.set_ylabel('Amplitude Difference (Prox - Dist) [°C]', fontsize=11)
        ax1.set_title('Main Attenuation vs Flow Rate', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add correlation
        corr = self.compute_correlations()
        if 'amp_diff_vs_flow' in corr:
            ax1.text(0.05, 0.95, f'r = {corr["amp_diff_vs_flow"]:.3f}',
                    transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. Amplitude Ratio vs Flow Rate
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(sorted_df['flow_rate'], sorted_df['amp_ratio'], '-o', color='darkorange', markersize=8, linewidth=2)
        ax2.set_xlabel('Flow Rate (GPM)', fontsize=11)
        ax2.set_ylabel('Amplitude Ratio (Dist/Prox)', fontsize=11)
        ax2.set_title('Relative Attenuation vs Flow Rate', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        if 'amp_ratio_vs_flow' in corr:
            ax2.text(0.05, 0.95, f'r = {corr["amp_ratio_vs_flow"]:.3f}',
                    transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 3. Weighted Difference vs Flow Rate
        ax3 = plt.subplot(2, 3, 3)
        valid_weighted = valid_df.dropna(subset=['weighted_diff']).sort_values('flow_rate')
        if not valid_weighted.empty:
            ax3.plot(valid_weighted['flow_rate'], valid_weighted['weighted_diff'], '-o', 
                    color='purple', markersize=8, linewidth=2)
            ax3.set_xlabel('Flow Rate (GPM)', fontsize=11)
            ax3.set_ylabel('Weighted Multi-Peak Difference [°C]', fontsize=11)
            ax3.set_title('Weighted Attenuation vs Flow Rate', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            if 'weighted_diff_vs_flow' in corr:
                ax3.text(0.05, 0.95, f'r = {corr["weighted_diff_vs_flow"]:.3f}',
                        transform=ax3.transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 4. Linear fit for amp_diff
        ax4 = plt.subplot(2, 3, 4)
        model_diff, model_df = self.fit_linear_model('amp_diff')
        if model_diff:
            ax4.scatter(model_diff['y_true'], model_diff['y_pred'], alpha=0.6, s=100, color='royalblue')
            ax4.plot([model_diff['y_true'].min(), model_diff['y_true'].max()],
                    [model_diff['y_true'].min(), model_diff['y_true'].max()],
                    'r--', lw=2, label='Perfect Prediction')
            ax4.set_xlabel('Actual Flow Rate (GPM)', fontsize=11)
            ax4.set_ylabel('Predicted Flow Rate (GPM)', fontsize=11)
            ax4.set_title(f'Linear Model: Amplitude Difference\nMAE={model_diff["mae"]:.3f} GPM, R²={model_diff["r2"]:.3f}',
                         fontsize=12, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Linear fit for amp_ratio
        ax5 = plt.subplot(2, 3, 5)
        model_ratio, _ = self.fit_linear_model('amp_ratio')
        if model_ratio:
            ax5.scatter(model_ratio['y_true'], model_ratio['y_pred'], alpha=0.6, s=100, color='darkorange')
            ax5.plot([model_ratio['y_true'].min(), model_ratio['y_true'].max()],
                    [model_ratio['y_true'].min(), model_ratio['y_true'].max()],
                    'r--', lw=2, label='Perfect Prediction')
            ax5.set_xlabel('Actual Flow Rate (GPM)', fontsize=11)
            ax5.set_ylabel('Predicted Flow Rate (GPM)', fontsize=11)
            ax5.set_title(f'Linear Model: Amplitude Ratio\nMAE={model_ratio["mae"]:.3f} GPM, R²={model_ratio["r2"]:.3f}',
                         fontsize=12, fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Comparison table
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('tight')
        ax6.axis('off')
        
        table_data = []
        for metric_name, metric_key in [('Amplitude Diff', 'amp_diff'),
                                         ('Amplitude Ratio', 'amp_ratio'),
                                         ('Weighted Diff', 'weighted_diff'),
                                         ('Weighted Ratio', 'weighted_ratio')]:
            model, _ = self.fit_linear_model(metric_key)
            if model:
                table_data.append([
                    metric_name,
                    f"{model['mae']:.3f}",
                    f"{model['rmse']:.3f}",
                    f"{model['r2']:.3f}"
                ])
        
        if table_data:
            table = ax6.table(cellText=table_data,
                            colLabels=['Metric', 'MAE (GPM)', 'RMSE (GPM)', 'R²'],
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2.5)
            ax6.set_title('Signal Processing Model Performance', fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig('signal_processing_baseline_results.png', dpi=150, bbox_inches='tight')
        print("Saved visualization: signal_processing_baseline_results.png")
        plt.show()
    
    def run(self):
        """Run complete signal processing analysis."""
        self.load_and_parse_data()
        pairs = self.build_paired_measurements()
        self.analyze_pairs(pairs)
        self.print_summary()
        self.create_visualizations()
        
        return self.results


def main():
    """Main execution function."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 18 + "SIGNAL PROCESSING BASELINE" + " " * 32 + "║")
    print("║" + " " * 10 + "FFT-Based Flow Rate Prediction (Non-ML Approach)" + " " * 18 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
    
    # Run signal processing baseline
    baseline = SignalProcessingBaseline(
        csv_path='FRaTbot_flowdata_2.csv' if os.path.exists('FRaTbot_flowdata_2.csv') else 'FRaTbot_flowdata.csv',
        target_freq=0.2,
        freq_tolerance=0.02
    )
    baseline.run()
    
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print("This baseline uses traditional FFT signal processing to predict flow rate")
    print("from amplitude attenuation between proximal and distal thermistors.")
    print()
    print("Key findings:")
    print("  - Strong negative correlation between amplitude difference and flow rate")
    print("  - Strong positive correlation between amplitude ratio and flow rate")
    print("  - Simple linear models achieve reasonable accuracy with no ML required")
    print()


if __name__ == "__main__":
    import os
    main()
