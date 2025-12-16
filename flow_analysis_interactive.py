#!/usr/bin/env python3
"""
Interactive Flow Analysis Script with Theoretical Temperature Amplitude Decay Predictions

This script provides an interactive visualization with:
1. A slider to adjust heat loss/attenuation scale factor
2. Checkboxes to include/exclude specific flow rate pairs

Physical System:
- Pipe diameter: 0.75 inches (0.01905 m)
- Fluid: Water
- Distance between thermometers: 50 ft (15.24 m)
- Flow rates: Variable (extracted from data, typically 3-9 GPM)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
import warnings
import os

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Data files
    'preprocessed_csv': 'fft_results_preprocessed/FRaTbot_flowdata_2_preprocessed_fft_peaks_20251212_111202.csv',
    'fallback_csv': 'FRaTbot_flowdata_2.csv',
    
    # Target frequency
    'target_frequency_hz': 0.200,
    
    # Physical parameters for theoretical model
    'pipe_diameter_inch': 0.75,
    'sensor_distance_ft': 50.0,
    'water_thermal_diffusivity_m2s': 1.47e-7,  # at 20°C
    
    # Initial attenuation scale factor
    'initial_scale_factor': 0.00053,
    
    # Scale factor range for slider
    'scale_factor_min': 0.0001,
    'scale_factor_max': 0.002,
    
    # Plotting
    'output_figure': 'flow_analysis_interactive.png',
    'figure_dpi': 150,
}


# ============================================================================
# THEORETICAL CALCULATION FUNCTIONS (from original script)
# ============================================================================

def calculate_peclet_number(flow_rate_gpm, pipe_diameter_m, thermal_diffusivity_m2s):
    """Calculate Péclet number from flow rate."""
    flow_rate_m3s = flow_rate_gpm * 6.309e-5
    pipe_area_m2 = np.pi * (pipe_diameter_m / 2.0) ** 2
    velocity_ms = flow_rate_m3s / pipe_area_m2
    peclet = (velocity_ms * pipe_diameter_m) / thermal_diffusivity_m2s
    return peclet


def calculate_theoretical_amplitude_ratio(flow_rate_gpm, frequency_hz, 
                                         distance_m, pipe_diameter_m, 
                                         thermal_diffusivity_m2s,
                                         scale_factor=1.0):
    """Calculate theoretical amplitude ratio based on phenomenological model."""
    omega = 2.0 * np.pi * frequency_hz
    alpha = thermal_diffusivity_m2s
    
    flow_rate_m3s = flow_rate_gpm * 6.309e-5
    pipe_area_m2 = np.pi * (pipe_diameter_m / 2.0) ** 2
    velocity_ms = flow_rate_m3s / pipe_area_m2
    radius_m = pipe_diameter_m / 2.0
    
    beta_base = (omega / velocity_ms) * np.sqrt(radius_m**2 / alpha)
    beta = scale_factor * beta_base
    exponent = -distance_m * beta
    ratio = np.exp(exponent)
    ratio = np.clip(ratio, 0.0, 1.0)
    
    return ratio


def calculate_theoretical_amplitude_difference(flow_rate_gpm, frequency_hz,
                                               distance_m, pipe_diameter_m,
                                               thermal_diffusivity_m2s,
                                               proximal_amplitude_C,
                                               scale_factor=1.0):
    """Calculate theoretical amplitude difference given proximal amplitude."""
    ratio = calculate_theoretical_amplitude_ratio(
        flow_rate_gpm, frequency_hz, distance_m, 
        pipe_diameter_m, thermal_diffusivity_m2s, scale_factor
    )
    difference = proximal_amplitude_C * (1.0 - ratio)
    return difference


# ============================================================================
# DATA LOADING (simplified from original)
# ============================================================================

def load_preprocessed_data(config):
    """Load preprocessed FFT results from CSV or use embedded data."""
    csv_path = config['preprocessed_csv']
    
    if os.path.exists(csv_path):
        print(f"Loading preprocessed data: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Define required columns and alternatives
        required_columns = {
            'flow_rate': ['flow_rate', 'flow_rate_gpm', 'flow_gpm', 'gpm'],
            'prox_amp_C': ['prox_amp_C', 'prox_amp', 'proximal_amp', 'amp_prox'],
            'dist_amp_C': ['dist_amp_C', 'dist_amp', 'distal_amp', 'amp_dist'],
            'amp_diff_C': ['amp_diff_C', 'amp_diff', 'amplitude_diff', 'diff'],
            'amp_ratio_dist_over_prox': ['amp_ratio_dist_over_prox', 'amp_ratio', 'ratio', 'amplitude_ratio'],
        }
        
        optional_columns = {
            'weighted_multi_peak_diff_C': ['weighted_multi_peak_diff_C', 'weighted_diff', 'multi_peak_diff'],
            'weighted_multi_peak_ratio': ['weighted_multi_peak_ratio', 'weighted_ratio', 'multi_peak_ratio'],
        }
        
        # Map columns
        column_mapping = {}
        alternatives_lower = {std: [a.lower() for a in alts] for std, alts in required_columns.items()}
        
        for std_name, alternatives in required_columns.items():
            found = False
            for alt_name in alternatives:
                if alt_name in df.columns:
                    column_mapping[alt_name] = std_name
                    found = True
                    break
            
            if not found:
                for col in df.columns:
                    if col.lower() in alternatives_lower[std_name]:
                        column_mapping[col] = std_name
                        found = True
                        break
            
            if not found:
                print(f"  ERROR: Required column '{std_name}' not found. Using embedded data.")
                break
        else:
            df = df.rename(columns=column_mapping)
            for std_name, alternatives in optional_columns.items():
                for alt_name in alternatives:
                    if alt_name in df.columns:
                        df = df.rename(columns={alt_name: std_name})
                        break
            print(f"  Successfully mapped columns")
            return df
    
    # Fallback to embedded data
    print("Using embedded experimental data...")
    data = {
        'pair_id': ['pair_1', 'pair_2', 'pair_3', 'pair_4', 'pair_5', 'pair_6', 'pair_7'],
        'flow_rate': [9.4, 8.6, 7.8, 6.7, 5.4, 3.9, 3.2],
        'prox_amp_C': [0.221, 0.432, 0.384, 0.456, 0.376, 0.417, 0.447],
        'dist_amp_C': [0.196, 0.251, 0.221, 0.151, 0.066, 0.051, 0.137],
        'amp_diff_C': [0.025, 0.181, 0.163, 0.306, 0.310, 0.367, 0.310],
        'amp_ratio_dist_over_prox': [0.885, 0.580, 0.577, 0.330, 0.176, 0.122, 0.308],
        'weighted_multi_peak_diff_C': [0.007, 0.130, 0.123, 0.190, 0.180, 0.224, 0.224],
        'weighted_multi_peak_ratio': [0.995, 0.641, 0.578, 0.488, 0.261, 0.319, 0.434],
    }
    return pd.DataFrame(data)


# ============================================================================
# INTERACTIVE VISUALIZATION
# ============================================================================

class InteractiveFlowAnalysis:
    """Interactive visualization with slider and checkboxes."""
    
    def __init__(self, experimental_df, config):
        self.experimental_df = experimental_df.sort_values('flow_rate').reset_index(drop=True)
        self.config = config
        self.n_pairs = len(self.experimental_df)
        
        # Initialize included pairs (all True initially)
        self.included_pairs = [True] * self.n_pairs
        
        # Physical parameters
        self.pipe_diameter_m = config['pipe_diameter_inch'] * 0.0254
        self.distance_m = config['sensor_distance_ft'] * 0.3048
        self.alpha = config['water_thermal_diffusivity_m2s']
        self.frequency_hz = config['target_frequency_hz']
        self.mean_prox_amp = self.experimental_df['prox_amp_C'].mean()
        
        # Initialize figure
        self.setup_figure()
        
    def setup_figure(self):
        """Setup the interactive figure with plots, slider, and checkboxes."""
        # Create figure with extra space for controls
        self.fig = plt.figure(figsize=(18, 10))
        
        # Main plot area (ratio plot - most important)
        self.ax_main = plt.subplot2grid((10, 3), (2, 0), colspan=2, rowspan=4)
        
        # Secondary plots
        self.ax_diff = plt.subplot2grid((10, 3), (7, 0), colspan=2, rowspan=3)
        
        # Info panel
        self.ax_info = plt.subplot2grid((10, 3), (2, 2), rowspan=8)
        self.ax_info.axis('off')
        
        # Slider area (top)
        ax_slider = plt.subplot2grid((10, 3), (0, 0), colspan=3)
        ax_slider.axis('off')
        
        # Create slider for attenuation scale factor
        slider_ax = self.fig.add_axes([0.15, 0.94, 0.7, 0.02])
        self.slider = Slider(
            slider_ax, 
            'Heat Loss / Attenuation Factor',
            self.config['scale_factor_min'],
            self.config['scale_factor_max'],
            valinit=self.config['initial_scale_factor'],
            valstep=0.00001,
            color='lightcoral'
        )
        self.slider.on_changed(self.update_plots)
        
        # Create checkboxes for data point selection
        checkbox_ax = self.fig.add_axes([0.05, 0.25, 0.08, 0.35])
        
        # Create labels for checkboxes
        labels = []
        for i, row in self.experimental_df.iterrows():
            labels.append(f"{row['flow_rate']:.1f} GPM")
        
        self.checkboxes = CheckButtons(checkbox_ax, labels, self.included_pairs)
        self.checkboxes.on_clicked(self.toggle_pair)
        
        # Initial plot
        self.update_plots(self.config['initial_scale_factor'])
        
        plt.suptitle('Interactive Flow Analysis: Experimental vs Theoretical Temperature Amplitude Decay',
                     fontsize=14, fontweight='bold', y=0.98)
        
    def toggle_pair(self, label):
        """Toggle inclusion of a specific flow rate pair."""
        # Find which checkbox was clicked
        for i, row in self.experimental_df.iterrows():
            if f"{row['flow_rate']:.1f} GPM" == label:
                self.included_pairs[i] = not self.included_pairs[i]
                break
        
        # Update plots
        self.update_plots(self.slider.val)
        
    def generate_theoretical_curves(self, scale_factor):
        """Generate theoretical predictions with given scale factor."""
        flow_min = self.experimental_df['flow_rate'].min() - 0.5
        flow_max = self.experimental_df['flow_rate'].max() + 0.5
        flow_rates = np.linspace(flow_min, flow_max, 100)
        
        ratios = []
        diffs = []
        for flow_rate in flow_rates:
            ratio = calculate_theoretical_amplitude_ratio(
                flow_rate, self.frequency_hz, self.distance_m, 
                self.pipe_diameter_m, self.alpha, scale_factor
            )
            ratios.append(ratio)
            
            diff = calculate_theoretical_amplitude_difference(
                flow_rate, self.frequency_hz, self.distance_m,
                self.pipe_diameter_m, self.alpha, self.mean_prox_amp, scale_factor
            )
            diffs.append(diff)
        
        return flow_rates, np.array(ratios), np.array(diffs)
    
    def update_plots(self, scale_factor):
        """Update all plots with new scale factor and included pairs."""
        # Get included data
        included_mask = np.array(self.included_pairs)
        exp_included = self.experimental_df[included_mask]
        
        if len(exp_included) == 0:
            return  # No data to plot
        
        # Generate theoretical curves
        theo_flow, theo_ratio, theo_diff = self.generate_theoretical_curves(scale_factor)
        
        # Clear axes
        self.ax_main.clear()
        self.ax_diff.clear()
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        # Plot 1: Amplitude Ratio (main plot)
        self.ax_main.plot(exp_included['flow_rate'], exp_included['amp_ratio_dist_over_prox'],
                         'o', color='darkorange', markersize=10, linewidth=2.5,
                         label='Experimental', zorder=3)
        self.ax_main.plot(theo_flow, theo_ratio,
                         '--', color='red', linewidth=3,
                         label='Theoretical', zorder=2)
        self.ax_main.set_xlabel('Flow Rate (GPM)', fontsize=12, fontweight='bold')
        self.ax_main.set_ylabel('Amplitude Ratio (Dist/Prox)', fontsize=12, fontweight='bold')
        self.ax_main.set_title('Temperature Amplitude Ratio vs Flow Rate', fontsize=13, fontweight='bold')
        self.ax_main.grid(True, alpha=0.3, linewidth=1.5)
        self.ax_main.legend(fontsize=11, loc='best')
        
        # Calculate and display correlation
        if len(exp_included) >= 2:
            corr = np.corrcoef(exp_included['flow_rate'], exp_included['amp_ratio_dist_over_prox'])[0, 1]
            self.ax_main.text(0.05, 0.95, f'Correlation: r = {corr:.3f}',
                            transform=self.ax_main.transAxes, fontsize=11,
                            verticalalignment='top', bbox=dict(boxstyle='round', 
                            facecolor='wheat', alpha=0.7))
        
        # Plot 2: Amplitude Difference
        self.ax_diff.plot(exp_included['flow_rate'], exp_included['amp_diff_C'],
                         'o', color='royalblue', markersize=10, linewidth=2.5,
                         label='Experimental', zorder=3)
        self.ax_diff.plot(theo_flow, theo_diff,
                         '--', color='red', linewidth=3,
                         label='Theoretical', zorder=2)
        self.ax_diff.set_xlabel('Flow Rate (GPM)', fontsize=12, fontweight='bold')
        self.ax_diff.set_ylabel('Amplitude Difference (Prox - Dist) [°C]', fontsize=12, fontweight='bold')
        self.ax_diff.set_title('Temperature Amplitude Attenuation vs Flow Rate', fontsize=13, fontweight='bold')
        self.ax_diff.grid(True, alpha=0.3, linewidth=1.5)
        self.ax_diff.legend(fontsize=11, loc='best')
        
        # Info panel
        info_text = (
            'SYSTEM PARAMETERS\n'
            '─────────────────────────\n'
            f'Pipe: {self.config["pipe_diameter_inch"]:.2f}" diameter\n'
            f'Distance: {self.config["sensor_distance_ft"]:.0f} ft\n'
            f'Frequency: {self.frequency_hz:.3f} Hz\n'
            f'Fluid: Water (α={self.alpha:.2e} m²/s)\n'
            '\n'
            'CURRENT SETTINGS\n'
            '─────────────────────────\n'
            f'Attenuation Factor: {scale_factor:.5f}\n'
            f'Data points shown: {len(exp_included)}/{self.n_pairs}\n'
            '\n'
            'THEORETICAL MODEL\n'
            '─────────────────────────\n'
            'Phenomenological decay:\n'
            '  A(x)/A₀ = exp(-β·x)\n'
            '\n'
            'where:\n'
            f'  β = k·(ω/v)·√(R²/α)\n'
            f'  k = {scale_factor:.5f}\n'
            '\n'
            'The attenuation factor accounts\n'
            'for heat loss, non-ideal flow,\n'
            'and other real-world effects.\n'
            '\n'
            'CONTROLS\n'
            '─────────────────────────\n'
            '• Drag slider to adjust model\n'
            '• Uncheck boxes to exclude\n'
            '  outliers or specific points\n'
        )
        
        self.ax_info.text(0.05, 0.95, info_text,
                         transform=self.ax_info.transAxes,
                         fontsize=9, verticalalignment='top',
                         family='monospace',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        self.fig.canvas.draw_idle()
    
    def show(self):
        """Display the interactive plot."""
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "INTERACTIVE FLOW ANALYSIS WITH THEORY" + " " * 26 + "║")
    print("║" + " " * 10 + "Adjust heat loss and toggle data points interactively" + " " * 13 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
    
    # Load experimental data
    print("=" * 80)
    print("LOADING EXPERIMENTAL DATA")
    print("=" * 80)
    experimental_df = load_preprocessed_data(CONFIG)
    print(f"\nLoaded {len(experimental_df)} flow rate pairs")
    print(f"Flow rates: {sorted(experimental_df['flow_rate'].values)} GPM")
    print()
    
    # Create interactive visualization
    print("=" * 80)
    print("CREATING INTERACTIVE VISUALIZATION")
    print("=" * 80)
    print("\nControls:")
    print("  • Slider: Adjust attenuation/heat loss factor to fit theory to data")
    print("  • Checkboxes: Toggle individual flow rate pairs on/off")
    print("\nClose the plot window when finished.")
    print()
    
    interactive_app = InteractiveFlowAnalysis(experimental_df, CONFIG)
    interactive_app.show()
    
    print("\n" + "=" * 80)
    print("INTERACTIVE SESSION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
