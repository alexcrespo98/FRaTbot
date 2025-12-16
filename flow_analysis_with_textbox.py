#!/usr/bin/env python3
"""
Flow Analysis Script with Theoretical Temperature Amplitude Decay Predictions
and Interactive Text Box Input

This script processes preprocessed FFT results and overlays theoretical predictions
with an adjustable offset parameter via text box input.

Physical System:
- Pipe diameter: 0.75 inches (0.01905 m)
- Fluid: Water
- Distance between thermometers: 50 ft (15.24 m)
- Assumption: Adiabatic (no heat loss through pipe walls)
- Flow rates: Variable (extracted from data, typically 3-9 GPM)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
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
    
    # Empirical fitting parameter for attenuation
    'attenuation_scale_factor': 0.00053,  # Dimensionless scaling factor
    
    # Default adjustment offset
    'default_adjustment': 0.0,
    
    # Plotting
    'output_figure': 'flow_analysis_with_theory.png',
    'figure_dpi': 150,
}


# ============================================================================
# THEORETICAL CALCULATION FUNCTIONS
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
                                         scale_factor=1.0,
                                         adjustment=0.0):
    """Calculate theoretical amplitude ratio with adjustment offset."""
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
    
    # Apply adjustment (offset)
    ratio = ratio + adjustment
    ratio = np.clip(ratio, 0.0, 1.0)
    
    return ratio


def calculate_theoretical_amplitude_difference(flow_rate_gpm, frequency_hz,
                                               distance_m, pipe_diameter_m,
                                               thermal_diffusivity_m2s,
                                               proximal_amplitude_C,
                                               scale_factor=1.0,
                                               adjustment=0.0):
    """Calculate theoretical amplitude difference with adjustment."""
    ratio = calculate_theoretical_amplitude_ratio(
        flow_rate_gpm, frequency_hz, distance_m, 
        pipe_diameter_m, thermal_diffusivity_m2s, scale_factor, adjustment
    )
    difference = proximal_amplitude_C * (1.0 - ratio)
    return difference


# ============================================================================
# DATA LOADING
# ============================================================================

def load_preprocessed_data(config):
    """Load preprocessed FFT results from CSV."""
    csv_path = config['preprocessed_csv']
    
    if os.path.exists(csv_path):
        print(f"Loading preprocessed data: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
        
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
                print(f"  ERROR: Required column '{std_name}' not found.")
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
    
    # Fallback
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
# INTERACTIVE VISUALIZATION CLASS
# ============================================================================

class InteractiveFlowAnalysisWithTextBox:
    """Interactive visualization with text box for adjustment parameter."""
    
    def __init__(self, experimental_df, config):
        self.experimental_df = experimental_df.sort_values('flow_rate')
        self.config = config
        self.current_adjustment = config['default_adjustment']
        
        # Physical parameters
        self.pipe_diameter_m = config['pipe_diameter_inch'] * 0.0254
        self.distance_m = config['sensor_distance_ft'] * 0.3048
        self.alpha = config['water_thermal_diffusivity_m2s']
        self.frequency_hz = config['target_frequency_hz']
        self.scale_factor = config['attenuation_scale_factor']
        self.mean_prox_amp = self.experimental_df['prox_amp_C'].mean()
        
        self.setup_figure()
    
    def setup_figure(self):
        """Setup figure with 6 plots and text box at top."""
        self.fig = plt.figure(figsize=(16, 12))
        
        # Add text box at the very top
        textbox_ax = self.fig.add_axes([0.35, 0.96, 0.15, 0.025])
        self.textbox = TextBox(textbox_ax, 'Adjustment: ', 
                               initial=str(self.current_adjustment),
                               label_pad=0.01)
        self.textbox.on_submit(self.update_adjustment)
        
        # Add instruction text
        instruction_ax = self.fig.add_axes([0.52, 0.96, 0.3, 0.025])
        instruction_ax.axis('off')
        instruction_ax.text(0, 0.5, '← Enter value and press Enter to update plots',
                          fontsize=10, verticalalignment='center')
        
        # Create 6 subplots (original layout)
        self.ax1 = plt.subplot(2, 3, 1)
        self.ax2 = plt.subplot(2, 3, 2)
        self.ax3 = plt.subplot(2, 3, 3)
        self.ax4 = plt.subplot(2, 3, 4)
        self.ax5 = plt.subplot(2, 3, 5)
        self.ax6 = plt.subplot(2, 3, 6)
        
        # Initial plot
        self.update_plots()
        
        plt.suptitle('Flow Analysis: Experimental vs Theoretical (Adjustable)',
                     fontsize=14, fontweight='bold', y=0.94)
    
    def update_adjustment(self, text):
        """Callback when text box value changes."""
        try:
            self.current_adjustment = float(text)
            self.update_plots()
        except ValueError:
            print(f"Invalid input: '{text}'. Please enter a number.")
            self.textbox.set_val(str(self.current_adjustment))
    
    def generate_theoretical_curves(self):
        """Generate theoretical predictions with current adjustment."""
        flow_min = self.experimental_df['flow_rate'].min() - 0.5
        flow_max = self.experimental_df['flow_rate'].max() + 0.5
        flow_rates = np.linspace(flow_min, flow_max, 100)
        
        ratios = []
        diffs = []
        for flow_rate in flow_rates:
            ratio = calculate_theoretical_amplitude_ratio(
                flow_rate, self.frequency_hz, self.distance_m,
                self.pipe_diameter_m, self.alpha, self.scale_factor,
                self.current_adjustment
            )
            ratios.append(ratio)
            
            diff = calculate_theoretical_amplitude_difference(
                flow_rate, self.frequency_hz, self.distance_m,
                self.pipe_diameter_m, self.alpha, self.mean_prox_amp,
                self.scale_factor, self.current_adjustment
            )
            diffs.append(diff)
        
        return flow_rates, np.array(ratios), np.array(diffs)
    
    def update_plots(self):
        """Update all 6 plots with current adjustment."""
        exp_sorted = self.experimental_df
        theo_flow, theo_ratio, theo_diff = self.generate_theoretical_curves()
        
        # Clear all axes
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6]:
            ax.clear()
        
        # Plot 1: Main Amplitude Difference vs Flow Rate
        self.ax1.plot(exp_sorted['flow_rate'], exp_sorted['amp_diff_C'], 
                     '-o', color='royalblue', markersize=8, linewidth=2, 
                     label='Experimental')
        self.ax1.plot(theo_flow, theo_diff,
                     '--', color='red', linewidth=2.5,
                     label='Theoretical')
        self.ax1.set_xlabel('Flow Rate (GPM)', fontsize=11)
        self.ax1.set_ylabel('Amplitude Difference [°C]', fontsize=11)
        self.ax1.set_title('Main Attenuation vs Flow Rate', fontsize=12, fontweight='bold')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend(fontsize=10)
        
        corr_diff = np.corrcoef(exp_sorted['flow_rate'], exp_sorted['amp_diff_C'])[0, 1]
        self.ax1.text(0.05, 0.95, f'r = {corr_diff:.3f}',
                transform=self.ax1.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 2: Main Amplitude Ratio vs Flow Rate
        self.ax2.plot(exp_sorted['flow_rate'], exp_sorted['amp_ratio_dist_over_prox'],
                     '-o', color='darkorange', markersize=8, linewidth=2,
                     label='Experimental')
        self.ax2.plot(theo_flow, theo_ratio,
                     '--', color='red', linewidth=2.5,
                     label='Theoretical')
        self.ax2.set_xlabel('Flow Rate (GPM)', fontsize=11)
        self.ax2.set_ylabel('Amplitude Ratio (Dist/Prox)', fontsize=11)
        self.ax2.set_title('Relative Attenuation vs Flow Rate', fontsize=12, fontweight='bold')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend(fontsize=10)
        
        corr_ratio = np.corrcoef(exp_sorted['flow_rate'], exp_sorted['amp_ratio_dist_over_prox'])[0, 1]
        self.ax2.text(0.05, 0.95, f'r = {corr_ratio:.3f}',
                transform=self.ax2.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 3: Weighted Multi-Peak Difference
        self.ax3.plot(exp_sorted['flow_rate'], exp_sorted['weighted_multi_peak_diff_C'],
                     '-o', color='purple', markersize=8, linewidth=2,
                     label='Experimental')
        self.ax3.plot(theo_flow, theo_diff,
                     '--', color='red', linewidth=2.5, alpha=0.7,
                     label='Theoretical')
        self.ax3.set_xlabel('Flow Rate (GPM)', fontsize=11)
        self.ax3.set_ylabel('Weighted Difference [°C]', fontsize=11)
        self.ax3.set_title('Weighted Attenuation vs Flow Rate', fontsize=12, fontweight='bold')
        self.ax3.grid(True, alpha=0.3)
        self.ax3.legend(fontsize=10)
        
        # Plot 4: Weighted Multi-Peak Ratio
        self.ax4.plot(exp_sorted['flow_rate'], exp_sorted['weighted_multi_peak_ratio'],
                     '-o', color='green', markersize=8, linewidth=2,
                     label='Experimental')
        self.ax4.plot(theo_flow, theo_ratio,
                     '--', color='red', linewidth=2.5,
                     label='Theoretical')
        self.ax4.set_xlabel('Flow Rate (GPM)', fontsize=11)
        self.ax4.set_ylabel('Weighted Ratio', fontsize=11)
        self.ax4.set_title('Weighted Relative Attenuation', fontsize=12, fontweight='bold')
        self.ax4.grid(True, alpha=0.3)
        self.ax4.legend(fontsize=10)
        
        # Plot 5: Proximal and Distal Amplitudes
        self.ax5.plot(exp_sorted['flow_rate'], exp_sorted['prox_amp_C'],
                     '-o', color='blue', markersize=8, linewidth=2, label='Proximal')
        self.ax5.plot(exp_sorted['flow_rate'], exp_sorted['dist_amp_C'],
                     '-s', color='red', markersize=8, linewidth=2, label='Distal')
        self.ax5.set_xlabel('Flow Rate (GPM)', fontsize=11)
        self.ax5.set_ylabel('Amplitude [°C]', fontsize=11)
        self.ax5.set_title('Proximal and Distal Amplitudes', fontsize=12, fontweight='bold')
        self.ax5.grid(True, alpha=0.3)
        self.ax5.legend(fontsize=10)
        
        # Plot 6: Physical Parameters
        self.ax6.axis('off')
        info_text = (
            f'PHYSICAL PARAMETERS\n'
            f'───────────────────────\n'
            f'Pipe: {self.config["pipe_diameter_inch"]:.2f}" ({self.pipe_diameter_m:.5f} m)\n'
            f'Distance: {self.config["sensor_distance_ft"]:.0f} ft ({self.distance_m:.2f} m)\n'
            f'Thermal diffusivity: {self.alpha:.2e} m²/s\n'
            f'Frequency: {self.frequency_hz:.3f} Hz\n'
            f'Scale factor: {self.scale_factor:.5f}\n'
            f'\n'
            f'CURRENT ADJUSTMENT\n'
            f'───────────────────────\n'
            f'Offset: {self.current_adjustment:.6f}\n'
            f'\n'
            f'THEORETICAL MODEL\n'
            f'───────────────────────\n'
            f'A(x)/A₀ = exp(-β·x) + offset\n'
            f'\n'
            f'where β = k·(ω/v)·√(R²/α)\n'
            f'\n'
            f'CORRELATIONS\n'
            f'───────────────────────\n'
            f'Ratio vs flow: r = {corr_ratio:.3f}\n'
            f'Diff vs flow: r = {corr_diff:.3f}\n'
            f'\n'
            f'Enter adjustment value in\n'
            f'text box above to shift\n'
            f'theoretical curves.'
        )
        
        self.ax6.text(0.05, 0.95, info_text,
                     transform=self.ax6.transAxes,
                     fontsize=9, verticalalignment='top',
                     family='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        self.fig.canvas.draw_idle()
    
    def show(self):
        """Display the interactive plot."""
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "FLOW ANALYSIS WITH TEXT BOX ADJUSTMENT" + " " * 24 + "║")
    print("║" + " " * 10 + "Enter adjustment value to shift theoretical curves" + " " * 17 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
    
    # Load experimental data
    print("=" * 80)
    print("LOADING EXPERIMENTAL DATA")
    print("=" * 80)
    experimental_df = load_preprocessed_data(CONFIG)
    print(f"\nLoaded {len(experimental_df)} flow rate pairs")
    print()
    
    # Create interactive visualization
    print("=" * 80)
    print("CREATING INTERACTIVE VISUALIZATION")
    print("=" * 80)
    print("\nControls:")
    print("  • Text Box: Enter adjustment offset and press Enter")
    print("  • Default: 0.0 (no adjustment)")
    print("  • Positive values shift curves up, negative shift down")
    print("\nClose the plot window when finished.")
    print()
    
    app = InteractiveFlowAnalysisWithTextBox(experimental_df, CONFIG)
    app.show()
    
    print("\n" + "=" * 80)
    print("SESSION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
