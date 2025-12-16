#!/usr/bin/env python3
"""
Flow Analysis Script for Jupyter Notebook with Interactive Controls

This version uses ipywidgets instead of matplotlib widgets for better
compatibility with Jupyter Notebook environments.

Usage in Jupyter:
    %matplotlib inline
    from flow_analysis_jupyter import create_interactive_analysis
    create_interactive_analysis()

Physical System:
- Pipe diameter: 0.75 inches (0.01905 m)
- Fluid: Water
- Distance between thermometers: 50 ft (15.24 m)
- Flow rates: Variable (extracted from data, typically 3-9 GPM)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os

# Import ipywidgets for Jupyter compatibility
try:
    from ipywidgets import FloatText, Checkbox, VBox, HBox, Label, Output, interact, interactive
    import ipywidgets as widgets
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False
    print("Warning: ipywidgets not available. Install with: pip install ipywidgets")

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'preprocessed_csv': 'fft_results_preprocessed/FRaTbot_flowdata_2_preprocessed_fft_peaks_20251212_111202.csv',
    'fallback_csv': 'FRaTbot_flowdata_2.csv',
    'target_frequency_hz': 0.200,
    'pipe_diameter_inch': 0.75,
    'sensor_distance_ft': 50.0,
    'water_thermal_diffusivity_m2s': 1.47e-7,
    'attenuation_scale_factor': 0.00053,
    'default_adjustment': 0.0,
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
        print(f"Loading: {csv_path}")
        df = pd.read_csv(csv_path)
        
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
                break
        else:
            df = df.rename(columns=column_mapping)
            for std_name, alternatives in optional_columns.items():
                for alt_name in alternatives:
                    if alt_name in df.columns:
                        df = df.rename(columns={alt_name: std_name})
                        break
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
# INTERACTIVE JUPYTER VISUALIZATION
# ============================================================================

def create_interactive_analysis():
    """
    Create an interactive analysis interface for Jupyter Notebook.
    
    This function creates ipywidgets controls and a matplotlib figure
    that work properly in Jupyter environments.
    """
    
    if not JUPYTER_AVAILABLE:
        print("ERROR: ipywidgets is required for Jupyter compatibility.")
        print("Install with: pip install ipywidgets")
        print("Then restart your Jupyter kernel.")
        return
    
    # Load data
    experimental_df = load_preprocessed_data(CONFIG)
    experimental_df = experimental_df.sort_values('flow_rate').reset_index(drop=True)
    n_pairs = len(experimental_df)
    
    # Physical parameters
    pipe_diameter_m = CONFIG['pipe_diameter_inch'] * 0.0254
    distance_m = CONFIG['sensor_distance_ft'] * 0.3048
    alpha = CONFIG['water_thermal_diffusivity_m2s']
    frequency_hz = CONFIG['target_frequency_hz']
    scale_factor = CONFIG['attenuation_scale_factor']
    
    # Create widgets
    adjustment_widget = FloatText(
        value=CONFIG['default_adjustment'],
        description='Adjustment:',
        disabled=False,
        style={'description_width': '100px'},
        layout=widgets.Layout(width='250px')
    )
    
    # Create checkboxes for each data point
    checkbox_widgets = []
    for i, row in experimental_df.iterrows():
        cb = Checkbox(
            value=True,
            description=f"{row['flow_rate']:.1f} GPM",
            disabled=False,
            indent=False,
            layout=widgets.Layout(width='120px')
        )
        checkbox_widgets.append(cb)
    
    # Output widget for the plot
    output = Output()
    
    def update_plot(*args):
        """Update the plot based on current widget values."""
        with output:
            output.clear_output(wait=True)
            
            # Get current adjustment
            adjustment = adjustment_widget.value
            
            # Get included data points
            included_mask = np.array([cb.value for cb in checkbox_widgets])
            exp_included = experimental_df[included_mask]
            
            if len(exp_included) == 0:
                exp_included = experimental_df
            
            # Calculate mean proximal amplitude
            mean_prox_amp = exp_included['prox_amp_C'].mean()
            
            # Generate theoretical curves
            flow_min = experimental_df['flow_rate'].min() - 0.5
            flow_max = experimental_df['flow_rate'].max() + 0.5
            theo_flow = np.linspace(flow_min, flow_max, 100)
            
            theo_ratios = []
            theo_diffs = []
            for flow_rate in theo_flow:
                ratio = calculate_theoretical_amplitude_ratio(
                    flow_rate, frequency_hz, distance_m, pipe_diameter_m, 
                    alpha, scale_factor, adjustment
                )
                theo_ratios.append(ratio)
                
                diff = calculate_theoretical_amplitude_difference(
                    flow_rate, frequency_hz, distance_m, pipe_diameter_m,
                    alpha, mean_prox_amp, scale_factor, adjustment
                )
                theo_diffs.append(diff)
            
            theo_ratios = np.array(theo_ratios)
            theo_diffs = np.array(theo_diffs)
            
            # Create figure
            fig = plt.figure(figsize=(16, 10))
            
            # Plot 1: Main Amplitude Difference
            ax1 = plt.subplot(2, 3, 1)
            ax1.plot(exp_included['flow_rate'], exp_included['amp_diff_C'], 
                     '-o', color='royalblue', markersize=8, linewidth=2, label='Experimental')
            ax1.plot(theo_flow, theo_diffs, '--', color='red', linewidth=2.5, label='Theoretical')
            ax1.set_xlabel('Flow Rate (GPM)', fontsize=11)
            ax1.set_ylabel('Amplitude Difference [°C]', fontsize=11)
            ax1.set_title('Main Attenuation vs Flow Rate', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=10)
            if len(exp_included) >= 2:
                corr = np.corrcoef(exp_included['flow_rate'], exp_included['amp_diff_C'])[0, 1]
                ax1.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax1.transAxes, 
                        fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Plot 2: Main Amplitude Ratio
            ax2 = plt.subplot(2, 3, 2)
            ax2.plot(exp_included['flow_rate'], exp_included['amp_ratio_dist_over_prox'],
                     '-o', color='darkorange', markersize=8, linewidth=2, label='Experimental')
            ax2.plot(theo_flow, theo_ratios, '--', color='red', linewidth=2.5, label='Theoretical')
            ax2.set_xlabel('Flow Rate (GPM)', fontsize=11)
            ax2.set_ylabel('Amplitude Ratio (Dist/Prox)', fontsize=11)
            ax2.set_title('Relative Attenuation vs Flow Rate', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=10)
            if len(exp_included) >= 2:
                corr = np.corrcoef(exp_included['flow_rate'], exp_included['amp_ratio_dist_over_prox'])[0, 1]
                ax2.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax2.transAxes,
                        fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Plot 3: Weighted Multi-Peak Difference
            ax3 = plt.subplot(2, 3, 3)
            ax3.plot(exp_included['flow_rate'], exp_included['weighted_multi_peak_diff_C'],
                     '-o', color='purple', markersize=8, linewidth=2, label='Experimental')
            ax3.plot(theo_flow, theo_diffs, '--', color='red', linewidth=2.5, alpha=0.7, label='Theoretical')
            ax3.set_xlabel('Flow Rate (GPM)', fontsize=11)
            ax3.set_ylabel('Weighted Difference [°C]', fontsize=11)
            ax3.set_title('Weighted Attenuation vs Flow Rate', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.legend(fontsize=10)
            
            # Plot 4: Weighted Multi-Peak Ratio
            ax4 = plt.subplot(2, 3, 4)
            ax4.plot(exp_included['flow_rate'], exp_included['weighted_multi_peak_ratio'],
                     '-o', color='green', markersize=8, linewidth=2, label='Experimental')
            ax4.plot(theo_flow, theo_ratios, '--', color='red', linewidth=2.5, label='Theoretical')
            ax4.set_xlabel('Flow Rate (GPM)', fontsize=11)
            ax4.set_ylabel('Weighted Ratio', fontsize=11)
            ax4.set_title('Weighted Relative Attenuation', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.legend(fontsize=10)
            
            # Plot 5: Proximal and Distal Amplitudes
            ax5 = plt.subplot(2, 3, 5)
            ax5.plot(exp_included['flow_rate'], exp_included['prox_amp_C'],
                     '-o', color='blue', markersize=8, linewidth=2, label='Proximal')
            ax5.plot(exp_included['flow_rate'], exp_included['dist_amp_C'],
                     '-s', color='red', markersize=8, linewidth=2, label='Distal')
            ax5.set_xlabel('Flow Rate (GPM)', fontsize=11)
            ax5.set_ylabel('Amplitude [°C]', fontsize=11)
            ax5.set_title('Proximal and Distal Amplitudes', fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3)
            ax5.legend(fontsize=10)
            
            # Plot 6: Info Panel
            ax6 = plt.subplot(2, 3, 6)
            ax6.axis('off')
            
            n_included = len(exp_included)
            if n_included >= 2:
                corr_r = np.corrcoef(exp_included['flow_rate'], exp_included['amp_ratio_dist_over_prox'])[0, 1]
                corr_d = np.corrcoef(exp_included['flow_rate'], exp_included['amp_diff_C'])[0, 1]
                corr_info = f'Ratio: r = {corr_r:.3f}\nDiff: r = {corr_d:.3f}'
            else:
                corr_info = 'Need ≥2 points'
            
            info_text = (
                f'PHYSICAL PARAMETERS\n'
                f'───────────────────────\n'
                f'Pipe: {CONFIG["pipe_diameter_inch"]:.2f}" ({pipe_diameter_m:.5f} m)\n'
                f'Distance: {CONFIG["sensor_distance_ft"]:.0f} ft ({distance_m:.2f} m)\n'
                f'Thermal diff: {alpha:.2e} m²/s\n'
                f'Frequency: {frequency_hz:.3f} Hz\n'
                f'Scale factor: {scale_factor:.5f}\n'
                f'\n'
                f'CURRENT SETTINGS\n'
                f'───────────────────────\n'
                f'Adjustment: {adjustment:.6f}\n'
                f'Data points: {n_included}/{n_pairs}\n'
                f'\n'
                f'THEORETICAL MODEL\n'
                f'───────────────────────\n'
                f'A(x)/A₀ = exp(-β·x) + adj\n'
                f'β = k·(ω/v)·√(R²/α)\n'
                f'\n'
                f'CORRELATIONS\n'
                f'───────────────────────\n'
                f'{corr_info}'
            )
            
            ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes,
                     fontsize=9, va='top', family='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
            
            plt.suptitle('Flow Analysis: Experimental vs Theoretical (Jupyter Interactive)',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
    
    # Attach observers
    adjustment_widget.observe(update_plot, names='value')
    for cb in checkbox_widgets:
        cb.observe(update_plot, names='value')
    
    # Create layout
    checkbox_box = VBox(
        [Label('Include Data Points:', style={'font_weight': 'bold'})] + checkbox_widgets,
        layout=widgets.Layout(padding='10px')
    )
    
    controls = HBox([
        adjustment_widget,
        Label('  '),
        checkbox_box
    ])
    
    # Display
    display(controls)
    display(output)
    
    # Initial plot
    update_plot()
    
    print("\n" + "="*60)
    print("INTERACTIVE CONTROLS")
    print("="*60)
    print("• Adjustment box: Enter value and observe plot update")
    print("• Checkboxes: Toggle to include/exclude data points")
    print("• All changes update the plot automatically")
    print("="*60)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" " * 15 + "JUPYTER NOTEBOOK FLOW ANALYSIS")
    print(" " * 10 + "Interactive Analysis with ipywidgets")
    print("="*70)
    print("\nIn Jupyter Notebook, run:")
    print("  %matplotlib inline")
    print("  from flow_analysis_jupyter import create_interactive_analysis")
    print("  create_interactive_analysis()")
    print("\n" + "="*70)
