#!/usr/bin/env python3
"""
Jupyter Notebook Cell - Complete Flow Analysis with Interactive Controls

USAGE IN JUPYTER NOTEBOOK:
Copy this entire file content into a single Jupyter cell and run it.
No imports needed - everything is self-contained!

Requirements:
- pip install numpy pandas matplotlib ipywidgets scipy

The cell will display:
1. Experimental data summary
2. Correlation analysis  
3. Physical parameters
4. Theoretical vs experimental comparison
5. All 6 interactive plots with adjustment controls
"""

# ============================================================================
# COPY EVERYTHING BELOW THIS LINE INTO A JUPYTER NOTEBOOK CELL
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import ipywidgets as widgets
from IPython.display import display, HTML
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Data files (will try to load, fallback to embedded data if not found)
    'preprocessed_csv': 'fft_results_preprocessed/FRaTbot_flowdata_2_preprocessed_fft_peaks_20251212_111202.csv',
    'fallback_csv': 'FRaTbot_flowdata_2.csv',
    
    # Target frequency
    'target_frequency_hz': 0.200,
    
    # Physical parameters for theoretical model
    'pipe_diameter_inch': 0.75,
    'sensor_distance_ft': 50.0,
    'water_thermal_diffusivity_m2s': 1.47e-7,  # at 20°C
    
    # Empirical fitting parameter
    'attenuation_scale_factor': 0.00053,
}

# ============================================================================
# EMBEDDED EXPERIMENTAL DATA (fallback if CSV not found)
# ============================================================================

EMBEDDED_DATA = {
    'flow_rate': [9.4, 8.6, 7.8, 6.7, 5.4, 3.9, 3.2],
    'prox_amp_C': [0.221, 0.432, 0.384, 0.456, 0.376, 0.417, 0.447],
    'dist_amp_C': [0.196, 0.251, 0.221, 0.151, 0.066, 0.051, 0.137],
    'amp_diff_C': [0.025, 0.181, 0.163, 0.306, 0.310, 0.367, 0.310],
    'amp_ratio': [0.885, 0.580, 0.577, 0.330, 0.176, 0.122, 0.308],
    'weighted_diff_C': [0.007, 0.130, 0.123, 0.190, 0.180, 0.224, 0.224],
    'weighted_ratio': [0.995, 0.641, 0.578, 0.488, 0.261, 0.319, 0.434],
}

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def find_column_case_insensitive(df, target_col, alternatives):
    """Find column by name, case-insensitive, trying alternatives."""
    cols_lower = {col.lower(): col for col in df.columns}
    
    # Try target first
    if target_col.lower() in cols_lower:
        return cols_lower[target_col.lower()]
    
    # Try alternatives
    for alt in alternatives:
        if alt.lower() in cols_lower:
            return cols_lower[alt.lower()]
    
    return None

def load_preprocessed_data(config):
    """Load data from CSV or use embedded fallback."""
    import os
    
    # Try preprocessed CSV first
    if os.path.exists(config['preprocessed_csv']):
        try:
            df = pd.read_csv(config['preprocessed_csv'])
            
            # Map column names
            col_map = {}
            col_map['flow_rate'] = find_column_case_insensitive(df, 'flow_rate', ['flow_gpm', 'flow', 'flowrate'])
            col_map['prox_amp_C'] = find_column_case_insensitive(df, 'prox_amp_C', ['prox_amp', 'proximal_amplitude', 'prox_amplitude'])
            col_map['dist_amp_C'] = find_column_case_insensitive(df, 'dist_amp_C', ['dist_amp', 'distal_amplitude', 'dist_amplitude'])
            col_map['amp_ratio'] = find_column_case_insensitive(df, 'amp_ratio_dist_over_prox', ['amp_ratio', 'ratio', 'amplitude_ratio'])
            
            # Check required columns
            missing = [k for k, v in col_map.items() if k != 'amp_ratio' and v is None]
            if missing:
                print(f"Warning: Missing columns {missing}, using embedded data")
                return pd.DataFrame(EMBEDDED_DATA)
            
            # Rename columns to standard names
            rename_dict = {v: k for k, v in col_map.items() if v is not None}
            df = df.rename(columns=rename_dict)
            
            # Calculate ratio if not present
            if 'amp_ratio' not in df.columns and 'prox_amp_C' in df.columns and 'dist_amp_C' in df.columns:
                df['amp_ratio'] = df['dist_amp_C'] / df['prox_amp_C']
            
            # Calculate difference if not present
            if 'amp_diff_C' not in df.columns and 'prox_amp_C' in df.columns and 'dist_amp_C' in df.columns:
                df['amp_diff_C'] = df['prox_amp_C'] - df['dist_amp_C']
            
            # Try to get weighted metrics (optional)
            weighted_diff = find_column_case_insensitive(df, 'weighted_multi_peak_diff_C', ['weighted_diff', 'weighted_difference'])
            weighted_ratio = find_column_case_insensitive(df, 'weighted_multi_peak_ratio', ['weighted_ratio'])
            
            if weighted_diff:
                df['weighted_diff_C'] = df[weighted_diff]
            if weighted_ratio:
                df['weighted_ratio'] = df[weighted_ratio]
            
            return df[['flow_rate', 'prox_amp_C', 'dist_amp_C', 'amp_diff_C', 'amp_ratio'] + 
                     (['weighted_diff_C'] if 'weighted_diff_C' in df.columns else []) +
                     (['weighted_ratio'] if 'weighted_ratio' in df.columns else [])]
            
        except Exception as e:
            print(f"Error loading CSV: {e}, using embedded data")
            return pd.DataFrame(EMBEDDED_DATA)
    
    # Use embedded data
    print("CSV not found, using embedded experimental data")
    return pd.DataFrame(EMBEDDED_DATA)

# ============================================================================
# THEORETICAL CALCULATION FUNCTIONS
# ============================================================================

def calculate_flow_velocity_ms(flow_rate_gpm, pipe_diameter_m):
    """Calculate flow velocity from flow rate in GPM."""
    flow_rate_m3s = flow_rate_gpm * 6.309e-5  # GPM to m³/s
    pipe_area_m2 = np.pi * (pipe_diameter_m / 2.0) ** 2
    velocity_ms = flow_rate_m3s / pipe_area_m2
    return velocity_ms

def calculate_theoretical_amplitude_ratio(flow_rate_gpm, frequency_hz, distance_m, 
                                         pipe_diameter_m, thermal_diffusivity_m2s, 
                                         scale_factor=1.0, adjustment=0.0):
    """Calculate theoretical amplitude ratio with phenomenological model."""
    omega = 2.0 * np.pi * frequency_hz
    velocity_ms = calculate_flow_velocity_ms(flow_rate_gpm, pipe_diameter_m)
    radius_m = pipe_diameter_m / 2.0
    
    # Attenuation coefficient
    beta = scale_factor * (omega / velocity_ms) * np.sqrt(radius_m**2 / thermal_diffusivity_m2s)
    
    # Amplitude ratio with adjustment offset
    ratio = np.exp(-distance_m * beta) + adjustment
    return ratio

def calculate_theoretical_amplitude_difference(flow_rate_gpm, prox_amp_C, frequency_hz, 
                                              distance_m, pipe_diameter_m, 
                                              thermal_diffusivity_m2s, scale_factor=1.0,
                                              adjustment=0.0):
    """Calculate theoretical amplitude difference."""
    ratio = calculate_theoretical_amplitude_ratio(
        flow_rate_gpm, frequency_hz, distance_m, pipe_diameter_m,
        thermal_diffusivity_m2s, scale_factor, adjustment
    )
    dist_amp_theoretical = prox_amp_C * ratio
    diff_theoretical = prox_amp_C - dist_amp_theoretical
    return diff_theoretical

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_plots(experimental_df, config, adjustment=0.0, included_indices=None):
    """Create all 6 plots with theoretical overlays."""
    
    if included_indices is None:
        included_indices = list(range(len(experimental_df)))
    
    # Filter data
    exp_filtered = experimental_df.iloc[included_indices].copy()
    
    # Unit conversions
    pipe_diameter_m = config['pipe_diameter_inch'] * 0.0254
    distance_m = config['sensor_distance_ft'] * 0.3048
    
    # Generate theoretical curves
    flow_range = np.linspace(exp_filtered['flow_rate'].min() - 0.5, 
                             exp_filtered['flow_rate'].max() + 0.5, 100)
    
    theory_ratios = np.array([
        calculate_theoretical_amplitude_ratio(
            fr, config['target_frequency_hz'], distance_m, pipe_diameter_m,
            config['water_thermal_diffusivity_m2s'], 
            config['attenuation_scale_factor'], adjustment
        ) for fr in flow_range
    ])
    
    mean_prox_amp = exp_filtered['prox_amp_C'].mean()
    theory_diffs = np.array([
        calculate_theoretical_amplitude_difference(
            fr, mean_prox_amp, config['target_frequency_hz'], distance_m,
            pipe_diameter_m, config['water_thermal_diffusivity_m2s'],
            config['attenuation_scale_factor'], adjustment
        ) for fr in flow_range
    ])
    
    # Calculate correlations
    if len(exp_filtered) >= 2:
        corr_ratio = np.corrcoef(exp_filtered['flow_rate'], exp_filtered['amp_ratio'])[0, 1]
        corr_diff = np.corrcoef(exp_filtered['flow_rate'], exp_filtered['amp_diff_C'])[0, 1]
    else:
        corr_ratio = corr_diff = np.nan
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: Main amplitude difference
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(exp_filtered['flow_rate'], exp_filtered['amp_diff_C'], 'o-', 
             color='blue', markersize=8, label='Experimental', linewidth=2)
    ax1.plot(flow_range, theory_diffs, '--', color='red', linewidth=2, 
             label='Theoretical')
    ax1.set_xlabel('Flow Rate (GPM)', fontsize=11)
    ax1.set_ylabel('Amplitude Difference [°C]', fontsize=11)
    ax1.set_title('Main Attenuation vs Flow Rate', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(0.05, 0.95, f'r = {corr_diff:.3f}', transform=ax1.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             verticalalignment='top')
    
    # Plot 2: Main amplitude ratio
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(exp_filtered['flow_rate'], exp_filtered['amp_ratio'], 'o-', 
             color='orange', markersize=8, label='Experimental', linewidth=2)
    ax2.plot(flow_range, theory_ratios, '--', color='red', linewidth=2, 
             label='Theoretical')
    ax2.set_xlabel('Flow Rate (GPM)', fontsize=11)
    ax2.set_ylabel('Amplitude Ratio (Dist/Prox)', fontsize=11)
    ax2.set_title('Relative Attenuation vs Flow Rate', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.text(0.05, 0.95, f'r = {corr_ratio:.3f}', transform=ax2.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             verticalalignment='top')
    
    # Plot 3: Weighted difference (if available)
    ax3 = fig.add_subplot(gs[0, 2])
    if 'weighted_diff_C' in exp_filtered.columns:
        ax3.plot(exp_filtered['flow_rate'], exp_filtered['weighted_diff_C'], 'o-',
                 color='purple', markersize=8, label='Experimental', linewidth=2)
        ax3.plot(flow_range, theory_diffs, '--', color='red', linewidth=2,
                 label='Theoretical')
        ax3.set_ylabel('Weighted Difference [°C]', fontsize=11)
    else:
        ax3.text(0.5, 0.5, 'Weighted metrics\nnot available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
    ax3.set_xlabel('Flow Rate (GPM)', fontsize=11)
    ax3.set_title('Weighted Attenuation vs Flow Rate', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Weighted ratio (if available)
    ax4 = fig.add_subplot(gs[1, 0])
    if 'weighted_ratio' in exp_filtered.columns:
        ax4.plot(exp_filtered['flow_rate'], exp_filtered['weighted_ratio'], 'o-',
                 color='green', markersize=8, label='Experimental', linewidth=2)
        ax4.plot(flow_range, theory_ratios, '--', color='red', linewidth=2,
                 label='Theoretical')
        ax4.set_ylabel('Weighted Ratio', fontsize=11)
    else:
        ax4.text(0.5, 0.5, 'Weighted metrics\nnot available',
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    ax4.set_xlabel('Flow Rate (GPM)', fontsize=11)
    ax4.set_title('Weighted Relative Attenuation', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Proximal and distal amplitudes
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(exp_filtered['flow_rate'], exp_filtered['prox_amp_C'], 'o-',
             color='blue', markersize=8, label='Proximal', linewidth=2)
    ax5.plot(exp_filtered['flow_rate'], exp_filtered['dist_amp_C'], 's-',
             color='red', markersize=8, label='Distal', linewidth=2)
    ax5.set_xlabel('Flow Rate (GPM)', fontsize=11)
    ax5.set_ylabel('Amplitude [°C]', fontsize=11)
    ax5.set_title('Proximal and Distal Amplitudes', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Info panel
    ax6 = fig.add_subplot(gs[1, 2:])
    ax6.axis('off')
    
    info_text = f"""PHYSICAL PARAMETERS
━━━━━━━━━━━━━━━━━━━━
Pipe: {config['pipe_diameter_inch']}" ({pipe_diameter_m:.5f} m)
Distance: {config['sensor_distance_ft']} ft ({distance_m:.2f} m)
Thermal diffusivity: {config['water_thermal_diffusivity_m2s']:.2e} m²/s
Frequency: {config['target_frequency_hz']} Hz
Scale factor: {config['attenuation_scale_factor']}

CURRENT SETTINGS
━━━━━━━━━━━━━━━━━━━━
Adjustment offset: {adjustment:.6f}
Data points shown: {len(exp_filtered)}/{len(experimental_df)}

THEORETICAL MODEL
━━━━━━━━━━━━━━━━━━━━
A(x)/A₀ = exp(-β·x) + offset

where β = k·(ω/v)·√(R²/α)

CORRELATIONS
━━━━━━━━━━━━━━━━━━━━
Ratio vs flow: r = {corr_ratio:.3f}
Diff vs flow: r = {corr_diff:.3f}

CONTROLS
━━━━━━━━━━━━━━━━━━━━
• Enter adjustment value in text box
• Uncheck boxes to exclude data points
"""
    
    ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Bottom row: Comparison table
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    # Calculate theoretical values for experimental flow rates
    theory_exp_ratios = [
        calculate_theoretical_amplitude_ratio(
            fr, config['target_frequency_hz'], distance_m, pipe_diameter_m,
            config['water_thermal_diffusivity_m2s'],
            config['attenuation_scale_factor'], adjustment
        ) for fr in exp_filtered['flow_rate']
    ]
    
    if len(exp_filtered) >= 2:
        rmse = np.sqrt(np.mean((exp_filtered['amp_ratio'].values - theory_exp_ratios)**2))
    else:
        rmse = np.nan
    
    comparison_text = "THEORETICAL vs EXPERIMENTAL COMPARISON\n\n"
    comparison_text += f"{'Flow (GPM)':<12} {'Exp Ratio':<12} {'Theory Ratio':<12} {'Error':<12}\n"
    comparison_text += "─" * 50 + "\n"
    
    for i, (flow, exp_ratio, theory_ratio) in enumerate(zip(
        exp_filtered['flow_rate'], exp_filtered['amp_ratio'], theory_exp_ratios
    )):
        error = exp_ratio - theory_ratio
        comparison_text += f"{flow:<12.1f} {exp_ratio:<12.3f} {theory_ratio:<12.3f} {error:+12.3f}\n"
    
    comparison_text += "─" * 50 + "\n"
    comparison_text += f"RMSE: {rmse:.4f}\n"
    
    ax7.text(0.05, 0.95, comparison_text, transform=ax7.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    fig.suptitle('Flow Analysis: Experimental vs Theoretical (Adjustable with Data Filtering)',
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    return fig

# ============================================================================
# MAIN INTERACTIVE FUNCTION
# ============================================================================

def create_interactive_analysis():
    """Create complete interactive analysis with all analytics."""
    
    # Load data
    experimental_df = load_preprocessed_data(CONFIG)
    
    # Display analytics header
    display(HTML("<h2 style='color: #2E86AB;'>🔬 Flow Analysis: Temperature Amplitude Decay</h2>"))
    
    # 1. Experimental Data Summary
    display(HTML("<h3 style='color: #A23B72;'>📊 Experimental Data Summary</h3>"))
    display(experimental_df.style.format({
        'flow_rate': '{:.1f}',
        'prox_amp_C': '{:.3f}',
        'dist_amp_C': '{:.3f}',
        'amp_diff_C': '{:.3f}',
        'amp_ratio': '{:.3f}',
    }).set_properties(**{'text-align': 'center'}).set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center'), ('background-color', '#f0f0f0')]}
    ]))
    
    # 2. Correlation Analysis
    corr_ratio = np.corrcoef(experimental_df['flow_rate'], experimental_df['amp_ratio'])[0, 1]
    corr_diff = np.corrcoef(experimental_df['flow_rate'], experimental_df['amp_diff_C'])[0, 1]
    
    display(HTML(f"""
    <h3 style='color: #A23B72;'>📈 Correlation Analysis</h3>
    <ul style='font-size: 14px;'>
        <li><b>Amplitude Ratio vs Flow Rate:</b> r = {corr_ratio:.3f}</li>
        <li><b>Amplitude Difference vs Flow Rate:</b> r = {corr_diff:.3f}</li>
    </ul>
    """))
    
    # 3. Physical Parameters
    pipe_diameter_m = CONFIG['pipe_diameter_inch'] * 0.0254
    distance_m = CONFIG['sensor_distance_ft'] * 0.3048
    
    display(HTML(f"""
    <h3 style='color: #A23B72;'>⚙️ Physical System Parameters</h3>
    <ul style='font-size: 14px;'>
        <li><b>Pipe Diameter:</b> {CONFIG['pipe_diameter_inch']}" ({pipe_diameter_m:.5f} m)</li>
        <li><b>Sensor Distance:</b> {CONFIG['sensor_distance_ft']} ft ({distance_m:.2f} m)</li>
        <li><b>Water Thermal Diffusivity:</b> {CONFIG['water_thermal_diffusivity_m2s']:.2e} m²/s (at 20°C)</li>
        <li><b>Target Frequency:</b> {CONFIG['target_frequency_hz']} Hz</li>
        <li><b>Attenuation Scale Factor:</b> {CONFIG['attenuation_scale_factor']}</li>
    </ul>
    """))
    
    # 4. Theoretical Model
    display(HTML("""
    <h3 style='color: #A23B72;'>📐 Theoretical Model</h3>
    <div style='background-color: #f9f9f9; padding: 15px; border-left: 4px solid #2E86AB; font-family: monospace;'>
        <b>Phenomenological Amplitude Decay:</b><br>
        A(x)/A₀ = exp(-β·x) + adjustment<br><br>
        <b>Where:</b><br>
        • β = k·(ω/v)·√(R²/α) [attenuation coefficient]<br>
        • k = 0.00053 [empirical scale factor]<br>
        • ω = 2πf [angular frequency, f = 0.200 Hz]<br>
        • v = flow velocity [m/s]<br>
        • R = pipe radius [0.00953 m]<br>
        • α = 1.47×10⁻⁷ m²/s [water thermal diffusivity]<br>
        • adjustment = user-defined offset [default: 0.0]
    </div>
    """))
    
    # 5. Theoretical vs Experimental Comparison
    theory_exp_ratios = [
        calculate_theoretical_amplitude_ratio(
            fr, CONFIG['target_frequency_hz'], distance_m, pipe_diameter_m,
            CONFIG['water_thermal_diffusivity_m2s'],
            CONFIG['attenuation_scale_factor'], 0.0
        ) for fr in experimental_df['flow_rate']
    ]
    
    rmse = np.sqrt(np.mean((experimental_df['amp_ratio'].values - theory_exp_ratios)**2))
    
    comparison_df = pd.DataFrame({
        'Flow Rate (GPM)': experimental_df['flow_rate'],
        'Exp Ratio': experimental_df['amp_ratio'],
        'Theory Ratio': theory_exp_ratios,
        'Error': experimental_df['amp_ratio'].values - theory_exp_ratios
    })
    
    display(HTML(f"<h3 style='color: #A23B72;'>🎯 Theoretical vs Experimental Comparison (RMSE: {rmse:.4f})</h3>"))
    display(comparison_df.style.format({
        'Flow Rate (GPM)': '{:.1f}',
        'Exp Ratio': '{:.3f}',
        'Theory Ratio': '{:.3f}',
        'Error': '{:+.3f}'
    }).set_properties(**{'text-align': 'center'}).set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center'), ('background-color', '#f0f0f0')]}
    ]))
    
    # 6. Interactive Plots with Controls
    display(HTML("<h3 style='color: #A23B72;'>📊 Interactive Plots with Adjustment Controls</h3>"))
    display(HTML("<p style='font-size: 14px;'><i>Adjust the offset value and toggle data points to see real-time updates</i></p>"))
    
    # Create widgets
    adjustment_widget = widgets.FloatText(
        value=0.0,
        description='Adjustment:',
        step=0.001,
        style={'description_width': '100px'}
    )
    
    # Create checkboxes for each data point
    checkboxes = []
    checkbox_labels = []
    for i, flow in enumerate(experimental_df['flow_rate']):
        cb = widgets.Checkbox(
            value=True,
            description=f'{flow:.1f} GPM',
            style={'description_width': '80px'}
        )
        checkboxes.append(cb)
        checkbox_labels.append(f'{flow:.1f} GPM')
    
    # Organize checkboxes in columns
    n_cols = 4
    n_rows = int(np.ceil(len(checkboxes) / n_cols))
    checkbox_grid = []
    for row in range(n_rows):
        row_boxes = checkboxes[row*n_cols:(row+1)*n_cols]
        if row_boxes:
            checkbox_grid.append(widgets.HBox(row_boxes))
    
    checkbox_vbox = widgets.VBox(checkbox_grid)
    
    # Output widget for plots
    output = widgets.Output()
    
    def update_plot(change=None):
        """Update plot when controls change."""
        with output:
            output.clear_output(wait=True)
            
            # Get current adjustment
            adj = adjustment_widget.value
            
            # Get included indices
            included = [i for i, cb in enumerate(checkboxes) if cb.value]
            
            # Create plots
            fig = create_plots(experimental_df, CONFIG, adj, included)
            plt.show()
    
    # Attach observers
    adjustment_widget.observe(update_plot, names='value')
    for cb in checkboxes:
        cb.observe(update_plot, names='value')
    
    # Display controls
    controls_box = widgets.VBox([
        widgets.HTML("<h4>Adjustment Offset</h4>"),
        adjustment_widget,
        widgets.HTML("<h4>Include Data Points</h4>"),
        checkbox_vbox
    ])
    
    display(controls_box)
    display(output)
    
    # Initial plot
    update_plot()
    
    display(HTML("""
    <hr style='margin-top: 30px;'>
    <p style='text-align: center; color: #666; font-size: 12px;'>
        Generated by FRaTbot Flow Analysis • Theoretical Model: Phenomenological Amplitude Decay
    </p>
    """))

# ============================================================================
# RUN THE ANALYSIS
# ============================================================================

# Automatically run when cell is executed
create_interactive_analysis()
