# Using Flow Analysis in Jupyter Notebook

## Problem with Matplotlib Widgets in Jupyter

The original `flow_analysis_with_textbox.py` uses matplotlib's built-in widgets (`TextBox` and `CheckButtons`), which have known compatibility issues with Jupyter Notebook:

- **Overlapping widgets**: Checkboxes and text boxes may overlap or render incorrectly
- **Non-interactive**: Widgets may not respond to clicks or text input
- **Backend issues**: `%matplotlib inline` doesn't support interactive matplotlib widgets

## Solution: Jupyter-Specific Version

Use `flow_analysis_jupyter.py` which uses **ipywidgets** instead of matplotlib widgets. This provides native Jupyter compatibility.

## Installation

Install required packages:

```bash
pip install numpy pandas matplotlib ipywidgets
```

For Jupyter Lab, also enable the extension:

```bash
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

## Usage in Jupyter Notebook

### Step 1: Import and Setup

```python
# In your Jupyter notebook cell:
%matplotlib inline
from flow_analysis_jupyter import create_interactive_analysis
```

### Step 2: Create Interactive Analysis

```python
# This will create interactive controls and plots
create_interactive_analysis()
```

## Features

### Interactive Controls

1. **Adjustment Float Text Box**
   - Enter any numeric value (default: 0.0)
   - Updates plot automatically when you press Enter or click away
   - Adds offset to theoretical amplitude ratios

2. **Flow Rate Checkboxes** (7 checkboxes, one per data pair)
   - All checked by default (include all data)
   - Uncheck to exclude specific flow rates from analysis
   - Useful for removing outliers (e.g., 3.9 GPM)
   - Updates correlations dynamically

3. **Automatic Plot Updates**
   - All 6 plots update automatically when you change controls
   - Correlations recalculate based on selected data
   - Theoretical curves adjust to your settings

### The 6 Plots

1. **Main Amplitude Difference vs Flow Rate** - Experimental vs theoretical attenuation
2. **Relative Attenuation vs Flow Rate** - Amplitude ratio showing thermal signal survival
3. **Weighted Multi-Peak Difference** - Alternative attenuation metric
4. **Weighted Relative Attenuation** - Alternative ratio metric
5. **Proximal and Distal Amplitudes** - Raw temperature signal amplitudes
6. **Info Panel** - System parameters, model equations, and current correlations

## Example Workflow

```python
# 1. Setup
%matplotlib inline
from flow_analysis_jupyter import create_interactive_analysis

# 2. Create interface
create_interactive_analysis()

# 3. Interact with controls:
#    - Type adjustment value (e.g., 0.05) and press Enter
#    - Uncheck 3.9 GPM if it's an outlier
#    - Observe how theoretical curves shift
#    - Watch correlation coefficients update
```

## Comparison of Versions

| Feature | `flow_analysis_with_textbox.py` | `flow_analysis_jupyter.py` |
|---------|--------------------------------|----------------------------|
| Environment | Desktop Python | Jupyter Notebook |
| Widget Library | matplotlib.widgets | ipywidgets |
| Text Box | May not work in Jupyter | ✓ Works in Jupyter |
| Checkboxes | May overlap in Jupyter | ✓ Properly spaced |
| Interactivity | ✓ Works in desktop | ✓ Works in Jupyter |
| Auto-update | Manual (press Enter) | Automatic |

## Troubleshooting

### "ipywidgets not available"

**Solution:** Install ipywidgets

```bash
pip install ipywidgets
```

Then restart your Jupyter kernel.

### Widgets not displaying

**For Jupyter Notebook:**
```bash
jupyter nbextension enable --py widgetsnbextension
```

**For Jupyter Lab:**
```bash
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

Then restart Jupyter.

### Plots not updating

- Make sure you're using `%matplotlib inline` (not `%matplotlib notebook` or `%matplotlib widget`)
- Try restarting the kernel and running all cells again
- Check that ipywidgets is properly installed

### Checkboxes still overlapping

- Try adjusting your browser zoom level
- Increase the width of your notebook browser window
- The layout is optimized for standard notebook widths (900+ pixels)

## Alternative: Use Desktop Version

If you prefer to run outside Jupyter, use the desktop version:

```bash
python flow_analysis_with_textbox.py
```

This will open a matplotlib window with interactive controls that work properly in desktop environments.

## Technical Details

### Why ipywidgets Works Better

- **Native Jupyter integration**: Built specifically for Jupyter environments
- **Proper event handling**: Uses Jupyter's communication protocol
- **Better layout control**: Uses HBox/VBox for predictable positioning
- **Automatic updates**: observe() pattern triggers updates immediately

### How It Works

```python
# Creates widgets
adjustment_widget = FloatText(value=0.0, description='Adjustment:')
checkbox_widgets = [Checkbox(value=True, description=f"{flow} GPM") for flow in flows]

# Attaches observers
adjustment_widget.observe(update_plot, names='value')
for cb in checkbox_widgets:
    cb.observe(update_plot, names='value')

# Displays in notebook
display(HBox([adjustment_widget, VBox(checkbox_widgets)]))
display(Output())  # Plot output area
```

## Data Requirements

The script looks for:
1. `fft_results_preprocessed/FRaTbot_flowdata_2_preprocessed_fft_peaks_20251212_111202.csv`
2. Falls back to embedded experimental data if CSV not found

Required columns (or alternates):
- `flow_rate` (or `flow_gpm`, `flow_rate_gpm`)
- `prox_amp_C` (or `prox_amp`, `proximal_amp`)
- `dist_amp_C` (or `dist_amp`, `distal_amp`)
- `amp_diff_C` (or `amp_diff`)
- `amp_ratio_dist_over_prox` (or `amp_ratio`, `ratio`)

## Support

For issues specific to:
- **Jupyter compatibility**: Use `flow_analysis_jupyter.py`
- **Desktop application**: Use `flow_analysis_with_textbox.py`
- **Static plots only**: Use `flow_analysis_script.py`
