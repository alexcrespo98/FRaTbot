# Interactive Flow Analysis Tool

## Overview

The `flow_analysis_interactive.py` script provides an interactive visualization for comparing experimental temperature amplitude decay data with theoretical predictions.

## Features

### 1. **Heat Loss / Attenuation Factor Slider**
- Located at the top of the window
- Drag the slider left/right to adjust the theoretical model's attenuation factor
- Range: 0.0001 to 0.002
- Default: 0.00053
- Real-time updates to theoretical curves as you drag

**What it does:** The attenuation factor accounts for heat loss through pipe walls, non-ideal flow conditions, and other real-world effects not captured in the idealized theoretical model. Adjusting this helps fit the theoretical predictions to match experimental observations.

### 2. **Flow Rate Pair Selection Checkboxes**
- Located on the left side
- 7 checkboxes (one for each flow rate: 3.2, 3.9, 5.4, 6.7, 7.8, 8.6, 9.4 GPM)
- Click to toggle data points on/off
- Useful for excluding outliers or focusing on specific flow rate ranges

**What it does:** Some experimental measurements may be outliers due to measurement errors, transient flow conditions, or other anomalies. You can uncheck these to see how the correlation improves without them.

### 3. **Interactive Plots**

The visualization includes two main plots:

- **Top Plot:** Temperature Amplitude Ratio (Dist/Prox) vs Flow Rate
  - Shows how much of the temperature signal survives the 50 ft journey
  - Higher ratios = less attenuation
  - Includes correlation coefficient (r-value)

- **Bottom Plot:** Temperature Amplitude Difference (Prox - Dist) vs Flow Rate
  - Shows absolute temperature attenuation in °C
  - Higher difference = more attenuation

### 4. **Information Panel**
- System parameters (pipe diameter, distance, frequency, fluid properties)
- Current settings (attenuation factor, number of data points shown)
- Theoretical model equation
- Usage instructions

## Usage

### Running the Interactive Script

```bash
python flow_analysis_interactive.py
```

### Requirements

```bash
pip install numpy pandas matplotlib scipy
```

### Workflow

1. **Launch the script** - A window will open with the interactive visualization
2. **Adjust the slider** - Fine-tune the attenuation factor to match theory with experiment
3. **Toggle checkboxes** - Exclude outliers or focus on specific flow rates
4. **Observe changes** - Both plots and correlation update in real-time
5. **Close window** - When satisfied with your analysis

### Tips

- **Finding the best fit:** Adjust the slider until the red theoretical curve closely follows the orange experimental points in the ratio plot
- **Identifying outliers:** Look for data points that deviate significantly from the trend. Uncheck them to see if correlation improves
- **Typical outlier:** The 3.9 GPM measurement (ratio = 0.122) appears to be an outlier compared to the general trend

## Comparison with Static Script

| Feature | `flow_analysis_script.py` | `flow_analysis_interactive.py` |
|---------|---------------------------|--------------------------------|
| Visualization | Static 6-panel figure | Interactive 2-panel with controls |
| Attenuation factor | Fixed (0.00053) | Adjustable via slider |
| Data filtering | All points always shown | Toggle individual points |
| Use case | Publication/documentation | Exploration/analysis |
| Output | PNG file saved to disk | Interactive window |

## Example Scenarios

### Scenario 1: Tuning the Model
"The theoretical curve doesn't quite match my data."
- **Solution:** Drag the slider to adjust the attenuation factor until curves align

### Scenario 2: Handling Outliers
"One data point seems wrong and skews my correlation."
- **Solution:** Uncheck that flow rate's checkbox to exclude it from analysis

### Scenario 3: Understanding Trends
"I want to see if high flow rates follow theory better than low flow rates."
- **Solution:** Uncheck low flow rate boxes and observe if correlation improves

## Technical Details

### Physical Model

The theoretical model uses phenomenological amplitude decay:

```
A(x)/A₀ = exp(-β·x)
```

where:
- `β = k·(ω/v)·√(R²/α)` (attenuation coefficient)
- `k` = attenuation scale factor (adjustable via slider)
- `ω` = angular frequency (2π·f)
- `v` = flow velocity
- `R` = pipe radius
- `α` = thermal diffusivity

### Data Format

The script loads data from:
1. `fft_results_preprocessed/FRaTbot_flowdata_2_preprocessed_fft_peaks_20251212_111202.csv` (if available)
2. Embedded experimental data (fallback)

Required columns:
- `flow_rate` (or `flow_gpm`, `flow_rate_gpm`)
- `prox_amp_C` (or `prox_amp`, `proximal_amp`)
- `dist_amp_C` (or `dist_amp`, `distal_amp`)
- `amp_diff_C` (or `amp_diff`)
- `amp_ratio_dist_over_prox` (or `amp_ratio`, `ratio`)

## Troubleshooting

**Issue:** "No module named 'matplotlib.widgets'"
- **Solution:** Update matplotlib: `pip install --upgrade matplotlib`

**Issue:** "Window doesn't appear"
- **Solution:** Make sure you're running in an environment with display support (not headless)

**Issue:** "Slider doesn't respond"
- **Solution:** Click and drag (don't just click). The slider updates in real-time.

**Issue:** "All checkboxes unchecked - blank plots"
- **Solution:** Check at least one checkbox to see data

## Files

- `flow_analysis_interactive.py` - Main interactive script
- `flow_analysis_script.py` - Original static analysis script (6 plots)
- `flow_analysis_interactive_demo.png` - Screenshot of interactive interface
- `INTERACTIVE_FLOW_ANALYSIS.md` - This documentation

## Credits

Implements theoretical model based on advection-diffusion equations for oscillatory thermal signals in pipe flow.
