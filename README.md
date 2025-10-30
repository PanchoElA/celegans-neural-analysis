## celegans-neural-analysis — Project overview

This repository contains analysis code and interactive visualizations for C. elegans neural calcium imaging data. It has been reorganized to separate
data-analysis scripts, interactive dashboard generators, and generated assets.

Top-level structure (after cleanup)

- `analysis_code/` — data preprocessing, PCA, static plotting and analysis scripts.
- `interactive_code/` — dashboard generators, Plotly/Dash apps and generated HTML (interactive outputs).
- `generated_images/` — PNG/JPG/SVG outputs produced by analysis scripts.
- `archive_removed_debug/` — archived debug/test files and backups of removed READMEs/requirements.
- `requirements_consolidated.txt` — consolidated, pinned dependency list for reproducible installs (preferred).

Quick start

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies:

```powershell
# Preferred: use the consolidated, pinned requirements for reproducible installs
pip install -r requirements_consolidated.txt
```

3. Run an analysis or open the interactive outputs:

```powershell
# Run static analysis scripts (from repo root)
python analysis_code/generate_PCA_Final_Clean.py

# Regenerate interactive HTML outputs (optional)
python interactive_code/Interactive_3D_Visualizations.py
python interactive_code/Simple_3D_Visualizations.py

# Open the interactive HTML files (one-click):
.\open_interactive_graphs.bat

# Note: The Dash server is optional; the static HTML files in `interactive_code/` are self-contained.
# To start the Dash server (only if you want a live app):
# python interactive_code/dash_neural_app.py
```

Notes
- The interactive generator scripts have been updated to write files into `interactive_code/` and to resolve the data CSV in the repo root if it's not found in the current working directory.
- Original README and desktop README files were archived in `archive_removed_debug/`.

If you need help running any specific script, tell me which one and I'll provide exact commands and any necessary adjustments.
# C. elegans Neural Activity Analysis

This repository contains Python scripts for analyzing neural activity data from *C. elegans* using calcium imaging data stored in NWB (Neurodata Without Borders) format.

## Overview

This project analyzes neural activity patterns in *C. elegans* using various computational approaches including raster plots, tuning curves, and Principal Component Analysis (PCA).

## Features

- **Raster Plot Visualization**: Heatmap-style visualization of neural activity across time
- **Tuning Curves Analysis**: Relationship between neural activity and behavioral variables (velocity)
- **PCA Analysis**: Dimensionality reduction to identify principal patterns in neural activity
- **Temporal Alignment**: Proper synchronization between neural and behavioral data

## Files Description

### Main Analysis Scripts

- `GCaMP_vs_Time.py` - Basic time series visualization of GCaMP signals for specific neurons
- `Raster_Plot.py` - Generate publication-quality raster plots showing neural activity patterns
- `Tuning_Curves.py` - Analysis of neural responses to velocity changes
- `PCA.py` - Principal Component Analysis of neural activity data

### Jupyter Notebook

- `Analisis NWB File.ipynb` - Interactive analysis notebook with step-by-step exploration

### Generated Outputs

- `PCA_Analysis_Complete.png` - Comprehensive PCA visualization with 6 different plots

## Requirements

```python
numpy
matplotlib
pynwb
scikit-learn
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/celegans-neural-analysis.git
cd celegans-neural-analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

4. Install required packages:
```bash
pip install numpy matplotlib pynwb scikit-learn
```

## Usage

### Running Individual Scripts

Execute any script using the virtual environment Python:

```bash
# Windows
.\venv\Scripts\python.exe script_name.py

# macOS/Linux
./venv/bin/python script_name.py
```

### Examples

1. **Generate Raster Plot**:
```bash
.\venv\Scripts\python.exe Raster_Plot.py
```

2. **Run PCA Analysis**:
```bash
.\venv\Scripts\python.exe PCA.py
```

3. **Create Tuning Curves**:
```bash
.\venv\Scripts\python.exe Tuning_Curves.py
```

## Data Format

This analysis works with NWB files containing:
- Neural calcium imaging data (ΔF/F signals)
- Behavioral data (velocity, position)
- Temporal alignment information

The expected data structure:
- Neural data: `nwbfile.processing['CalciumActivity']['SignalCalciumImResponseSeries']`
- Behavioral data: `nwbfile.processing['Behavior']`

## Key Results

### PCA Analysis Results
- **PC1**: 20.9% of variance (primary neural pattern)
- **PC2**: 17.7% of variance (secondary pattern)
- **PC3**: 10.6% of variance (tertiary pattern)
- **Complexity**: Requires 21 components to capture 80% of neural activity
- **Interpretation**: Highly complex, distributed neural network typical of functional nervous systems

### Neural Network Characteristics
- 147 neurons analyzed
- 1615 temporal points (~16.2 minutes of recording)
- Distributed activity patterns (no single dominant pattern)
- Rich behavioral repertoire

## Scientific Background

This analysis pipeline is designed for studying the neural dynamics of *C. elegans*, a model organism with a completely mapped connectome of 302 neurons. The methods implemented here allow for:

1. **Pattern Discovery**: Identifying recurring neural activity patterns
2. **Behavioral Correlation**: Linking neural states to behavioral outputs
3. **Dimensionality Reduction**: Simplifying complex neural dynamics for interpretation
4. **Temporal Dynamics**: Understanding how neural states evolve over time

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this analysis pipeline.

## License

This project is open source. Please cite appropriately if used in academic work.
