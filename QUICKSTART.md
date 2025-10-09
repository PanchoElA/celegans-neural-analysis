# C. elegans Neural Activity Analysis

## Quick Start Guide

This repository contains analysis scripts for *C. elegans* neural activity data from calcium imaging experiments.

### Main Analysis Types

1. **Raster Plots** (`Raster_Plot.py`) - Visualize neural activity patterns
2. **Tuning Curves** (`Tuning_Curves.py`) - Neural response to behavioral variables  
3. **PCA Analysis** (`PCA.py`) - Dimensionality reduction and pattern identification

### Running the Analysis

Make sure you have the virtual environment activated and run:

```bash
# For raster plots
.\venv\Scripts\python.exe Raster_Plot.py

# For PCA analysis  
.\venv\Scripts\python.exe PCA.py

# For tuning curves
.\venv\Scripts\python.exe Tuning_Curves.py
```

### Expected Outputs

- High-resolution PNG plots
- Console output with analysis results
- Statistical summaries

See the full README.md for detailed documentation.

## Data Requirements

- NWB format files with calcium imaging data
- Behavioral tracking data (velocity, position)
- Proper temporal alignment between neural and behavioral signals