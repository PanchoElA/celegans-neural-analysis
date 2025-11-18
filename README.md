# C. elegans Neural Activity Analysis

This repository contains a comprehensive Python suite for analyzing neural activity and behavior in *C. elegans* from calcium imaging data. The project is designed to work with data in the Neurodata Without Borders (NWB) format.

## Overview

This project provides scripts to load, process, and analyze complex neural datasets. It allows for the exploration of brain-wide activity patterns and their correlation with specific worm behaviors, such as velocity.

The primary analyses include:

  * **NWB Data Exploration**: An interactive notebook to inspect the NWB file structure and its contents.
  * **Basic Visualization**: Scripts to generate raster plots and basic GCaMP time-series plots.
  * **Behavioral Correlation**: Analysis of neural tuning curves against behavioral variables like velocity.
  * **Dimensionality Reduction**: A large suite of scripts for performing Principal Component Analysis (PCA) to identify dominant neural patterns, including analyses of derivatives and 2D/3D visualizations.
  * **Interactive Dashboards**: Generation of interactive 3D visualizations and dashboards (using Plotly and Dash) for dynamic data exploration.

-----

## Repository Structure

The repository is organized into modules for analysis, interactive code, and precomputed data.

```
/
│
├── Analisis NWB File.ipynb         # Jupyter Notebook for exploring the NWB file
├── Neural_Dashboard_Interactive.html # Example of a generated interactive dashboard
├── requirements.txt                # Full list of Python dependencies
├── README.md                       # This documentation file
├── QUICKSTART.md                   # A brief guide to running scripts
│
├── analysis_code/                  # Contains all main .py analysis scripts
│   ├── Raster_Plot.py
│   ├── Tuning_Curves.py
│   ├── PCA_Enhanced_Analysis.py
│   └── ... (and many other PCA/analysis scripts)
│
├── interactive_code/               # Scripts for generating interactive Plotly/Dash plots
│   ├── Interactive_3D_Visualizations.py
│   ├── dash_neural_app.py
│   └── ...
│
├── generated_images/               # Default output directory for static plots (e.g., .png)
│   ├── Raster_Plot_Output.png
│   ├── PCA_Complete_Analysis.png
│   └── ...
│
└── *.csv                           # Precomputed data files (e.g., from the NWB)
    ├── principal_components_top5.csv
    ├── neural_data_dataframe.csv
    └── ...
```

-----

## Data Requirements

### NWB Dataset

This analysis suite is designed for a specific NWB file which is not included in the repository due to its large size.

  * **File:** `sub-2023-01-09-15-SWF702_ses-20230109_behavior+image+ophys.nwb`
  * **Source:** DANDI Archive
  * **Dandiset:** `000776`
  * **Release:** `0.241009.1509`
  * **Download:** [https://dandiarchive.org/dandiset/000776/0.241009.1509](https://dandiarchive.org/dandiset/000776/0.241009.1509)

### Data Format

The scripts expect to find data at specific paths within the NWB file, which you can verify using the `Analisis NWB File.ipynb` notebook.

  * **Neural Data:** `nwbfile.processing['CalciumActivity']['SignalCalciumImResponseSeries']`
  * **Neural Timestamps:** `nwbfile.processing['CalciumActivity']['SignalCalciumImResponseSeries'].timestamps`
  * **Neuron IDs:** `nwbfile.processing['CalciumActivity']['NeuronIDs'].labels`
  * **Behavioral Data:** `nwbfile.processing['Behavior']` (contains `velocity`, `head_curvature`, etc.)

### Local Usage

1.  Download the NWB file from the DANDI archive.
2.  Place the file in the root of this repository OR set the environment variable `CELEGANS_NWB_PATH` to its full path.

If you do not download the NWB file, many scripts can fall back on using the precomputed `.csv` files included in the repository.

-----

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/panchoela/celegans-neural-analysis.git
    cd celegans-neural-analysis
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    # Create the environment
    python -m venv venv

    # Activate on Windows
    venv\Scripts\activate

    # Activate on macOS/Linux
    source venv/bin/activate
    ```

3.  **Install requirements:**
    This project uses specific library versions. Install them directly from the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

-----

## Software Requirements

The `requirements.txt` file specifies the following key libraries:

  * `numpy==1.23.5`
  * `pandas==1.5.3`
  * `scipy==1.9.3`
  * `scikit-learn==1.1.3`
  * `matplotlib==3.6.3`
  * `plotly==5.13.1` (For interactive plots)
  * `kaleido==0.2.1` (For exporting Plotly plots)
  * `pynwb==2.2.0` (For reading NWB files)
  * `hdmf==3.0.0` (A dependency for pynwb)

-----

## Usage

Ensure your virtual environment is activated before running any scripts.

### 1\. Explore the NWB Data

For a step-by-step breakdown of the NWB file structure, run the Jupyter Notebook:

```bash
jupyter notebook "Analisis NWB File.ipynb"
```

### 2\. Run Analysis Scripts

Execute scripts from the `analysis_code` directory. Many scripts will save their output plots to the `generated_images/` folder.

```bash
# Example: Generate a raster plot
python analysis_code/Raster_Plot.py

# Example: Run an enhanced PCA analysis
python analysis_code/PCA_Enhanced_Analysis.py

# Example: Create tuning curves
python analysis_code/Tuning_Curves.py
```

### 3\. View Interactive Visualizations

The HTML files in the root and in `interactive_code/` can be opened directly in your web browser.

  * `Neural_Dashboard_Interactive.html`
  * `interactive_code/3D_Dashboard_Neural_Integrado.html`

You can also run the interactive visualization scripts (this may start a local web server):

```bash
# Example: Run the main Dash application
python interactive_code/dash_neural_app.py
```

-----

## Key Scientific Findings (from Data)

This analysis is based on a recording of **147 neurons** over **1615 temporal points** (approx. 16.2 minutes).

### PCA Results

The PCA reveals a highly complex and distributed neural network, with no single dominant pattern.

  * **PC1**: 20.9% of variance
  * **PC2**: 17.7% of variance
  * **PC3**: 10.6% of variance
  * **Complexity**: 21 components are required to capture 80% of the neural activity's variance.

-----

Let me know if you'd like to dive into modifying any of these scripts, understanding the NWB data structure better, or organizing the repository further\!