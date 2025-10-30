## interactive_code — Generator scripts & interactive outputs

This folder contains the generator scripts and lightweight Dash/launcher code that produce the interactive HTML dashboards for the project. The generator scripts create standalone Plotly HTML files (no server required) and write them into this folder by design.

Where the HTML lives
- All generated interactive HTML files are written into this directory (`interactive_code/`). Look for files with the `.html` extension (for example: `Interactive_3D_PCA_Trajectory.html`, `Neural_Dashboard_Interactive.html`, `3D_Trayectoria_PCA_Neural.html`, etc.).

Main scripts
- `Interactive_3D_Visualizations.py` — primary generator that computes PCA/derivatives, fits splines and writes a set of comprehensive interactive HTML dashboards.
- `Simple_3D_Visualizations.py` — smaller generator producing a minimal set of 3D visualization HTML files (useful for quick checks).
- `run_visualizations.py` — convenience launcher that runs one or more generators in sequence.
- `dash_neural_app.py` — Dash app implementation (optional). Run this only if you want a live Dash server; static HTML pages are sufficient for most use cases.
- `integrated_dash_app.py`, `neural_desktop_app_v2.py`, `launcher.py` — supporting apps/launchers (use as needed).

How to run
- From the repository root (recommended):

```powershell
# Activate your virtual environment first
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Regenerate the full set of interactive HTML files
python interactive_code/Interactive_3D_Visualizations.py

# Or run the simpler generator
python interactive_code/Simple_3D_Visualizations.py

# Optional: start a Dash server
python interactive_code/dash_neural_app.py
```

Notes about data paths and behavior
- The generator scripts are written to be robust when executed from the repo root: if they don't find the `neural_data_dataframe.csv` in the current working directory they will look in the parent/repo root directory automatically.
- Generated HTML files are self-contained (they embed Plotly data/JS). You can open them directly in a browser (Chrome/Firefox/Edge) without starting Python.

Windows console note
- Printing emojis to the Windows console previously caused Unicode errors on some systems. The repository generators have been cleaned to avoid emoji characters in stdout. If you still see encoding errors, make sure your environment encoding supports UTF-8 or run PowerShell with UTF-8 enabled.

Troubleshooting
- If a generator fails with an import error, ensure the virtual environment is active and `pip install -r requirements.txt` has completed successfully.
- If a script can't locate the CSV, confirm `neural_data_dataframe.csv` exists in the repo root. The generators try the script's folder and then the parent folder.
- If HTML interactivity is slow for very large datasets, try running `Simple_3D_Visualizations.py` to produce lighter-weight outputs or reduce the number of plotted points in the generator.

Want different behavior?
- If you'd prefer generators to write outputs to a different directory, I can add a `--outdir` CLI flag to the generator scripts and update callers (small change).

Contact / next steps
- If you want, I can: add an `--outdir` option, produce `generated_images/README.md`, or consolidate and pin `requirements.txt`. Tell me which and I'll implement it.
