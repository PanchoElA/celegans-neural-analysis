# predicting_model

This folder contains a script to merge principal component (PC) data with behavioral traces
extracted from the NWB file and run a simple regression to predict neural activity (PC1) from behavior.

Files
- `predict_neural_activity.py` — main script. Loads `principal_components_top5.csv`, reads behavior from the NWB file,
  aligns data (using timestamps if found), saves `merged_neural_behavior_data.csv` to this folder, and runs a linear regression.

Usage
1. Make sure `pynwb` and other dependencies are installed (see repository `requirements.txt`).
2. Put the NWB file in the repository root or set `CELEGANS_NWB_PATH` to its full path.
3. Run:

```bash
python predicting_model/predict_neural_activity.py
```

Output
- `predicting_model/merged_neural_behavior_data.csv` — merged dataset for downstream modeling.

Notes
- The script tries timestamp-based alignment first. If timestamps are missing or inconsistent it falls back to length-based alignment.
- If behavior keys aren't found in the NWB under `processing['Behavior']`, the script attempts a broader search across processing modules.
