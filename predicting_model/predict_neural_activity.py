"""
predict_neural_activity.py

Loads top-5 principal components CSV, extracts behavioral traces from the NWB file,
merges them (using timestamps when available), saves a merged CSV, and runs a
simple regression to predict PC1 from behavior.

Usage:
    python predict_neural_activity.py

Optional environment variable:
    CELEGANS_NWB_PATH -> full path to the NWB file

This script is intentionally robust to several NWB layout variants and will
attempt timestamp-based alignment first, then fallback to simple length-based
alignment (with trimming or NaNs) if timestamps aren't available.
"""

import os
import sys
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pynwb
except Exception as e:
    print("pynwb not found. Please install with `pip install pynwb`.")
    raise

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def get_nwb_file_path() -> Optional[str]:
    nwb_env_path = os.environ.get('CELEGANS_NWB_PATH')
    local_path = 'sub-2023-01-09-15-SWF702_ses-20230109_behavior+image+ophys.nwb'

    if nwb_env_path and os.path.exists(nwb_env_path):
        print(f"Loading NWB file from environment variable: {nwb_env_path}")
        return nwb_env_path
    if os.path.exists(local_path):
        print(f"Loading NWB file from local path: {local_path}")
        return local_path
    return None


def load_pcs(csv_path: str = 'principal_components_top5.csv') -> pd.DataFrame:
    # Try to load precomputed PCs CSV
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded PCs CSV with shape {df.shape}")
        return df

    # Fallback: attempt to compute PCs from available neural_data_dataframe.csv
    fallback = 'neural_data_dataframe.csv'
    if os.path.exists(fallback):
        print(f"'{csv_path}' not found — will compute PCs from '{fallback}' using PCA(5).")
        df_neural = pd.read_csv(fallback)
        # Expect rows=timepoints, cols=neurons (+ optional Time_minutes)
        # Drop non-numeric/time columns
        drop_cols = [c for c in df_neural.columns if not pd.api.types.is_numeric_dtype(df_neural[c])]
        df_numeric = df_neural.drop(columns=drop_cols) if drop_cols else df_neural.copy()

        # If there's a Time_minutes or timestamp column, drop it
        for tcol in ['Time_minutes', 'timestamp', 'time']:
            if tcol in df_numeric.columns:
                df_numeric = df_numeric.drop(columns=[tcol])

        # Compute PCA on neurons x time (samples are rows = timepoints)
        pca = PCA(n_components=5)
        pcs = pca.fit_transform(df_numeric.values)
        pc_cols = [f'PC{i+1}' for i in range(pcs.shape[1])]
        df_pcs = pd.DataFrame(pcs, columns=pc_cols)
        df_pcs.to_csv(csv_path, index=False)
        print(f"Computed PCs and saved to '{csv_path}' with shape {df_pcs.shape}")
        return df_pcs

    raise FileNotFoundError(f"Neural PCs CSV not found: {csv_path} and fallback '{fallback}' also missing.")


def _extract_neural_timestamps(nwbfile) -> Optional[np.ndarray]:
    # Try to find calcium activity module and get timestamps from the first TS
    if 'CalciumActivity' in nwbfile.processing:
        mod = nwbfile.processing['CalciumActivity']
        # iterate over data_interfaces
        for di in getattr(mod, 'data_interfaces', {}).values():
            if hasattr(di, 'timestamps'):
                try:
                    ts = np.asarray(di.timestamps)
                    if ts.size > 0:
                        return ts
                except Exception:
                    continue
    # fallback: search broadly for any TimeSeries with timestamps
    for m in nwbfile.processing.values():
        for di in getattr(m, 'data_interfaces', {}).values():
            if hasattr(di, 'timestamps'):
                try:
                    ts = np.asarray(di.timestamps)
                    if ts.size > 0:
                        return ts
                except Exception:
                    continue
    return None


def _get_behavior_time_series(behavior_mod, key: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """Attempt multiple strategies to obtain (data, timestamps, source_name) for a behavior key.
    Returns (data_array_or_None, timestamps_or_None, source_description_or_None).

    Important: require name-matching when selecting nested TimeSeries. Do not return
    the first arbitrary TimeSeries as a fallback — that caused duplicated columns.
    """
    di_map = getattr(behavior_mod, 'data_interfaces', None)

    # 1) direct mapping (good case)
    if di_map and key in di_map:
        di = di_map[key]
        # if this object directly exposes data/timestamps
        if hasattr(di, 'data'):
            data = np.asarray(di.data[:])
            ts = np.asarray(di.timestamps) if hasattr(di, 'timestamps') else None
            return data, ts, f"processing[{getattr(behavior_mod,'name', 'Behavior')}]/{key}"

    # 2) indexing lookup (some NWB layouts)
    try:
        obj = behavior_mod[key]
    except Exception:
        obj = None

    if obj is not None:
        # if obj itself is a TimeSeries-like
        if hasattr(obj, 'data'):
            data = np.asarray(obj.data[:])
            ts = np.asarray(obj.timestamps) if hasattr(obj, 'timestamps') else None
            return data, ts, f"processing[{getattr(behavior_mod,'name', 'Behavior')}]/{key}"

        # try nested access by key or attribute
        nested = None
        try:
            if hasattr(obj, '__getitem__') and key in obj:
                nested = obj[key]
        except Exception:
            nested = None
        if nested is None:
            nested = getattr(obj, key, None)
        if nested is not None and hasattr(nested, 'data'):
            data = np.asarray(nested.data[:])
            ts = np.asarray(nested.timestamps) if hasattr(nested, 'timestamps') else None
            return data, ts, f"processing[{getattr(behavior_mod,'name', 'Behavior')}]/{key} (nested)"

    # 3) search the di_map keys for matching names (case-insensitive)
    if di_map:
        for name, di in di_map.items():
            try:
                if key.lower() == name.lower() and hasattr(di, 'data'):
                    data = np.asarray(di.data[:])
                    ts = np.asarray(di.timestamps) if hasattr(di, 'timestamps') else None
                    return data, ts, f"processing[{getattr(behavior_mod,'name', 'Behavior')}]/{name}"
            except Exception:
                continue

    # 4) inspect each data_interface for nested time_series mapping (BehavioralTimeSeries)
    if di_map:
        for name, di in di_map.items():
            # prefer exact nested ts name matches
            try:
                ts_map = getattr(di, 'time_series', None)
                if ts_map:
                    # try exact match first
                    for ts_name, tsobj in ts_map.items():
                        if key.lower() == ts_name.lower() or key.lower() in ts_name.lower():
                            try:
                                data = np.asarray(tsobj.data[:])
                                ts = np.asarray(tsobj.timestamps) if hasattr(tsobj, 'timestamps') else None
                                return data, ts, f"processing[{getattr(behavior_mod,'name','Behavior')}]/{name}.time_series/{ts_name}"
                            except Exception:
                                continue
                    # no matching name — do NOT return the first ts by default
            except Exception:
                continue

    # 5) as last resort, inspect attributes of each data_interface but require a name match
    if di_map:
        for di in di_map.values():
            for attr in dir(di):
                # only consider attributes whose name contains key
                if key.lower() not in attr.lower():
                    continue
                try:
                    sub = getattr(di, attr)
                    if hasattr(sub, 'data'):
                        data = np.asarray(sub.data[:])
                        ts = np.asarray(sub.timestamps) if hasattr(sub, 'timestamps') else None
                        return data, ts, f"attribute match: {attr}"
                except Exception:
                    continue

    # not found
    return None, None, None


def load_and_merge_data(
    pcs_csv: str = 'principal_components_top5.csv',
    behaviors_to_extract: Optional[Dict[str, str]] = None,
    output_csv: str = 'predicting_model/merged_neural_behavior_data.csv',
) -> Optional[pd.DataFrame]:
    if behaviors_to_extract is None:
        behaviors_to_extract = {
            'Velocity': 'velocity',
            'HeadCurvature': 'head_curvature',
            'BodyCurvature': 'body_curvature',
            'Pumping': 'pumping',
        }

    try:
        df_neural = load_pcs(pcs_csv)
    except FileNotFoundError as e:
        print(e)
        return None

    nwb_path = get_nwb_file_path()
    if not nwb_path:
        print("NWB file not found. Set CELEGANS_NWB_PATH or place the NWB file in repo root.")
        return None

    with pynwb.NWBHDF5IO(nwb_path, 'r') as io:
        nwbfile = io.read()
        print("Loaded NWB file.")

        if 'Behavior' not in nwbfile.processing:
            print("Warning: 'Behavior' processing module not found in NWB. Attempting to proceed by searching.")

        behavior_mod = nwbfile.processing.get('Behavior', None)

        merged_df = df_neural.copy()

        # Try to get neural timestamps (from CalciumActivity) to use as alignment column
        neural_ts = _extract_neural_timestamps(nwbfile)
        if neural_ts is not None and neural_ts.size == len(merged_df):
            merged_df['timestamp'] = neural_ts
            print("Added neural timestamps to merged DataFrame for alignment.")
        elif neural_ts is not None:
            # if timestamp vector differs in length, still add it if equal to any PC length
            if neural_ts.size >= 1:
                print(f"Neural timestamps found but length ({neural_ts.size}) != PCs ({len(merged_df)}). Will try timestamp-based merge when possible.")
        else:
            print("No neural timestamps discovered in NWB; will attempt length-based alignment fallback.")

        for col_name, nwb_key in behaviors_to_extract.items():
            data_arr = None
            ts_arr = None
            source_desc = None

            if behavior_mod is not None:
                try:
                    data_arr, ts_arr, source_desc = _get_behavior_time_series(behavior_mod, nwb_key)
                except Exception as e:
                    print(f"  Warning: failed to extract '{nwb_key}' using helper: {e}")

            if data_arr is None:
                # last resort: try searching any processing module for time series with the key
                for pm in nwbfile.processing.values():
                    try:
                        d, t, src = _get_behavior_time_series(pm, nwb_key)
                        if d is not None:
                            data_arr, ts_arr, source_desc = d, t, src
                            break
                    except Exception:
                        continue

            if data_arr is None:
                print(f"  Warning: Could not locate behavior '{nwb_key}' in NWB. Skipping {col_name}.")
                merged_df[col_name] = np.nan
                continue

            # Report source if available
            if source_desc:
                print(f"  Extracted '{nwb_key}' from: {source_desc}")

            # Align using timestamps if possible
            if 'timestamp' in merged_df and ts_arr is not None and len(ts_arr) > 0:
                # build small DF for behavior and merge_asof
                df_beh = pd.DataFrame({'timestamp': ts_arr, col_name: data_arr})
                # ensure sorted
                merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
                df_beh = df_beh.sort_values('timestamp').reset_index(drop=True)
                try:
                    merged_df = pd.merge_asof(merged_df, df_beh, on='timestamp', direction='nearest')
                    print(f"  Merged '{col_name}' by timestamp (merge_asof).")
                except Exception as e:
                    print(f"  Warning: timestamp merge failed for {col_name}: {e}. Falling back to length-based alignment.")
                    # length-based fallback below

            # If the column not added yet, fallback to length-based alignment
            if col_name not in merged_df.columns:
                if len(data_arr) == len(merged_df):
                    merged_df[col_name] = data_arr
                    print(f"  Merged '{col_name}' by equal-length alignment.")
                elif len(data_arr) > len(merged_df):
                    # trim
                    merged_df[col_name] = data_arr[:len(merged_df)]
                    print(f"  Warning: '{col_name}' longer than PCs; trimming to match length.")
                else:
                    # shorter: create column with NaNs then fill first len(data_arr)
                    col = np.full(len(merged_df), np.nan)
                    col[:len(data_arr)] = data_arr
                    merged_df[col_name] = col
                    print(f"  Warning: '{col_name}' shorter than PCs; inserting and filling trailing NaNs.")

        # Save merged CSV
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        merged_df.to_csv(output_csv, index=False)
        print(f"Saved merged DataFrame to: {output_csv}")
        return merged_df


def run_prediction(merged_df: pd.DataFrame):
    # Features and target
    candidate_features = [c for c in merged_df.columns if c not in ['timestamp'] and not c.startswith('PC')]
    # prefer canonical names if present
    behavior_features = [c for c in ['Velocity', 'HeadCurvature', 'BodyCurvature', 'Pumping'] if c in merged_df.columns]
    if not behavior_features:
        # fallback to any numeric columns that are not PC columns, pick up to 4
        behavior_features = [c for c in candidate_features if pd.api.types.is_numeric_dtype(merged_df[c])]
        behavior_features = behavior_features[:4]

    target_candidates = [c for c in merged_df.columns if c.lower().startswith('pc')]
    if not target_candidates:
        raise ValueError('No PC columns found to predict (e.g. PC1).')
    target = 'PC1' if 'PC1' in merged_df.columns else target_candidates[0]

    essential = behavior_features + [target]
    data_clean = merged_df[essential].dropna()
    if data_clean.empty:
        raise ValueError('No complete cases available after dropping NaNs for essential columns.')

    X = data_clean[behavior_features].values
    y = data_clean[target].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.3, shuffle=False)
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    print('\n--- MODEL EVALUATION ---')
    print(f'R^2 on test set: {r2:.4f}')

    coeffs = pd.DataFrame({'feature': behavior_features, 'coef': model.coef_})
    coeffs = coeffs.sort_values('coef', ascending=False)
    print('\n--- COEFFICIENTS ---')
    print(coeffs.to_string(index=False))


def main():
    try:
        merged = load_and_merge_data()
        if merged is None:
            print('Failed to create merged dataset. Exiting.')
            return
        run_prediction(merged)
    except Exception as e:
        print(f"Fatal error: {e}")
        raise


if __name__ == '__main__':
    main()
