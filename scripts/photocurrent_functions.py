import os
import io
from networkx import display
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, gaussian_filter1d
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d
from typing import Dict, Iterable, Tuple, List, Union


def load_dta_folder(
    folder_path: str,
    header_marker: str = 'Pt\tT\tVf',
    required_numeric_cols: Tuple[str, ...] = ('T', 'Vf', 'Im'),
    extensions: Iterable[str] = ('.DTA',),
    encoding: str = 'utf-8',
    verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Walk a folder tree, parse text-based .DTA files, and build cleaned pandas DataFrames.

    Parameters
    ----------
    folder_path : str
        Root directory to search.
    header_marker : str, default 'Pt\\tT\\tVf'
        The exact header line that marks the start of the tabular data block.
    required_numeric_cols : tuple of str, default ('T','Vf','Im')
        Columns that must be present and convertible to numeric. Rows with NaN in these after conversion are dropped.
    extensions : iterable of str, default ('.DTA',)
        File extensions to include (case-insensitive).
    encoding : str, default 'utf-8'
        File encoding used to read text; errors are ignored.
    verbose : bool, default True
        If True, prints progress and warnings.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Dictionary mapping a sanitized base filename to its cleaned DataFrame.

    Notes
    -----
    - Any columns named like 'Unnamed: x' are dropped.
    - Lines that cannot be parsed by pandas are skipped (`on_bad_lines='skip'`).
    - If the header marker is not found in a file, that file is skipped.
    """
    dataframes: Dict[str, pd.DataFrame] = {}
    exts = tuple(ext.lower() for ext in extensions)

    if verbose:
        print(f"Starting to process files in: {folder_path}\n")

    for root, _, files in os.walk(folder_path):
        for file in files:
            if not file.lower().endswith(exts):
                continue

            file_path = os.path.join(root, file)
            if verbose:
                print(f"Processing: {file_path}")

            try:
                # --- Step 1: Read the entire file content ---
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    content = f.read()

                # --- Step 2: Locate the start of the data block ---
                header_start_pos = content.find(header_marker)

                if header_start_pos == -1:
                    if verbose:
                        print(f" -> Warning: Could not find header '{header_marker}' in {file}. Skipping.")
                    continue

                # Isolate the text block containing the header and all subsequent data
                data_block = content[header_start_pos:]

                # --- Step 3: Use pandas to read this block intelligently ---
                df = pd.read_csv(
                    io.StringIO(data_block),
                    sep='\t',
                    skipinitialspace=True,
                    on_bad_lines='skip')

                # --- Step 4: Clean the newly created DataFrame ---
                # Drop any extra "Unnamed" columns that might be created
                df = df.loc[:, ~df.columns.str.contains(r'^Unnamed', na=False)]

                # Force essential columns to be numeric right away
                for col in required_numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                # Drop rows where key data couldn't be converted
                existing_required = [c for c in required_numeric_cols if c in df.columns]
                if existing_required:
                    df.dropna(subset=existing_required, inplace=True)

                # Create a clean name and store the DataFrame
                df_name = os.path.splitext(file)[0].replace('-', '_').replace('.', '_')
                dataframes[df_name] = df

                if verbose:
                    print(f" -> Successfully created and cleaned DataFrame: '{df_name}'")

            except Exception as e:
                if verbose:
                    print(f" -> An error occurred while processing {file}: {e}")

    return dataframes

def group_by_potential(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    desired_potentials: List[float],
    round_decimals: int = 1,
    verbose: bool = True
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Group chrono data by rounded potential steps and prepare per-step DataFrames.

    Parameters
    ----------
    data : DataFrame or dict[str, DataFrame]
        Either a single DataFrame or a dict of DataFrames.
        Each DataFrame must have columns: 'T' (time, s), 'Im' (current, A), 'Vf' (potential, V).
    desired_potentials : list[float]
        Rounded potentials (e.g., [ -0.2, 0.0, 0.2 ]) to keep.
    round_decimals : int, default 1
        Number of decimals to round 'Vf' before grouping.
    verbose : bool, default True
        Print progress.

    Returns
    -------
    dict[str, dict[str, DataFrame]]
        { name: { "X.XV": DataFrame(Time_s, Current_uA), ... }, ... }
    """

    # Normalize input to dict form
    if isinstance(data, pd.DataFrame):
        dataframes = {"df": data}
    else:
        dataframes = data

    data_by_potential: Dict[str, Dict[str, pd.DataFrame]] = {}

    for name, df in dataframes.items():
        if verbose:
            print(f"Processing: '{name}'")
        try:
            if not set(["T", "Im", "Vf"]).issubset(df.columns):
                raise ValueError("Missing required columns: 'T', 'Im', 'Vf'.")

            # Round and filter
            df = df.copy()
            df["Vf_rounded"] = df["Vf"].round(decimals=round_decimals)
            df = df[df["Vf_rounded"].isin(desired_potentials)]

            # Nothing to do?
            if df.empty:
                if verbose:
                    print(" -> No rows match desired_potentials after rounding.")
                data_by_potential[name] = {}
                continue

            data_by_potential[name] = {}

            # Group and build per-step frames
            for potential_step, group_df in df.groupby("Vf_rounded"):
                # Fix -0.0
                if abs(potential_step) < 1e-6:
                    potential_step = 0.0

                final_df = pd.DataFrame({
                    "Time_s": group_df["T"].values,
                    "Current_uA": (group_df["Im"] * 1_000_000).values
                })

                # Reset time to start at zero for each step
                final_df["Time_s"] = final_df["Time_s"] - final_df["Time_s"].min()

                potential_key = f"{potential_step:.1f}V"
                data_by_potential[name][potential_key] = final_df

            if verbose:
                print(f" -> Successfully grouped into potentials: {list(data_by_potential[name].keys())}")

        except Exception as e:
            if verbose:
                print(f" -> An unexpected error occurred with '{name}': {e}")

    if verbose:
        print("\n----------------------------------------------------")
        print("Grouping complete.")
        print("The 'data_by_potential' dictionary is ready.")
        print("----------------------------------------------------")

    return data_by_potential



def process_photocurrent_robust(
    df_raw: pd.DataFrame,
    start_time_s: float = 30.0,
    # ---- Outlier control (Hampel / MAD) ----
    hampel_window_s: float = 1.0,
    hampel_nsigmas: float = 4.0,
    # ---- Baseline detection ----
    use_rolling_baseline: bool = True,
    baseline_window_s: float = 60.0,
    baseline_quantile: float = 0.10,
    baseline_smoothing_sigma: float = 15.0,
    # ---- Averaging (NEW: Plateau Range) ----
    # Average ONLY between these percentiles (e.g., 70th to 95th)
    # This cuts off the top 5% of data (the high outliers)
    plateau_range: tuple = (0.70, 0.95),
    smoothing_sigma: float = 1.0,
):
    # 0) Guardrails & prep
    if df_raw is None or df_raw.empty: return None, None
    df = df_raw.copy().sort_values('Time_s').drop_duplicates('Time_s')
    df = df[df['Time_s'] >= start_time_s].copy()
    if df.empty: return None, None
    df['Time_s'] -= start_time_s
    dt = np.median(np.diff(df['Time_s']))
    if dt <= 0: dt = df['Time_s'].diff().mean()
    fs = 1.0 / dt if dt > 0 else 1.0
    y_raw = df['Current_uA'].to_numpy().astype(float)

    # 1) FAST Hampel Despike
    win = max(3, int(hampel_window_s * fs))
    y_ser = pd.Series(y_raw)
    rol_med = y_ser.rolling(win, center=True, min_periods=1).median()
    rol_mad = (y_ser - rol_med).abs().rolling(win, center=True, min_periods=1).median()
    mask = (y_ser - rol_med).abs() > (hampel_nsigmas * 1.4826 * rol_mad)
    y_despiked = y_raw.copy()
    y_despiked[mask] = rol_med[mask]

    # 2) Baseline (Rolling Quantile)
    b_win = int(baseline_window_s * fs)
    baseline = pd.Series(y_despiked).rolling(b_win, center=True, min_periods=1).quantile(baseline_quantile)
    baseline = baseline.bfill().ffill().to_numpy()
    if baseline_smoothing_sigma > 0:
        baseline = gaussian_filter1d(baseline, sigma=baseline_smoothing_sigma)

    # 3) Subtract & Smooth
    photo = np.clip(y_despiked - baseline, 0, None)
    if smoothing_sigma > 0:
        photo = gaussian_filter1d(photo, sigma=smoothing_sigma)

    # 4) NEW: Robust Averaging (Trimmed Mean)
    # Calculate the lower and upper bounds of your plateau
    q_low = np.nanquantile(photo, plateau_range[0])
    q_high = np.nanquantile(photo, plateau_range[1])

    # Select points ONLY within this safe range
    mask_plateau = (photo >= q_low) & (photo <= q_high)

    # Calculate average of these safe points
    if mask_plateau.any():
        avg_photo = float(np.nanmean(photo[mask_plateau]))
    else:
        avg_photo = np.nan

    # 5) Package
    df_out = pd.DataFrame({
        'Time_s': df['Time_s'].to_numpy(),
        'Current_uA_raw': y_raw,
        'Current_uA_despiked': y_despiked,
        'Baseline': baseline,
        'Photocurrent_uA': photo
    })
    return df_out, avg_photo


# def process_photocurrent_robust(
#     df_raw: pd.DataFrame,
#     start_time_s: float = 30.0,
#     smoothing_sigma: float = 2.0,
#     # ---- Outlier control (Hampel / MAD) ----
#     hampel_window_s: float = 1.0,
#     hampel_nsigmas: float = 4.0,
#     # ---- New Rolling Baseline Parameters ----
#     use_rolling_baseline: bool = True,
#     baseline_window_s: float = 60.0,   # MUST be > than widest light pulse
#     baseline_quantile: float = 0.10,   # "Rides" the bottom 10% of noise
#     # ---- Old Anchor Baseline Parameters (kept for fallback) ----
#     min_dark_gap_s: float = 10.0,
#     anchor_quantile: float = 0.50,
#     anchor_halfwin_s: float = 2.0,
#     # ---- Common ----
#     baseline_smoothing_sigma: float = 2.0, # Increased slightly for rolling method
#     light_on_quantile: float = 0.80,
#     return_intermediates: bool = False
# ):
#     # ---------- 0) Guardrails & prep ----------
#     if df_raw is None or df_raw.empty: return None, None
#     df = df_raw.copy()
#     df = df.sort_values('Time_s').drop_duplicates(subset='Time_s')
#     df = df[df['Time_s'] >= start_time_s].copy()
#     if df.empty: return None, None
#     df['Time_s'] = df['Time_s'] - start_time_s
#     df.reset_index(drop=True, inplace=True)

#     dt = np.median(np.diff(df['Time_s'].to_numpy()))
#     if not np.isfinite(dt) or dt <= 0: dt = df['Time_s'].diff().mean()
#     if not np.isfinite(dt) or dt <= 0: return None, None
#     fs = 1.0 / dt
#     y_raw = df['Current_uA'].to_numpy().astype(float)

#     # ---------- 1) FAST Hampel Despike (Pandas Optimized) ----------
#     hampel_win = max(3, int(round(hampel_window_s * fs)))
#     y_series = pd.Series(y_raw)
    
#     rolling_med = y_series.rolling(window=hampel_win, center=True, min_periods=1).median()
#     rolling_mad = (y_series - rolling_med).abs().rolling(window=hampel_win, center=True, min_periods=1).median()
#     rolling_sigma = 1.4826 * rolling_mad
    
#     outlier_mask = (y_series - rolling_med).abs() > (hampel_nsigmas * rolling_sigma)
#     y_despiked = y_series.copy()
#     y_despiked[outlier_mask] = rolling_med[outlier_mask]
#     y_despiked = y_despiked.to_numpy()

#     # ---------- 2 & 3) Baseline Detection ----------
#     if use_rolling_baseline:
#         # --- NEW METHOD: Rolling Quantile ---
#         # 1. Define window width in samples
#         b_win = int(baseline_window_s * fs)
#         # 2. Calculate rolling quantile (the "floor" of the data)
#         baseline = pd.Series(y_despiked).rolling(window=b_win, center=True, min_periods=1).quantile(baseline_quantile)
#         # 3. Fill edges where rolling window doesn't have enough data
#         baseline = baseline.bfill().ffill()
#         # 4. Convert to numpy
#         baseline = baseline.to_numpy()
        
#     else:
#         # --- OLD METHOD: Anchors (kept as fallback) ---
#         y_for_anchors = gaussian_filter1d(y_despiked, sigma=max(0.5, baseline_smoothing_sigma))
#         min_peak_dist = max(1, int(round(min_dark_gap_s * fs)))
#         # Find dark regions (peaks in inverted signal)
#         dark_idxs, _ = find_peaks(-y_for_anchors, distance=min_peak_dist, height=None) # height removed for robustness
#         anchor_indices = np.unique(np.concatenate(([0], dark_idxs, [len(df) - 1])))

#         halfwin = max(1, int(round(anchor_halfwin_s * fs)))
#         anchor_vals = []
#         for idx in anchor_indices:
#             l, r = max(0, idx - halfwin), min(len(df), idx + halfwin + 1)
#             anchor_vals.append(np.quantile(y_despiked[l:r], anchor_quantile))
#         baseline = np.interp(np.arange(len(df)), anchor_indices, anchor_vals)

#     # Common: Final Baseline Smoothing
#     if baseline_smoothing_sigma > 0:
#         # If using rolling, we might need stronger smoothing to remove "steps"
#         actual_smooth = baseline_smoothing_sigma * (5.0 if use_rolling_baseline else 1.0)
#         baseline = gaussian_filter1d(baseline, sigma=actual_smooth)

#     # ---------- 4) Subtract & Clip ----------
#     photo = np.clip(y_despiked - baseline, 0, None)
#     if smoothing_sigma > 0:
#         photo = gaussian_filter1d(photo, sigma=smoothing_sigma)

#     # ---------- 5) Average Light-On ----------
#     thr = np.quantile(photo, light_on_quantile) if np.any(np.isfinite(photo)) else np.nan
#     avg_photo = float(np.nanmean(photo[photo >= thr])) if np.isfinite(thr) else np.nan

#     # ---------- 6) Package ----------
#     df_out = pd.DataFrame({
#         'Time_s': df['Time_s'].to_numpy(),
#         'Current_uA_raw': y_raw[:len(df)],
#         'Current_uA_despiked': y_despiked,
#         'Baseline': baseline,
#         'Photocurrent_uA': photo
#     })

#     if return_intermediates:
#         return df_out, avg_photo, {'fs': fs}
#     return df_out, avg_photo


####### PLOT FUNCTION ########
def align_window_and_pad(
    df: pd.DataFrame,
    time_col: str = "Time_s",
    y_col: str = "Photocurrent_uA",
    baseline_window_s: float = 5.0,     # baseline estimated from the first seconds of the trace
    threshold_frac: float = 0.20,       # onset threshold = baseline + frac*(max-baseline)
    hold_s: float = 0.30,               # require signal stays above/below threshold this long
    prepad_s: float = 2.0,              # prepend this much zero baseline BEFORE the first onset
    pad_to_xmax: float | None = 165.0   # if set, extend zeros to this time
) -> pd.DataFrame:
    """Return a new DataFrame aligned to first onset, with zero padding at start and after last pulse."""
    if df.empty:
        return df.copy()

    x = df[time_col].to_numpy()
    y = df[y_col].to_numpy().astype(float)

    # sampling rate (robust)
    dt = np.median(np.diff(x)) if len(x) > 1 else np.nan
    fs = 1.0 / dt if (np.isfinite(dt) and dt > 0) else 0.0
    hold_n = max(1, int(round(hold_s * fs))) if fs > 0 else 1

    # baseline & threshold
    base_mask = x <= (x.min() + baseline_window_s)
    baseline = np.median(y[base_mask]) if base_mask.any() else np.median(y[:max(3, len(y)//20)])
    thr = baseline + threshold_frac * max(1e-12, (y.max() - baseline))

    # find first onset (cross & hold above threshold)
    above = y >= thr
    onset_idx = None
    i = 0
    while i < len(above) - hold_n:
        if above[i] and above[i:i+hold_n].all():
            onset_idx = i
            break
        i += 1
    if onset_idx is None:
        # no onset—just start at 0 with optional pad_to_xmax zeros
        out = df[[time_col, y_col]].copy()
        out[time_col] = out[time_col] - out[time_col].min()  # start at 0
        if pad_to_xmax is not None and out[time_col].max() < pad_to_xmax:
            extra = pd.DataFrame({time_col: [pad_to_xmax], y_col: [0.0]})
            out = pd.concat([out, extra], ignore_index=True)
        return out

    # find last offset (end of the last contiguous "above" segment)
    idx_above = np.flatnonzero(above)
    if idx_above.size:
        # contiguous segments by run-length; take the last segment's end
        gaps = np.where(np.diff(idx_above) > 1)[0]
        last_start_pos = (gaps[-1] + 1) if gaps.size else 0
        last_segment = idx_above[last_start_pos:]
        last_off_idx = last_segment[-1]
    else:
        last_off_idx = onset_idx

    # time shift so onset occurs at t = prepad_s
    t0 = x[onset_idx] - prepad_s
    new_t = x - t0

    # force baseline zeros before onset (t < 0)
    y2 = y.copy()
    y2[new_t < 0] = 0.0
    new_t[new_t < 0] = 0.0  # clamp negative times to 0 so all traces start at 0

    # force zeros AFTER the last pulse (from last_off_idx onward if it goes below thr)
    # ensure we zero **after** the falling edge holds below threshold
    j = last_off_idx + 1
    while j < len(y2) - hold_n and not (~above[j:j+hold_n]).all():
        j += 1
    if j < len(y2):
        y2[j:] = 0.0

    out = pd.DataFrame({time_col: new_t, y_col: y2})

    # Optionally extend zeros to a common right limit
    if pad_to_xmax is not None:
        tmax = out[time_col].max()
        if np.isfinite(pad_to_xmax) and pad_to_xmax > tmax:
            out = pd.concat(
                [out, pd.DataFrame({time_col: [pad_to_xmax], y_col: [0.0]})],
                ignore_index=True
            )

    # Monotonic & dedup times for clean plotting
    out = out.sort_values(time_col).drop_duplicates(time_col).reset_index(drop=True)
    return out