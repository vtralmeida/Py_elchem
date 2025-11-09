import os
import io
import pandas as pd
import numpy as np
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
    smoothing_sigma: float = 2.0,
    # ---- Outlier control (Hampel / MAD) ----
    hampel_window_s: float = 1.0,     # window length (seconds) for outlier detection
    hampel_nsigmas: float = 4.0,      # threshold multiplier on MAD
    # ---- Baseline detection ----
    min_dark_gap_s: float = 45.0,     # expected minimum time between dark segments
    anchor_quantile: float = 0.10,    # low-quantile for anchor value (robust against spikes)
    anchor_halfwin_s: float = 2.0,    # +/- seconds around each anchor to take quantile
    baseline_smoothing_sigma: float = 1.0,  # smooth baseline slightly
    # ---- Averaging ----
    light_on_quantile: float = 0.80,  # top-quantile to define “light on”
    return_intermediates: bool = False
):
    """
    Robust photocurrent processing with outlier rejection and a spike-resistant dynamic baseline.

    Expects columns:
        - 'Time_s' (seconds, monotonically increasing)
        - 'Current_uA' (microamps)

    Returns
    -------
    df_processed : DataFrame with columns:
        ['Time_s', 'Current_uA_raw', 'Current_uA_despiked', 'Baseline', 'Photocurrent_uA']
    avg_photocurrent : float (mean over the top quantile of 'Photocurrent_uA')
    (optionally) extras dict when return_intermediates=True
    """
    # ---------- 0) Guardrails & prep ----------
    if df_raw is None or df_raw.empty:
        return None, None

    # Work on a copy, ensure sorted time and no duplicates
    df = df_raw.copy()
    df = df.sort_values('Time_s').drop_duplicates(subset='Time_s')
    # Cut to analysis window and zero time
    df = df[df['Time_s'] >= start_time_s].copy()
    if df.empty:
        return None, None
    df['Time_s'] = df['Time_s'] - start_time_s
    df.reset_index(drop=True, inplace=True)

    # Sampling rate estimate (robust to jitter)
    dt = np.median(np.diff(df['Time_s'].to_numpy()))
    if not np.isfinite(dt) or dt <= 0:
        # Fallback to mean if median fails
        dt = df['Time_s'].diff().mean()
    if not np.isfinite(dt) or dt <= 0:
        return None, None
    fs = 1.0 / dt

    # Keep original (raw) current
    y_raw = df['Current_uA'].to_numpy().astype(float)

    # ---------- 1) Despike with a Hampel filter (MAD-based) ----------
    def hampel_despike(y, win_samples, n_sigmas=4.0):
        y = y.copy()
        half = int(max(1, win_samples // 2))
        for i in range(len(y)):
            left = max(0, i - half)
            right = min(len(y), i + half + 1)
            window = y[left:right]
            med = np.median(window)
            mad = np.median(np.abs(window - med))  # median absolute deviation
            # Consistent with normal dist: sigma ≈ 1.4826 * MAD
            sigma = 1.4826 * mad if mad > 0 else 0.0
            if sigma > 0 and np.abs(y[i] - med) > n_sigmas * sigma:
                # Replace outlier by local median
                y[i] = med
        return y

    hampel_win = max(3, int(round(hampel_window_s * fs)))
    y_despiked = hampel_despike(y_raw, hampel_win, hampel_nsigmas)

    # Optional tiny smoothing to help baseline anchors (does not blur steps much)
    y_for_anchors = gaussian_filter1d(y_despiked, sigma=max(0.5, baseline_smoothing_sigma))

    # ---------- 2) Find dark anchors (on *inverted* despiked/smoothed current) ----------
    min_peak_distance = max(1, int(round(min_dark_gap_s * fs)))
    inv = -y_for_anchors
    # dark_idxs, _ = find_peaks(inv, distance=min_peak_distance)
    print(min_peak_distance)
    dark_idxs, _ = find_peaks(inv, distance=100,height=-.10)

    # Always include start/end as anchors
    anchor_indices = np.unique(
        np.concatenate(([0], dark_idxs, [len(df) - 1]))
    )

    # Robust anchor values: low quantile within a small neighborhood (immune to spikes)
    halfwin = max(1, int(round(anchor_halfwin_s * fs)))
    anchor_values = []
    for idx in anchor_indices:
        left = max(0, idx - halfwin)
        right = min(len(df), idx + halfwin + 1)
        anchor_values.append(np.quantile(y_despiked[left:right], anchor_quantile))
    anchor_values = np.asarray(anchor_values, dtype=float)

    # ---------- 3) Build dynamic baseline (interp + slight smooth) ----------
    baseline = np.interp(np.arange(len(df)), anchor_indices, anchor_values)
    if baseline_smoothing_sigma and baseline_smoothing_sigma > 0:
        baseline = gaussian_filter1d(baseline, sigma=baseline_smoothing_sigma)

    # ---------- 4) Subtract baseline, clip at 0, optional smoothing ----------
    photo = y_despiked - baseline
    photo = np.clip(photo, 0, None)
    if smoothing_sigma and smoothing_sigma > 0:
        photo = gaussian_filter1d(photo, sigma=smoothing_sigma)

    # ---------- 5) Average photocurrent over “light on” region ----------
    thr = np.quantile(photo, light_on_quantile) if np.any(np.isfinite(photo)) else np.nan
    avg_photo = float(np.nanmean(photo[photo >= thr])) if np.isfinite(thr) else np.nan

    # ---------- 6) Package result ----------
    df_out = pd.DataFrame({
        'Time_s': df['Time_s'].to_numpy(),
        'Current_uA_raw': y_raw[:len(df)],          # raw (post time-cut)
        'Current_uA_despiked': y_despiked,
        'Baseline': baseline,
        'Photocurrent_uA': photo
    })

    if return_intermediates:
        extras = {
            'sampling_rate_hz': fs,
            'anchor_indices': anchor_indices,
            'anchor_values': anchor_values,
            'median_dt_s': float(dt)
        }
        return df_out, avg_photo, extras

    return df_out, avg_photo


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