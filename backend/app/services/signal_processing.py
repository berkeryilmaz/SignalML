import io
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd


def ensure_2d_array(signals: Union[np.ndarray, List[List[float]], List[float]]) -> np.ndarray:
    """Ensures input is a 2D float32 numpy array of shape (num_samples, seq_len)."""
    arr = np.asarray(signals, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def first_n(signals: np.ndarray, n: int) -> np.ndarray:
    """Selects the first n samples along the sequence length."""
    arr = ensure_2d_array(signals)
    n = max(1, min(n, arr.shape[1]))
    return arr[:, :n]


def last_n(signals: np.ndarray, n: int) -> np.ndarray:
    """Selects the last n samples along the sequence length."""
    arr = ensure_2d_array(signals)
    n = max(1, min(n, arr.shape[1]))
    return arr[:, -n:]


def pad_signals(
    signals: np.ndarray,
    target_length: int,
    mode: str = "constant",
    constant_values: float = 0.0
) -> np.ndarray:
    """
    Pads signals to target_length.
    Supports mode: 'constant', 'edge', 'reflect'.
    """
    arr = ensure_2d_array(signals)
    num_samples, current_length = arr.shape
    if current_length >= target_length:
        return arr

    pad_width = target_length - current_length
    if mode == "constant":
        return np.pad(arr, ((0, 0), (0, pad_width)), mode="constant", constant_values=constant_values)
    elif mode in ("edge", "reflect"):
        return np.pad(arr, ((0, 0), (0, pad_width)), mode=mode)
    else:
        return np.pad(arr, ((0, 0), (0, pad_width)), mode="constant", constant_values=constant_values)


def truncate_signals(
    signals: np.ndarray,
    target_length: int,
    position: str = "end"
) -> np.ndarray:
    """
    Truncates signals to target_length.
    position can be: 'end' (keep start), 'start' (keep end), or 'center'.
    """
    arr = ensure_2d_array(signals)
    num_samples, current_length = arr.shape
    if current_length <= target_length:
        return arr

    diff = current_length - target_length
    if position == "end":
        return arr[:, :target_length]
    elif position == "start":
        return arr[:, diff:]
    elif position == "center":
        start_idx = diff // 2
        return arr[:, start_idx : start_idx + target_length]
    else:
        return arr[:, :target_length]


def to_fixed_length(
    signals: np.ndarray,
    target_length: int,
    pad_mode: str = "constant",
    pad_value: float = 0.0,
    truncate_pos: str = "end"
) -> np.ndarray:
    """Brings all signals to exact length target_length by either truncating or padding."""
    arr = ensure_2d_array(signals)
    current_length = arr.shape[1]
    if current_length == target_length:
        return arr
    elif current_length > target_length:
        return truncate_signals(arr, target_length, position=truncate_pos)
    else:
        return pad_signals(arr, target_length, mode=pad_mode, constant_values=pad_value)


def align_starts(signals: np.ndarray, threshold_ratio: float = 0.1) -> np.ndarray:
    """
    Aligns signal onsets: for each signal, finds the first sample where absolute amplitude
    exceeds threshold_ratio * max_abs, and shifts the signal so onset starts at index 0.
    """
    arr = ensure_2d_array(signals)
    num_samples, seq_len = arr.shape
    aligned = np.zeros_like(arr)

    for i in range(num_samples):
        sig = arr[i]
        max_val = np.max(np.abs(sig))
        if max_val == 0:
            aligned[i] = sig
            continue

        threshold = threshold_ratio * max_val
        indices = np.where(np.abs(sig) >= threshold)[0]
        if len(indices) == 0:
            aligned[i] = sig
        else:
            first_idx = indices[0]
            shifted = sig[first_idx:]
            aligned[i, :len(shifted)] = shifted

    return aligned


def align_ends(signals: np.ndarray, threshold_ratio: float = 0.1) -> np.ndarray:
    """
    Aligns signal offsets: finds last sample exceeding threshold and shifts so offset is at the end.
    """
    arr = ensure_2d_array(signals)
    num_samples, seq_len = arr.shape
    aligned = np.zeros_like(arr)

    for i in range(num_samples):
        sig = arr[i]
        max_val = np.max(np.abs(sig))
        if max_val == 0:
            aligned[i] = sig
            continue

        threshold = threshold_ratio * max_val
        indices = np.where(np.abs(sig) >= threshold)[0]
        if len(indices) == 0:
            aligned[i] = sig
        else:
            last_idx = indices[-1]
            trimmed = sig[:last_idx + 1]
            aligned[i, -len(trimmed):] = trimmed

    return aligned


def peak_asymmetric_window(
    signals: np.ndarray,
    left_m: int,
    right_k: int,
    mode: str = "max_abs"
) -> np.ndarray:
    """
    Extracts an asymmetric window [peak - left_m : peak + right_k + 1] around the peak.
    If the window extends outside signal boundaries, it is zero-padded.
    The resulting signal length is exactly left_m + right_k + 1.
    """
    arr = ensure_2d_array(signals)
    num_samples, seq_len = arr.shape
    window_length = left_m + right_k + 1
    result = np.zeros((num_samples, window_length), dtype=np.float32)

    for i in range(num_samples):
        sig = arr[i]
        if mode == "max_abs":
            peak_idx = int(np.argmax(np.abs(sig)))
        else:
            peak_idx = int(np.argmax(sig))

        start_idx = peak_idx - left_m
        end_idx = peak_idx + right_k + 1

        # Calculate overlap with original signal
        src_start = max(0, start_idx)
        src_end = min(seq_len, end_idx)

        dest_start = src_start - start_idx
        dest_end = dest_start + (src_end - src_start)

        result[i, dest_start:dest_end] = sig[src_start:src_end]

    return result


def peak_window(
    signals: np.ndarray,
    window_size: int,
    mode: str = "max_abs"
) -> np.ndarray:
    """Symmetric peak window around the maximum or max_abs peak."""
    half = window_size // 2
    left_m = half
    right_k = window_size - 1 - half
    return peak_asymmetric_window(signals, left_m, right_k, mode=mode)


def slice_signals(
    signals: np.ndarray,
    start: int = 0,
    stop: Optional[int] = None,
    stride: int = 1
) -> np.ndarray:
    """Slices signals with start, stop, and stride."""
    arr = ensure_2d_array(signals)
    stride = max(1, stride)
    return arr[:, start:stop:stride]


def normalize_minmax(
    signals: np.ndarray,
    feature_range: Tuple[float, float] = (0.0, 1.0),
    per_signal: bool = True
) -> np.ndarray:
    """Applies min-max scaling to feature_range."""
    arr = np.nan_to_num(ensure_2d_array(signals), copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = feature_range
    span = high - low

    if per_signal:
        mins = np.min(arr, axis=1, keepdims=True)
        maxs = np.max(arr, axis=1, keepdims=True)
        diff = maxs - mins
        diff[diff == 0] = 1.0
        return low + (arr - mins) / diff * span
    else:
        min_val = np.min(arr)
        max_val = np.max(arr)
        diff = max_val - min_val
        if diff == 0:
            diff = 1.0
        return low + (arr - min_val) / diff * span


def normalize_zscore(
    signals: np.ndarray,
    eps: float = 1e-8,
    per_signal: bool = True
) -> np.ndarray:
    """Applies Z-score standardization (mean=0, std=1)."""
    arr = np.nan_to_num(ensure_2d_array(signals), copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    if per_signal:
        means = np.mean(arr, axis=1, keepdims=True)
        stds = np.std(arr, axis=1, keepdims=True)
        stds[stds < eps] = 1.0
        return (arr - means) / stds
    else:
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        if std_val < eps:
            std_val = 1.0
        return (arr - mean_val) / std_val


def handle_missing(
    signals: np.ndarray,
    strategy: str = "interpolate",
    fill_value: float = 0.0
) -> np.ndarray:
    """
    Handles missing values (NaN or Inf).
    Strategies: 'interpolate', 'fill_value', 'fill_mean', 'drop'.
    """
    arr = ensure_2d_array(signals).copy()
    # Replace infinities with NaN
    arr[~np.isfinite(arr)] = np.nan

    if strategy == "drop":
        valid_rows = ~np.isnan(arr).any(axis=1)
        res = arr[valid_rows]
        return res if len(res) > 0 else np.zeros((1, arr.shape[1]), dtype=np.float32)

    if not np.isnan(arr).any():
        return arr

    if strategy == "fill_value":
        return np.nan_to_num(arr, nan=fill_value)

    if strategy == "fill_mean":
        for i in range(arr.shape[0]):
            sig = arr[i]
            nans = np.isnan(sig)
            if nans.all():
                arr[i] = fill_value
            elif nans.any():
                mean_val = np.nanmean(sig)
                arr[i, nans] = mean_val
        return arr

    if strategy == "interpolate":
        # Linear interpolation per signal
        for i in range(arr.shape[0]):
            sig = arr[i]
            nans = np.isnan(sig)
            if nans.all():
                arr[i] = fill_value
            elif nans.any():
                x = np.arange(len(sig))
                sig[nans] = np.interp(x[nans], x[~nans], sig[~nans])
                arr[i] = sig
        return arr

    return np.nan_to_num(arr, nan=fill_value)


def run_pipeline(signals: np.ndarray, steps: List[Dict[str, Any]]) -> np.ndarray:
    """
    Sequentially applies a list of transformation steps to the signals array.
    Each step is a dict: {"name": str, "params": dict}.
    """
    arr = ensure_2d_array(signals)

    for step in steps:
        name = step.get("name", "").lower()
        params = step.get("params", {})

        if name == "first_n":
            n = int(params.get("n", arr.shape[1]))
            arr = first_n(arr, n)
        elif name == "last_n":
            n = int(params.get("n", arr.shape[1]))
            arr = last_n(arr, n)
        elif name == "pad":
            target_len = int(params.get("target_length", arr.shape[1]))
            mode = params.get("mode", "constant")
            cval = float(params.get("constant_values", 0.0))
            arr = pad_signals(arr, target_len, mode=mode, constant_values=cval)
        elif name == "truncate":
            target_len = int(params.get("target_length", arr.shape[1]))
            pos = params.get("position", "end")
            arr = truncate_signals(arr, target_len, position=pos)
        elif name in ("to_fixed_length", "fixed_length"):
            target_len = int(params.get("target_length", 512))
            pad_mode = params.get("pad_mode", "constant")
            pad_val = float(params.get("pad_value", 0.0))
            trunc_pos = params.get("truncate_pos", "end")
            arr = to_fixed_length(arr, target_len, pad_mode=pad_mode, pad_value=pad_val, truncate_pos=trunc_pos)
        elif name == "align_starts":
            ratio = float(params.get("threshold_ratio", 0.1))
            arr = align_starts(arr, threshold_ratio=ratio)
        elif name == "align_ends":
            ratio = float(params.get("threshold_ratio", 0.1))
            arr = align_ends(arr, threshold_ratio=ratio)
        elif name == "peak_asymmetric_window":
            left_m = int(params.get("left_m", 100))
            right_k = int(params.get("right_k", 200))
            mode = params.get("mode", "max_abs")
            arr = peak_asymmetric_window(arr, left_m, right_k, mode=mode)
        elif name == "peak_window":
            win_size = int(params.get("window_size", 256))
            mode = params.get("mode", "max_abs")
            arr = peak_window(arr, win_size, mode=mode)
        elif name == "slice":
            start = int(params.get("start", 0))
            stop = int(params.get("stop")) if params.get("stop") is not None else None
            stride = int(params.get("stride", 1))
            arr = slice_signals(arr, start=start, stop=stop, stride=stride)
        elif name in ("normalize_minmax", "minmax"):
            frange = (float(params.get("min", 0.0)), float(params.get("max", 1.0)))
            per_sig = bool(params.get("per_signal", True))
            arr = normalize_minmax(arr, feature_range=frange, per_signal=per_sig)
        elif name in ("normalize_zscore", "zscore"):
            per_sig = bool(params.get("per_signal", True))
            arr = normalize_zscore(arr, per_signal=per_sig)
        elif name == "handle_missing":
            strat = params.get("strategy", "interpolate")
            fval = float(params.get("fill_value", 0.0))
            arr = handle_missing(arr, strategy=strat, fill_value=fval)

    return arr


def export_signals(
    signals: np.ndarray,
    labels: Optional[np.ndarray] = None,
    output_format: str = "csv",
    target_length: Optional[int] = None,
    separate_labels: bool = False
) -> Tuple[bytes, str, Dict[str, Any]]:
    """
    Exports signals verifying exact row count and exact N column count.
    No extraneous index columns are added.
    Returns:
        (file_bytes, mime_type, verification_metadata)
    """
    arr = ensure_2d_array(signals)
    if target_length is not None and target_length > 0:
        arr = to_fixed_length(arr, target_length)

    num_rows, num_cols = arr.shape
    verification = {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "format": output_format.lower(),
        "has_labels": labels is not None,
        "separate_labels": separate_labels
    }

    fmt = output_format.lower()
    if fmt in ("csv", "txt", "tsv"):
        sep = "\t" if fmt in ("txt", "tsv") else ","
        df = pd.DataFrame(arr)

        if labels is not None and not separate_labels:
            df["label"] = labels

        buf = io.StringIO()
        # header=False, index=False ensures pure data rows with no index column
        df.to_csv(buf, sep=sep, header=False, index=False)
        content_bytes = buf.getvalue().encode("utf-8")
        mime = "text/csv" if fmt == "csv" else "text/plain"
        return content_bytes, mime, verification

    elif fmt == "npy":
        buf = io.BytesIO()
        np.save(buf, arr)
        content_bytes = buf.getvalue()
        return content_bytes, "application/octet-stream", verification

    elif fmt == "npz":
        buf = io.BytesIO()
        save_dict = {"signals": arr}
        if labels is not None:
            save_dict["labels"] = labels
        np.savez_compressed(buf, **save_dict)
        content_bytes = buf.getvalue()
        return content_bytes, "application/octet-stream", verification

    else:
        raise ValueError(f"Unsupported export format: {output_format}")
