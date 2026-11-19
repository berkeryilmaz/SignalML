import io
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d, binary_dilation, median_filter, grey_opening
from scipy.signal import detrend as scipy_detrend


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


def subtract_median(
    signals: np.ndarray,
    per_signal: bool = True
) -> np.ndarray:
    """Subtracts the median value from each signal (or across dataset) to shift baseline to 0."""
    arr = np.nan_to_num(ensure_2d_array(signals), copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    if per_signal:
        medians = np.median(arr, axis=1, keepdims=True)
        return (arr - medians).astype(np.float32)
    else:
        median_val = float(np.median(arr))
        return (arr - median_val).astype(np.float32)


def _estimate_mode_1d(sig: np.ndarray, bins: int = 100) -> float:
    """
    Estimates the baseline mode (most frequent amplitude) of a 1D signal.
    Uses exact value counts when the signal is discrete/quantized, or histogram
    dominant-bin sample average for continuous signals.
    """
    if sig.size == 0:
        return 0.0
    s_min = float(np.min(sig))
    s_max = float(np.max(sig))
    if s_max <= s_min:
        return s_min

    vals, counts = np.unique(sig, return_counts=True)
    max_count = int(np.max(counts))
    # If an exact value repeats frequently (quantized ADC / flat baseline) or bins <= 0
    if bins <= 0 or vals.size <= 32 or (max_count >= max(3, int(0.15 * sig.size))):
        return float(vals[int(np.argmax(counts))])

    n_bins = max(10, int(bins))
    hist, edges = np.histogram(sig, bins=n_bins, range=(s_min, s_max))
    k_max = int(np.argmax(hist))
    lo, hi = edges[k_max], edges[k_max + 1]
    if k_max == n_bins - 1:
        in_bin = sig[(sig >= lo) & (sig <= hi)]
    else:
        in_bin = sig[(sig >= lo) & (sig < hi)]
    if in_bin.size > 0:
        return float(np.mean(in_bin))
    return float(0.5 * (lo + hi))


def subtract_mode(
    signals: np.ndarray,
    bins: int = 100,
    per_signal: bool = True
) -> np.ndarray:
    """
    Subtracts the most frequent amplitude (mode) from each signal (or across dataset)
    to shift baseline to 0.
    """
    arr = np.nan_to_num(ensure_2d_array(signals), copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    if per_signal:
        modes = np.array([[_estimate_mode_1d(arr[i], bins=bins)] for i in range(arr.shape[0])], dtype=np.float32)
        return (arr - modes).astype(np.float32)
    else:
        mode_val = _estimate_mode_1d(arr.ravel(), bins=bins)
        return (arr - mode_val).astype(np.float32)


def _estimate_gaussian_peak_1d(sig: np.ndarray, bins: int = 100, fit_window_bins: int = 5) -> float:
    """
    Builds an amplitude histogram of `sig`, isolates the dominant baseline lobe around
    the histogram maximum, fits a 1D Gaussian curve A * exp(-(x - mu)^2 / (2 * sigma^2)),
    and returns the fitted Gaussian peak center `mu` as the baseline offset.
    """
    if sig.size == 0:
        return 0.0
    s_min = float(np.min(sig))
    s_max = float(np.max(sig))
    if s_max <= s_min:
        return s_min

    n_bins = max(16, int(bins))
    hist, edges = np.histogram(sig, bins=n_bins, range=(s_min, s_max))
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = float(edges[1] - edges[0])

    k_max = int(np.argmax(hist))
    win = max(2, int(fit_window_bins))
    start_k = max(0, k_max - win)
    end_k = min(n_bins, k_max + win + 1)

    x_loc = centers[start_k:end_k]
    y_loc = hist[start_k:end_k].astype(np.float64)

    # Keep bins that belong to the main baseline lobe (>10% of peak height)
    mask = y_loc >= (0.10 * float(hist[k_max]))
    if np.sum(mask) >= 3:
        x_fit = x_loc[mask]
        y_fit = y_loc[mask]
    else:
        x_fit = x_loc
        y_fit = y_loc

    w_sum = float(np.sum(y_fit))
    if w_sum <= 0:
        return float(centers[k_max])

    # Initial moment estimates in the peak window
    mu0 = float(np.sum(x_fit * y_fit) / w_sum)
    var0 = float(np.sum(y_fit * (x_fit - mu0) ** 2) / w_sum)
    sigma0 = max(np.sqrt(max(var0, 0.0)), bin_width * 0.5)
    a0 = float(hist[k_max])

    # Weighted log-parabolic Gaussian estimate as a fast, exact closed-form initializer
    pos = y_fit > 0
    if np.sum(pos) >= 3:
        xp = x_fit[pos]
        yp = y_fit[pos]
        try:
            coeffs = np.polyfit(xp, np.log(yp), deg=2, w=yp)
            if coeffs[0] < -1e-12:
                mu_quad = float(-coeffs[1] / (2.0 * coeffs[0]))
                if float(x_fit[0]) <= mu_quad <= float(x_fit[-1]):
                    mu0 = mu_quad
        except Exception:
            pass

    # Nonlinear least-squares Gaussian refinement via scipy.optimize.curve_fit
    if x_fit.size >= 3:
        try:
            from scipy.optimize import curve_fit

            def _gauss(x, a, mu, sigma):
                return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

            lo_bound = [0.0, float(x_fit[0]), bin_width * 0.1]
            hi_bound = [float(hist[k_max]) * 2.5 + 1.0, float(x_fit[-1]), (s_max - s_min) + bin_width]
            mu0_clamped = min(max(mu0, lo_bound[1]), hi_bound[1])
            sigma0_clamped = min(max(sigma0, lo_bound[2]), hi_bound[2])

            popt, _ = curve_fit(
                _gauss,
                x_fit,
                y_fit,
                p0=[a0, mu0_clamped, sigma0_clamped],
                bounds=(lo_bound, hi_bound),
                maxfev=400
            )
            mu_fit = float(popt[1])
            if np.isfinite(mu_fit):
                return mu_fit
        except Exception:
            pass

    return mu0


def subtract_gaussian_peak(
    signals: np.ndarray,
    bins: int = 100,
    fit_window_bins: int = 5,
    per_signal: bool = True
) -> np.ndarray:
    """
    Computes an amplitude histogram, fits a Gaussian curve around the dominant peak,
    and subtracts the Gaussian peak center (mu) so the baseline is centered at 0.
    """
    arr = np.nan_to_num(ensure_2d_array(signals), copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    if per_signal:
        peaks = np.array(
            [[_estimate_gaussian_peak_1d(arr[i], bins=bins, fit_window_bins=fit_window_bins)] for i in range(arr.shape[0])],
            dtype=np.float32
        )
        return (arr - peaks).astype(np.float32)
    else:
        peak_val = _estimate_gaussian_peak_1d(arr.ravel(), bins=bins, fit_window_bins=fit_window_bins)
        return (arr - peak_val).astype(np.float32)


def subtract_mean(
    signals: np.ndarray,
    per_signal: bool = True
) -> np.ndarray:
    """
    Subtracts the arithmetic mean (DC offset) from each signal (or across dataset)
    to shift baseline to 0.
    """
    arr = np.nan_to_num(ensure_2d_array(signals), copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.shape[1] == 0:
        return arr.astype(np.float32)
    if per_signal:
        means = np.mean(arr, axis=1, keepdims=True)
        return (arr - means).astype(np.float32)
    else:
        mean_val = float(np.mean(arr))
        return (arr - mean_val).astype(np.float32)


def subtract_first_n_median(
    signals: np.ndarray,
    n: int = 50,
    per_signal: bool = True
) -> np.ndarray:
    """
    Computes the median of the first n samples and subtracts it from the entire signal.
    Preserves the full signal length.
    """
    arr = np.nan_to_num(ensure_2d_array(signals), copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.shape[1] == 0:
        return arr.astype(np.float32)
    n_clamped = max(1, min(int(n), arr.shape[1]))
    sub = arr[:, :n_clamped]
    if per_signal:
        base = np.median(sub, axis=1, keepdims=True)
        return (arr - base).astype(np.float32)
    else:
        base_val = float(np.median(sub))
        return (arr - base_val).astype(np.float32)


def subtract_first_n_mean(
    signals: np.ndarray,
    n: int = 50,
    per_signal: bool = True
) -> np.ndarray:
    """
    Computes the arithmetic mean of the first n samples and subtracts it from the entire signal.
    Preserves the full signal length.
    """
    arr = np.nan_to_num(ensure_2d_array(signals), copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.shape[1] == 0:
        return arr.astype(np.float32)
    n_clamped = max(1, min(int(n), arr.shape[1]))
    sub = arr[:, :n_clamped]
    if per_signal:
        base = np.mean(sub, axis=1, keepdims=True)
        return (arr - base).astype(np.float32)
    else:
        base_val = float(np.mean(sub))
        return (arr - base_val).astype(np.float32)


def subtract_last_n_median(
    signals: np.ndarray,
    n: int = 50,
    stat: str = "median",
    per_signal: bool = True
) -> np.ndarray:
    """
    Computes baseline from the last n samples using median or mean,
    and subtracts it from the entire signal. Preserves full signal length.
    """
    arr = np.nan_to_num(ensure_2d_array(signals), copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.shape[1] == 0:
        return arr.astype(np.float32)
    n_clamped = max(1, min(int(n), arr.shape[1]))
    sub = arr[:, -n_clamped:]
    use_mean = str(stat).strip().lower() == "mean"
    if per_signal:
        base = np.mean(sub, axis=1, keepdims=True) if use_mean else np.median(sub, axis=1, keepdims=True)
        return (arr - base).astype(np.float32)
    else:
        base_val = float(np.mean(sub)) if use_mean else float(np.median(sub))
        return (arr - base_val).astype(np.float32)


def subtract_linear_baseline(
    signals: np.ndarray,
    mode: str = "edge",
    n_edge: int = 50,
    stat: str = "median"
) -> np.ndarray:
    """
    Removes a linear baseline ramp/drift from each signal:
    - mode in ('ols', 'detrend', 'linear_fit'): fits a least-squares linear trend across the full signal.
    - mode == 'edge' (default): computes baseline at start and end edge windows and subtracts the connecting ramp.
    """
    arr = np.nan_to_num(ensure_2d_array(signals), copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    num_samples, seq_len = arr.shape
    if seq_len == 0:
        return arr.astype(np.float32)

    m = (mode or "edge").lower()
    if m in ("ols", "detrend", "linear_fit"):
        if seq_len < 2:
            return (arr - arr).astype(np.float32)
        try:
            return scipy_detrend(arr, axis=1, type="linear").astype(np.float32)
        except Exception:
            t = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)
            t_mean = 0.5
            t_dev = t - t_mean
            var_t = float(np.sum(t_dev ** 2))
            if var_t <= 1e-12:
                return (arr - arr).astype(np.float32)
            slope = np.sum(arr * t_dev[None, :], axis=1, keepdims=True) / var_t
            intercept = np.mean(arr, axis=1, keepdims=True) - slope * t_mean
            trend = intercept + slope * t[None, :]
            return (arr - trend).astype(np.float32)
    else:
        k = max(1, min(int(n_edge), seq_len))
        use_mean = str(stat).strip().lower() == "mean"
        if use_mean:
            y_start = np.mean(arr[:, :k], axis=1, keepdims=True)
            y_end = np.mean(arr[:, -k:], axis=1, keepdims=True)
        else:
            y_start = np.median(arr[:, :k], axis=1, keepdims=True)
            y_end = np.median(arr[:, -k:], axis=1, keepdims=True)

        if seq_len == 1:
            return (arr - y_start).astype(np.float32)

        t_ramp = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)[None, :]
        ramp = y_start + (y_end - y_start) * t_ramp
        return (arr - ramp).astype(np.float32)


def subtract_polynomial_baseline(
    signals: np.ndarray,
    degree: int = 2,
    exclude_peak_sigma: float = 2.0
) -> np.ndarray:
    """
    Fits and subtracts a polynomial baseline of degree `degree` per signal along
    x = np.linspace(-1.0, 1.0, seq_len, dtype=np.float64).
    Uses iterative sigma clipping when exclude_peak_sigma > 0 to prevent large peaks
    from distorting the baseline fit.
    """
    arr = np.nan_to_num(ensure_2d_array(signals), copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    num_samples, seq_len = arr.shape
    if seq_len <= 2:
        return subtract_median(arr, per_signal=True)

    deg = max(1, min(int(degree), 8, seq_len - 1))
    x = np.linspace(-1.0, 1.0, seq_len, dtype=np.float64)
    out = np.empty((num_samples, seq_len), dtype=np.float32)

    for i in range(num_samples):
        sig = arr[i].astype(np.float64)
        mask = np.ones(seq_len, dtype=bool)

        if exclude_peak_sigma > 0:
            for _ in range(2):
                try:
                    coeffs = np.polyfit(x[mask], sig[mask], deg)
                    baseline = np.polyval(coeffs, x)
                except Exception:
                    break
                res = sig - baseline
                res_med = float(np.median(res))
                mad_std = 1.4826 * float(np.median(np.abs(res - res_med)))
                sigma = mad_std if mad_std > 1e-8 else float(np.std(res))
                if sigma <= 1e-8:
                    break
                new_mask = np.abs(res - res_med) <= float(exclude_peak_sigma) * sigma
                if int(np.sum(new_mask)) >= deg + 2:
                    mask = new_mask
                else:
                    break

        try:
            coeffs = np.polyfit(x[mask], sig[mask], deg)
            baseline = np.polyval(coeffs, x)
        except Exception:
            try:
                coeffs = np.polyfit(x, sig, deg)
                baseline = np.polyval(coeffs, x)
            except Exception:
                baseline = float(np.median(sig))

        out[i] = (sig - baseline).astype(np.float32)

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def subtract_rolling_baseline(
    signals: np.ndarray,
    window_size: int = 101,
    method: str = "median"
) -> np.ndarray:
    """
    Estimates and subtracts a local moving baseline per signal using a rolling filter
    (median, uniform mean, or morphological opening).
    """
    arr = np.nan_to_num(ensure_2d_array(signals), copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    seq_len = arr.shape[1]
    if seq_len == 0:
        return arr.astype(np.float32)
    if seq_len <= 2:
        return subtract_median(arr, per_signal=True)

    win = max(3, min(int(window_size), max(3, seq_len)))
    m = (method or "median").lower()

    if m == "mean":
        win = min(win, seq_len)
        baseline = uniform_filter1d(arr, size=win, axis=1, mode="nearest")
    elif m in ("min_opening", "opening", "tophat"):
        win = min(win, seq_len)
        baseline = grey_opening(arr, size=(1, win), mode="nearest")
    else:
        if win % 2 == 0:
            win += 1
        if win > seq_len:
            win = seq_len if seq_len % 2 == 1 else max(1, seq_len - 1)
        baseline = median_filter(arr, size=(1, win), mode="nearest")

    return (arr - baseline).astype(np.float32)


def reverse_signals(
    signals: np.ndarray,
    mode: str = "polarity"
) -> np.ndarray:
    """
    Reverses signals according to `mode`:
    - 'polarity' (default): Inverts signal amplitude (x' = -x), preserving 0 baseline.
    - 'time': Reverses sample order along the time axis (x'[t] = x[N - 1 - t]).
    - 'both': Inverts polarity and reverses time axis (-x[:, ::-1]).
    """
    arr = np.nan_to_num(ensure_2d_array(signals), copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    m = (mode or "polarity").lower()
    if m in ("time", "temporal", "horizontal"):
        return np.ascontiguousarray(arr[:, ::-1], dtype=np.float32)
    elif m == "both":
        return np.ascontiguousarray(-arr[:, ::-1], dtype=np.float32)
    else:
        return (-arr).astype(np.float32)


def scale_min(
    signals: np.ndarray,
    mode: str = "divide_abs_min",
    per_signal: bool = True,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Scales signals using their minimum value:
    - 'divide_abs_min' (default): x' = x / |min(x)| (preserves 0 baseline, scales negative peak to -1).
    - 'divide_min': x' = x / min(x) (preserves 0 baseline, scales negative minimum to +1).
    - 'subtract_min': x' = x - min(x) (shifts minimum value to 0).
    """
    arr = np.nan_to_num(ensure_2d_array(signals), copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    m = (mode or "divide_abs_min").lower()
    if per_signal:
        mins = np.min(arr, axis=1, keepdims=True)
        if m == "subtract_min":
            return (arr - mins).astype(np.float32)
        elif m == "divide_min":
            denom = np.where(np.abs(mins) < eps, 1.0, mins)
            return (arr / denom).astype(np.float32)
        else:
            denom = np.where(np.abs(mins) < eps, 1.0, np.abs(mins))
            return (arr / denom).astype(np.float32)
    else:
        min_val = float(np.min(arr))
        if m == "subtract_min":
            return (arr - min_val).astype(np.float32)
        elif m == "divide_min":
            denom = min_val if abs(min_val) >= eps else 1.0
            return (arr / denom).astype(np.float32)
        else:
            denom = abs(min_val) if abs(min_val) >= eps else 1.0
            return (arr / denom).astype(np.float32)


def scale_max(
    signals: np.ndarray,
    mode: str = "divide_max",
    per_signal: bool = True,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Scales signals using their maximum value while preserving 0 baseline:
    - 'divide_max' (default): x' = x / max(x) (scales positive maximum to +1.0).
    - 'divide_max_abs': x' = x / max(|x|) (scales largest magnitude to +/-1.0).
    """
    arr = np.nan_to_num(ensure_2d_array(signals), copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    m = (mode or "divide_max").lower()
    if per_signal:
        if m in ("divide_max_abs", "max_abs", "maxabs"):
            maxs = np.max(np.abs(arr), axis=1, keepdims=True)
        else:
            maxs = np.max(arr, axis=1, keepdims=True)
        denom = np.where(np.abs(maxs) < eps, 1.0, maxs)
        return (arr / denom).astype(np.float32)
    else:
        if m in ("divide_max_abs", "max_abs", "maxabs"):
            max_val = float(np.max(np.abs(arr)))
        else:
            max_val = float(np.max(arr))
        denom = max_val if abs(max_val) >= eps else 1.0
        return (arr / denom).astype(np.float32)


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
            per_sig_raw = params.get("per_signal", True)
            per_sig = per_sig_raw.lower() in ("true", "1", "yes") if isinstance(per_sig_raw, str) else bool(per_sig_raw)
            arr = normalize_zscore(arr, per_signal=per_sig)
        elif name in ("subtract_median", "baseline_median", "median_baseline"):
            per_sig_raw = params.get("per_signal", True)
            per_sig = per_sig_raw.lower() in ("true", "1", "yes") if isinstance(per_sig_raw, str) else bool(per_sig_raw)
            arr = subtract_median(arr, per_signal=per_sig)
        elif name in ("subtract_mode", "baseline_mode", "mode_baseline"):
            bins = int(params.get("bins", 100))
            per_sig_raw = params.get("per_signal", True)
            per_sig = per_sig_raw.lower() in ("true", "1", "yes") if isinstance(per_sig_raw, str) else bool(per_sig_raw)
            arr = subtract_mode(arr, bins=bins, per_signal=per_sig)
        elif name in ("subtract_gaussian_peak", "gaussian_baseline", "baseline_gaussian", "histogram_gaussian_baseline"):
            bins = int(params.get("bins", 100))
            fit_win = int(params.get("fit_window_bins", 5))
            per_sig_raw = params.get("per_signal", True)
            per_sig = per_sig_raw.lower() in ("true", "1", "yes") if isinstance(per_sig_raw, str) else bool(per_sig_raw)
            arr = subtract_gaussian_peak(arr, bins=bins, fit_window_bins=fit_win, per_signal=per_sig)
        elif name in ("subtract_mean", "baseline_mean", "mean_baseline", "dc_offset"):
            per_sig_raw = params.get("per_signal", True)
            per_sig = per_sig_raw.lower() in ("true", "1", "yes") if isinstance(per_sig_raw, str) else bool(per_sig_raw)
            arr = subtract_mean(arr, per_signal=per_sig)
        elif name in ("subtract_first_n_median", "first_n_median", "baseline_first_n_median", "pretrigger_median"):
            n = int(params.get("n", 50))
            per_sig_raw = params.get("per_signal", True)
            per_sig = per_sig_raw.lower() in ("true", "1", "yes") if isinstance(per_sig_raw, str) else bool(per_sig_raw)
            arr = subtract_first_n_median(arr, n=n, per_signal=per_sig)
        elif name in ("subtract_first_n_mean", "first_n_mean", "baseline_first_n_mean", "pretrigger_mean"):
            n = int(params.get("n", 50))
            per_sig_raw = params.get("per_signal", True)
            per_sig = per_sig_raw.lower() in ("true", "1", "yes") if isinstance(per_sig_raw, str) else bool(per_sig_raw)
            arr = subtract_first_n_mean(arr, n=n, per_signal=per_sig)
        elif name in ("subtract_last_n_median", "last_n_median", "baseline_last_n_median", "subtract_last_n_baseline", "tail_baseline"):
            n = int(params.get("n", 50))
            stat = str(params.get("stat", "median"))
            per_sig_raw = params.get("per_signal", True)
            per_sig = per_sig_raw.lower() in ("true", "1", "yes") if isinstance(per_sig_raw, str) else bool(per_sig_raw)
            arr = subtract_last_n_median(arr, n=n, stat=stat, per_signal=per_sig)
        elif name in ("subtract_linear_baseline", "linear_baseline", "baseline_linear", "linear_detrend", "detrend_linear"):
            mode = str(params.get("mode", "edge"))
            n_edge = int(params.get("n_edge", 50))
            stat = str(params.get("stat", "median"))
            arr = subtract_linear_baseline(arr, mode=mode, n_edge=n_edge, stat=stat)
        elif name in ("subtract_polynomial_baseline", "polynomial_baseline", "baseline_polynomial", "poly_baseline"):
            degree = int(params.get("degree", 2))
            exclude_peak_sigma = float(params.get("exclude_peak_sigma", 2.0))
            arr = subtract_polynomial_baseline(arr, degree=degree, exclude_peak_sigma=exclude_peak_sigma)
        elif name in ("subtract_rolling_baseline", "rolling_baseline", "baseline_rolling", "moving_median_baseline"):
            window_size = int(params.get("window_size", 101))
            method = str(params.get("method", "median"))
            arr = subtract_rolling_baseline(arr, window_size=window_size, method=method)
        elif name in ("reverse", "reverse_signal", "reverse_signals", "invert"):
            mode = str(params.get("mode", "polarity"))
            arr = reverse_signals(arr, mode=mode)
        elif name in ("scale_min", "min_scaler"):
            mode = str(params.get("mode", "divide_abs_min"))
            per_sig_raw = params.get("per_signal", True)
            per_sig = per_sig_raw.lower() in ("true", "1", "yes") if isinstance(per_sig_raw, str) else bool(per_sig_raw)
            arr = scale_min(arr, mode=mode, per_signal=per_sig)
        elif name in ("scale_max", "max_scaler"):
            mode = str(params.get("mode", "divide_max"))
            per_sig_raw = params.get("per_signal", True)
            per_sig = per_sig_raw.lower() in ("true", "1", "yes") if isinstance(per_sig_raw, str) else bool(per_sig_raw)
            arr = scale_max(arr, mode=mode, per_signal=per_sig)
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
    separate_labels: bool = False,
    include_id: bool = False,
    sample_ids: Optional[List[str]] = None,
    extra_columns_dict: Optional[Dict[str, List[str]]] = None
) -> Tuple[bytes, str, Dict[str, Any]]:
    """
    Exports signals verifying exact row count and exact N column count.
    No extraneous index columns are added.
    When include_id=True or extra_columns_dict is provided, prepends id and extra columns with a header row.
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
        "separate_labels": separate_labels,
        "include_id": include_id,
        "extra_columns": list(extra_columns_dict.keys()) if extra_columns_dict else []
    }

    fmt = output_format.lower()
    if fmt in ("csv", "txt", "tsv"):
        sep = "\t" if fmt in ("txt", "tsv") else ","

        if include_id or (extra_columns_dict and len(extra_columns_dict) > 0):
            df_dict: Dict[str, Any] = {}
            if include_id:
                if sample_ids is not None and len(sample_ids) >= num_rows:
                    df_dict["id"] = sample_ids[:num_rows]
                else:
                    df_dict["id"] = [str(i) for i in range(num_rows)]

            if extra_columns_dict:
                for col_name, col_vals in extra_columns_dict.items():
                    if len(col_vals) >= num_rows:
                        df_dict[col_name] = col_vals[:num_rows]
                    else:
                        df_dict[col_name] = col_vals + [""] * (num_rows - len(col_vals))

            for j in range(num_cols):
                df_dict[f"s{j}"] = arr[:, j]

            if labels is not None and not separate_labels:
                df_dict["label"] = labels

            df = pd.DataFrame(df_dict)
            buf = io.StringIO()
            df.to_csv(buf, sep=sep, header=True, index=False)
            content_bytes = buf.getvalue().encode("utf-8")
            mime = "text/csv" if fmt == "csv" else "text/plain"
            return content_bytes, mime, verification
        else:
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
        if include_id and sample_ids is not None:
            save_dict["sample_ids"] = np.array(sample_ids[:num_rows], dtype=str)
        if extra_columns_dict:
            import json
            save_dict["extra_columns_json"] = np.array(json.dumps(extra_columns_dict))
        np.savez_compressed(buf, **save_dict)
        content_bytes = buf.getvalue()
        return content_bytes, "application/octet-stream", verification

    else:
        raise ValueError(f"Unsupported export format: {output_format}")


def compute_regional_labels(
    base_signal: np.ndarray,
    threshold_sigma: float = 1.5,
    existing_region_labels: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Computes or formats a pointwise regional event mask of shape (N, seq_len, 1).
    
    If existing_region_labels is provided, it is adapted to shape (N, seq_len, 1).
    Otherwise, computes an automatic regional event mask per signal where amplitude
    exceeds mean + threshold_sigma * std, with morphological dilation so contiguous
    events form solid regions marked 1.0, and background 0.0.
    """
    arr = np.asarray(base_signal, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim == 3:
        sig = arr[:, :, 0]
    else:
        sig = arr

    N, seq_len = sig.shape

    if existing_region_labels is not None:
        reg = np.asarray(existing_region_labels, dtype=np.float32)
        if reg.ndim == 1:
            reg = reg.reshape(1, -1)
        if reg.ndim == 3:
            reg = reg[:, :, 0]
        # Align rows and seq_len
        if reg.shape[0] != N or reg.shape[1] != seq_len:
            out_reg = np.zeros((N, seq_len), dtype=np.float32)
            n_copy = min(N, reg.shape[0])
            l_copy = min(seq_len, reg.shape[1])
            out_reg[:n_copy, :l_copy] = reg[:n_copy, :l_copy]
            reg = out_reg
        return np.expand_dims(reg, axis=-1).astype(np.float32)

    # Automatic event detection
    means = np.mean(sig, axis=1, keepdims=True)
    stds = np.std(sig, axis=1, keepdims=True)
    stds = np.where(stds < 1e-8, 1.0, stds)

    # Active points exceeding mean + threshold_sigma * std
    threshold = means + float(threshold_sigma) * stds
    raw_mask = (sig >= threshold)

    # Baseline expansion / dilation window (radius ~ seq_len // 64, min 2)
    radius = max(2, seq_len // 64)
    structure = np.ones((1, 2 * radius + 1), dtype=bool)
    dilated_mask = binary_dilation(raw_mask, structure=structure)

    mask_float = dilated_mask.astype(np.float32)
    return np.expand_dims(mask_float, axis=-1)


def build_signal_channels(
    base_signal: np.ndarray,
    channels: Optional[List[str]] = None,
    normalize_channels: bool = False,
    region_threshold: float = 2.0,
    existing_region_labels: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Constructs multi-channel 1D signal array from a base 1D/3D signal.

    Parameters:
        base_signal: shape (N, seq_len) or (N, seq_len, C).
        channels: list of requested channel IDs. Supported:
            - 'raw': Raw primary 1D signal x(t)
            - 'raw_all': All channels from base_signal
            - 'derivative_1' ('d1', 'dx_dt', 'first_derivative'): 1st derivative along axis 1 (np.gradient)
            - 'derivative_2' ('d2', 'd2x_dt2', 'second_derivative'): 2nd derivative along axis 1 (np.gradient twice)
            - 'region_mask' ('regional_labels', 'region_labels'): Binary event mask (N, seq_len)
            - 'envelope_rms' ('rms'): Moving RMS envelope along axis 1
            - 'rolling_mean' ('smooth'): Moving average along axis 1
            - 'fft_magnitude' ('fft'): Real FFT magnitude profile interpolated to seq_len
        normalize_channels: If True, z-score normalize derived channels (except binary region_mask).
        region_threshold: Sigma threshold for automatic regional event mask.
        existing_region_labels: Optional pre-computed regional labels (N, seq_len) or (N, seq_len, 1).

    Returns:
        float32 numpy array of shape (N, seq_len, len(channels)).
    """
    arr = np.asarray(base_signal, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim == 2:
        sig = arr
        N, seq_len = sig.shape
    elif arr.ndim == 3:
        N, seq_len, C = arr.shape
        sig = arr[:, :, 0]
    else:
        sig = arr.reshape(arr.shape[0], -1)
        N, seq_len = sig.shape

    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)

    if not channels:
        channels = ["raw"]

    def _canonical_ch(ch: str) -> str:
        c = str(ch).strip().lower().replace("-", "_")
        if c in ("raw", "original", "sig", "signal", "x"):
            return "raw"
        if c in ("raw_all", "all_raw"):
            return "raw_all"
        if c in ("derivative_1", "d1", "dx_dt", "first_derivative", "diff1", "deriv1"):
            return "derivative_1"
        if c in ("derivative_2", "d2", "d2x_dt2", "second_derivative", "diff2", "deriv2"):
            return "derivative_2"
        if c in ("region_mask", "regional_labels", "region_labels", "mask", "segmentation", "regions"):
            return "region_mask"
        if c in ("envelope_rms", "rms", "envelope"):
            return "envelope_rms"
        if c in ("rolling_mean", "smooth", "moving_average", "mean"):
            return "rolling_mean"
        if c in ("fft_magnitude", "fft", "spectrum", "rfft"):
            return "fft_magnitude"
        return c

    channel_arrays: List[np.ndarray] = []
    w = max(3, seq_len // 32)

    for raw_key in channels:
        key = _canonical_ch(raw_key)

        if key == "raw":
            ch_data = sig.copy()
        elif key == "raw_all":
            if arr.ndim == 3:
                for c in range(arr.shape[2]):
                    channel_arrays.append(arr[:, :, c].copy())
                continue
            else:
                ch_data = sig.copy()
        elif key == "derivative_1":
            ch_data = np.gradient(sig, axis=1).astype(np.float32)
        elif key == "derivative_2":
            ch_data = np.gradient(np.gradient(sig, axis=1), axis=1).astype(np.float32)
        elif key == "region_mask":
            ch_data = compute_regional_labels(
                base_signal=sig,
                threshold_sigma=region_threshold,
                existing_region_labels=existing_region_labels
            )[:, :, 0]
        elif key == "envelope_rms":
            sig_sq = sig ** 2
            mean_sq = uniform_filter1d(sig_sq, size=w, axis=1, mode="reflect")
            ch_data = np.sqrt(np.maximum(0.0, mean_sq)).astype(np.float32)
        elif key == "rolling_mean":
            ch_data = uniform_filter1d(sig, size=w, axis=1, mode="reflect").astype(np.float32)
        elif key == "fft_magnitude":
            rfft = np.abs(np.fft.rfft(sig, axis=1))
            rfft_len = rfft.shape[1]
            if rfft_len == seq_len:
                ch_data = rfft.astype(np.float32)
            else:
                x_in = np.linspace(0.0, 1.0, rfft_len, endpoint=True)
                x_out = np.linspace(0.0, 1.0, seq_len, endpoint=True)
                ch_data = np.empty((N, seq_len), dtype=np.float32)
                for i in range(N):
                    ch_data[i] = np.interp(x_out, x_in, rfft[i])
        else:
            ch_data = sig.copy()

        ch_data = np.nan_to_num(ch_data, nan=0.0, posinf=0.0, neginf=0.0)

        if normalize_channels and key != "region_mask":
            m = np.mean(ch_data, axis=1, keepdims=True)
            s = np.std(ch_data, axis=1, keepdims=True)
            s = np.where(s < 1e-8, 1.0, s)
            ch_data = (ch_data - m) / s

        channel_arrays.append(ch_data)

    stacked = np.stack(channel_arrays, axis=-1).astype(np.float32)
    return np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)

