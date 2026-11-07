import pytest
import io
import numpy as np
import pandas as pd

from app.services.signal_processing import (
    first_n,
    last_n,
    pad_signals,
    truncate_signals,
    to_fixed_length,
    align_starts,
    align_ends,
    peak_window,
    peak_asymmetric_window,
    normalize_minmax,
    normalize_zscore,
    subtract_median,
    handle_missing,
    export_signals,
    run_pipeline
)


def test_first_n_and_last_n():
    signals = np.arange(50, dtype=np.float32).reshape(5, 10)
    
    # First 4
    f4 = first_n(signals, 4)
    assert f4.shape == (5, 4)
    assert np.array_equal(f4[0], [0, 1, 2, 3])

    # Last 3
    l3 = last_n(signals, 3)
    assert l3.shape == (5, 3)
    assert np.array_equal(l3[0], [7, 8, 9])


def test_pad_and_truncate():
    signals = np.ones((4, 10), dtype=np.float32)

    # Padding with constant
    padded = pad_signals(signals, target_length=15, mode="constant", constant_values=0.0)
    assert padded.shape == (4, 15)
    assert np.all(padded[:, 10:] == 0.0)

    # Edge padding
    signals[0, -1] = 5.0
    padded_edge = pad_signals(signals, target_length=12, mode="edge")
    assert padded_edge[0, 11] == 5.0

    # Truncation
    trunc_end = truncate_signals(signals, target_length=6, position="end")
    assert trunc_end.shape == (4, 6)

    trunc_start = truncate_signals(signals, target_length=4, position="start")
    assert trunc_start.shape == (4, 4)

    # To Fixed Length
    fixed_up = to_fixed_length(signals, target_length=20)
    assert fixed_up.shape == (4, 20)

    fixed_down = to_fixed_length(signals, target_length=5)
    assert fixed_down.shape == (4, 5)


def test_align_starts_and_ends():
    # Signal with delayed onset at index 4
    signals = np.zeros((2, 20), dtype=np.float32)
    signals[0, 4:10] = [1.0, 2.0, 3.0, 4.0, 2.0, 1.0]
    signals[1, 7:15] = [2.0, 4.0, 6.0, 8.0, 4.0, 2.0, 1.0, 0.5]

    aligned_starts = align_starts(signals, threshold_ratio=0.1)
    # The first significant non-zero point should now be at index 0
    assert aligned_starts[0, 0] >= 0.4
    assert aligned_starts[1, 0] >= 0.8

    aligned_ends = align_ends(signals, threshold_ratio=0.1)
    # The last significant point should be placed near the end
    assert aligned_ends[0, -1] != 0.0 or aligned_ends[0, -2] != 0.0


def test_peak_windowing():
    # Signal with peak at index 15
    signals = np.zeros((3, 40), dtype=np.float32)
    signals[0, 15] = 10.0
    signals[1, 25] = 15.0
    signals[2, 5] = 8.0

    # Asymmetric window: M=4 left, K=6 right -> window length is 4 + 6 + 1 = 11
    asym = peak_asymmetric_window(signals, left_m=4, right_k=6, mode="max_abs")
    assert asym.shape == (3, 11)
    # In every row, peak must be at index left_m = 4
    assert asym[0, 4] == 10.0
    assert asym[1, 4] == 15.0
    assert asym[2, 4] == 8.0

    # Symmetric window: size 21
    sym = peak_window(signals, window_size=21)
    assert sym.shape == (3, 21)


def test_normalization_and_missing_values():
    signals = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [10.0, 20.0, 30.0, 40.0, 50.0]
    ], dtype=np.float32)

    # Min-Max
    mm = normalize_minmax(signals, feature_range=(0.0, 1.0), per_signal=True)
    assert np.isclose(mm[0, 0], 0.0)
    assert np.isclose(mm[0, -1], 1.0)
    assert np.isclose(mm[1, 0], 0.0)
    assert np.isclose(mm[1, -1], 1.0)

    # Z-Score
    zs = normalize_zscore(signals, per_signal=True)
    assert np.isclose(np.mean(zs[0]), 0.0, atol=1e-5)
    assert np.isclose(np.std(zs[0]), 1.0, atol=1e-5)

    # Median Baseline Subtraction (per signal)
    med_sig = subtract_median(signals, per_signal=True)
    assert np.allclose(med_sig[0], [-2.0, -1.0, 0.0, 1.0, 2.0])
    assert np.allclose(med_sig[1], [-20.0, -10.0, 0.0, 10.0, 20.0])
    assert np.allclose(np.median(med_sig, axis=1), [0.0, 0.0])

    # Median Baseline Subtraction via run_pipeline
    piped = run_pipeline(signals, [{"name": "subtract_median", "params": {"per_signal": True}}])
    assert np.allclose(piped, med_sig)

    # Missing values
    dirty = np.array([[1.0, np.nan, 3.0, 4.0, np.inf]], dtype=np.float32)
    clean = handle_missing(dirty, strategy="interpolate")
    assert not np.isnan(clean).any()
    assert np.isclose(clean[0, 1], 2.0)  # interpolated between 1.0 and 3.0


def test_exact_n_signals_export():
    signals = np.random.randn(8, 200).astype(np.float32)
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    # Export to exact 256 samples per row in CSV
    target_N = 256
    content_bytes, mime, verif = export_signals(
        signals=signals,
        labels=labels,
        output_format="csv",
        target_length=target_N,
        separate_labels=False
    )

    assert verif["num_rows"] == 8
    assert verif["num_cols"] == target_N
    assert mime == "text/csv"

    # Read back CSV and check columns: 256 data columns + 1 label column = 257 columns
    df = pd.read_csv(io.BytesIO(content_bytes), header=None)
    assert df.shape == (8, target_N + 1)
    # Check that there is NO extraneous index column
    # The first row first value should be a float close to signals[0, 0]
    assert isinstance(df.iloc[0, 0], (float, np.floating))

    # Export to NPY
    npy_bytes, _, _ = export_signals(signals, output_format="npy", target_length=target_N)
    loaded_npy = np.load(io.BytesIO(npy_bytes))
    assert loaded_npy.shape == (8, target_N)
