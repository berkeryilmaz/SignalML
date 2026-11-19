import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.database import ensure_default_project
from app.services.data_manager import DataManager
from app.services.workflow_engine import WorkflowEngine
from app.services.signal_processing import (
    subtract_mean,
    subtract_median,
    subtract_first_n_median,
    subtract_first_n_mean,
    subtract_last_n_median,
    subtract_linear_baseline,
    subtract_polynomial_baseline,
    subtract_rolling_baseline,
    run_pipeline,
)


def test_subtract_first_n_median():
    # Pre-trigger baseline of +7.5 in the first 20 samples, followed by a wide large pulse [20:100]
    sig = np.full((2, 100), 7.5, dtype=np.float32)
    sig[0, 20:90] = 100.0  # 70% of signal is pulse -> global median would be 100.0!
    sig[1, :] = -3.2
    sig[1, 25:85] = 50.0

    # Whole-signal median fails on wide pulses (>50% duty cycle), whereas first_n_median succeeds!
    whole_med = subtract_median(sig, per_signal=True)
    assert not np.allclose(whole_med[0, :20], 0.0)

    corrected = subtract_first_n_median(sig, n=20, per_signal=True)
    assert corrected.shape == (2, 100)
    assert corrected.dtype == np.float32
    assert np.allclose(corrected[0, :20], 0.0, atol=1e-5)
    assert np.allclose(corrected[0, 20:90], 92.5, atol=1e-5)
    assert np.allclose(corrected[1, :20], 0.0, atol=1e-5)
    assert np.allclose(corrected[1, 25:85], 53.2, atol=1e-5)

    # Global scope (per_signal=False)
    sig_global = np.full((2, 50), 4.0, dtype=np.float32)
    sig_global[:, 20:] = 20.0
    corr_global = subtract_first_n_median(sig_global, n=15, per_signal=False)
    assert np.allclose(corr_global[:, :20], 0.0, atol=1e-5)

    # Edge cases: n=1, n > seq_len, NaN/Inf
    dirty = np.array([[np.nan, 5.0, 5.0, np.inf, 10.0]], dtype=np.float32)
    corr_dirty = subtract_first_n_median(dirty, n=3, per_signal=True)
    assert np.all(np.isfinite(corr_dirty))
    assert np.isclose(corr_dirty[0, 1], 0.0)

    corr_large_n = subtract_first_n_median(sig, n=5000, per_signal=True)
    assert corr_large_n.shape == sig.shape


def test_subtract_first_n_mean():
    rng = np.random.default_rng(123)
    pre_noise = rng.normal(loc=12.0, scale=0.05, size=(3, 40)).astype(np.float32)
    pulse = np.full((3, 60), 80.0, dtype=np.float32)
    sig = np.hstack([pre_noise, pulse])

    corrected = subtract_first_n_mean(sig, n=40, per_signal=True)
    assert corrected.shape == (3, 100)
    # Mean of the first 40 samples for each row must be ~0.0
    assert np.allclose(np.mean(corrected[:, :40], axis=1), 0.0, atol=1e-5)
    assert np.allclose(np.mean(corrected[:, 40:], axis=1), 68.0, atol=0.1)


def test_subtract_last_n_median():
    # Pulse at the beginning [0:40], resting tail baseline at [40:100] with offset -6.0
    sig = np.full((2, 100), -6.0, dtype=np.float32)
    sig[:, :40] = 30.0

    corr_med = subtract_last_n_median(sig, n=30, stat="median", per_signal=True)
    assert corr_med.shape == (2, 100)
    assert np.allclose(corr_med[:, -30:], 0.0, atol=1e-5)
    assert np.allclose(corr_med[:, :40], 36.0, atol=1e-5)

    corr_mean = subtract_last_n_median(sig, n=30, stat="mean", per_signal=True)
    assert np.allclose(corr_mean[:, -30:], 0.0, atol=1e-5)


def test_subtract_mean():
    sig = np.array([[2.0, 4.0, 6.0, 8.0], [10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
    corr = subtract_mean(sig, per_signal=True)
    assert np.allclose(np.mean(corr, axis=1), [0.0, 0.0], atol=1e-5)
    # Standard deviation is preserved (unlike z-score)
    assert np.allclose(np.std(corr, axis=1), np.std(sig, axis=1), atol=1e-5)


def test_subtract_linear_baseline():
    # Signal with linear drift from 2.0 to 12.0 + strong pulse in the middle [40:60]
    t = np.linspace(0.0, 1.0, 100, dtype=np.float32)
    ramp = 2.0 + 10.0 * t
    sig = ramp.reshape(1, -1).copy()
    sig[0, 45:55] += 50.0

    # Edge mode ignores the central pulse and connects start/end baselines
    corr_edge = subtract_linear_baseline(sig, mode="edge", n_edge=10, stat="mean")
    assert corr_edge.shape == (1, 100)
    # Outside the pulse [10:40], residual should be near 0 (within edge window average offset)
    assert float(np.max(np.abs(corr_edge[0, 15:40]))) < 0.6
    # Pulse height of 50.0 is preserved
    assert float(np.max(corr_edge[0, 45:55])) > 49.0

    # OLS mode on pure linear ramp zeroes it completely
    pure_ramp = np.vstack([2.0 + 10.0 * t, -5.0 + 3.0 * t]).astype(np.float32)
    corr_ols = subtract_linear_baseline(pure_ramp, mode="ols")
    assert np.allclose(corr_ols, 0.0, atol=1e-4)


def test_subtract_polynomial_baseline():
    seq_len = 200
    x = np.linspace(-1.0, 1.0, seq_len, dtype=np.float32)
    poly_bg = 3.0 + 1.5 * x + 4.0 * (x ** 2)
    sig = poly_bg.reshape(1, -1).copy()
    # Add sharp narrow pulse at the center
    sig[0, 95:105] += 40.0

    corr = subtract_polynomial_baseline(sig, degree=2, exclude_peak_sigma=2.0)
    assert corr.shape == (1, seq_len)
    # Background regions [0:80] and [120:200] should be close to 0.0
    assert float(np.max(np.abs(corr[0, :80]))) < 0.2
    assert float(np.max(np.abs(corr[0, 120:]))) < 0.2
    # Pulse peak should still be ~40.0
    assert np.isclose(float(np.max(corr[0, 95:105])), 40.0, atol=0.3)


def test_subtract_rolling_baseline():
    seq_len = 200
    t = np.linspace(0.0, 2.0 * np.pi, seq_len, dtype=np.float32)
    slow_drift = 5.0 + 2.0 * np.sin(t)
    sig = slow_drift.reshape(1, -1).copy()
    # Add narrow spike at index 100 (width 3)
    sig[0, 99:102] += 25.0

    corr_med = subtract_rolling_baseline(sig, window_size=31, method="median")
    assert corr_med.shape == (1, seq_len)
    # Slow drift is removed away from the spike
    assert float(np.median(np.abs(corr_med[0, 20:80]))) < 0.35
    # Narrow spike is preserved by moving median
    assert float(np.max(corr_med[0, 99:102])) > 24.0

    corr_mean = subtract_rolling_baseline(sig, window_size=31, method="mean")
    assert corr_mean.shape == (1, seq_len)

    corr_open = subtract_rolling_baseline(sig, window_size=31, method="min_opening")
    assert corr_open.shape == (1, seq_len)
    assert float(np.max(corr_open[0, 99:102])) > 24.0


def test_pipeline_and_api_and_workflow_integration(tmp_path):
    sig = np.full((3, 80), 6.0, dtype=np.float32)
    sig[:, 30:50] = 26.0

    steps = [
        {"name": "subtract_first_n_median", "params": {"n": 20, "per_signal": True}},
        {"name": "subtract_first_n_mean", "params": {"n": 20, "per_signal": True}},
        {"name": "subtract_last_n_median", "params": {"n": 20, "stat": "median", "per_signal": True}},
        {"name": "subtract_mean", "params": {"per_signal": True}},
        {"name": "subtract_linear_baseline", "params": {"mode": "edge", "n_edge": 15, "stat": "median"}},
        {"name": "subtract_polynomial_baseline", "params": {"degree": 2, "exclude_peak_sigma": 2.0}},
        {"name": "subtract_rolling_baseline", "params": {"window_size": 21, "method": "median"}},
    ]
    out = run_pipeline(sig, steps)
    assert out.shape == (3, 80)
    assert np.all(np.isfinite(out))

    # Test API preview and WorkflowEngine with subtract_first_n_median
    proj = ensure_default_project()
    proj_id = proj["id"]
    npy_path = tmp_path / "test_baseline_ds.npy"
    np.save(str(npy_path), sig)
    ds = DataManager.register_dataset(project_id=proj_id, name="Baseline Test DS", source_path=npy_path)
    ds_id = ds["id"]

    client = TestClient(app)
    resp = client.post(
        "/api/signals/process_preview",
        json={
            "dataset_id": ds_id,
            "pipeline": [{"name": "subtract_first_n_median", "params": {"n": 20, "per_signal": True}}],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["pipeline_steps_applied"] == 1
    proc_y = body["results"][0]["processed"]["y"]
    assert np.allclose(proc_y[:20], 0.0, atol=1e-5)
    assert np.allclose(proc_y[30:50], 20.0, atol=1e-5)

    # Test WorkflowEngine DAG with subtract_first_n_median node
    graph = {
        "nodes": [
            {"id": "1", "type": "data_source", "properties": {"dataset_id": ds_id}},
            {"id": "2", "type": "csv_parser", "properties": {}},
            {"id": "3", "type": "subtract_first_n_median", "properties": {"n": 20, "per_signal": True}},
        ],
        "links": [
            {"id": "l1", "origin_id": "1", "origin_slot": 0, "target_id": "2", "target_slot": 0},
            {"id": "l2", "origin_id": "2", "origin_slot": 0, "target_id": "3", "target_slot": 0},
        ],
    }
    wf_res = WorkflowEngine.execute_workflow(proj_id, graph)
    assert wf_res["success"] is True
    assert wf_res["node_results"]["3"]["status"] == "success"
