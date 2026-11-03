import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app.services.data_manager import DataManager
from app.services.clustering_service import ClusteringService
from app.services.workflow_engine import WorkflowEngine
from app.services.signal_processing import normalize_zscore, normalize_minmax

client = TestClient(app)


def test_column_exclusion_and_raw_preview(tmp_path):
    # Create CSV with ID columns, signal columns, and a label
    csv_file = tmp_path / "signals_with_ids.csv"
    df = pd.DataFrame({
        "id": [101, 102, 103, 104, 105],
        "patient_id": ["P1", "P2", "P3", "P4", "P5"],
        "sig_0": [1.0, 2.0, 3.0, 4.0, 5.0],
        "sig_1": [2.0, 3.0, 4.0, 5.0, 6.0],
        "sig_2": [3.0, 4.0, 5.0, 6.0, 7.0],
        "label": [0, 1, 0, 1, 0]
    })
    df.to_csv(csv_file, index=False)

    # 1. Load with column exclusion
    data, labels, cols = DataManager.load_raw_data(
        csv_file,
        has_header=True,
        label_col="label",
        exclude_cols=["id", "patient_id"]
    )
    assert data.shape == (5, 3)
    assert cols == ["sig_0", "sig_1", "sig_2"]
    assert len(labels) == 5

    # 2. Test preview_raw_file
    preview = DataManager.preview_raw_file(
        csv_file,
        has_header=True,
        offset=1,
        max_rows=2,
        exclude_cols=["id", "patient_id"],
        label_col="label"
    )
    assert preview["total_rows"] == 5
    assert preview["total_cols"] == 3
    assert preview["offset"] == 1
    assert len(preview["preview_rows"]) == 2
    assert "id" in preview["suggested_id_cols"]
    assert "patient_id" in preview["suggested_id_cols"]
    assert len(preview["sample_waveforms"]) == 2


def test_api_raw_preview_and_register(tmp_path):
    # Create CSV
    csv_file = tmp_path / "test_api_preview.csv"
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "s0": [0.1, 0.2, 0.3, 0.4],
        "s1": [0.5, 0.6, 0.7, 0.8],
        "s2": [0.9, 1.0, 1.1, 1.2],
    })
    df.to_csv(csv_file, index=False)

    # Call preview_raw API endpoint
    res = client.post("/api/data/preview_raw", json={
        "file_path": str(csv_file),
        "delimiter": ",",
        "has_header": True,
        "offset": 0,
        "max_rows": 2,
        "exclude_cols": ["id"]
    })
    assert res.status_code == 200
    data = res.json()
    assert data["total_rows"] == 4
    assert data["total_cols"] == 3
    assert "id" in data["excluded_columns"]
    assert data["active_columns"] == ["s0", "s1", "s2"]


def test_project_crud_and_disk_cleanup(tmp_path):
    # 1. Create project
    create_res = client.post("/api/projects", json={
        "name": "Temporary Test Project",
        "description": "Will be edited and deleted"
    })
    assert create_res.status_code == 200
    proj = create_res.json()
    proj_id = proj["id"]

    # 2. Update project
    update_res = client.put(f"/api/projects/{proj_id}", json={
        "name": "Updated Test Project Name",
        "description": "Updated Description"
    })
    assert update_res.status_code == 200
    assert update_res.json()["name"] == "Updated Test Project Name"

    # 3. Delete project
    del_res = client.delete(f"/api/projects/{proj_id}")
    assert del_res.status_code == 200
    assert del_res.json()["id"] == proj_id

    # Verify project not found
    get_res = client.get(f"/api/projects/{proj_id}")
    assert get_res.status_code == 404


def test_clustering_group_waveforms(tmp_path):
    # 1. Register a dataset with 20 signals of length 64
    csv_file = tmp_path / "cluster_signals.csv"
    arr = np.random.randn(20, 64).astype(np.float32)
    pd.DataFrame(arr).to_csv(csv_file, header=False, index=False)

    reg = DataManager.register_dataset(
        project_id="default-project",
        name="Cluster Test Dataset",
        source_path=csv_file,
        has_header=False
    )
    dataset_id = reg["id"]

    # 2. Run clustering with k=2
    cluster_res = ClusteringService.run_clustering(
        project_id="default-project",
        source_type="dataset",
        source_id=dataset_id,
        algorithm="kmeans",
        n_clusters=2
    )
    clustering_id = cluster_res["clustering_id"]

    # 3. Request group waveforms for cluster 0
    group_wf = ClusteringService.get_cluster_group_waveforms(
        clustering_id=clustering_id,
        cluster_label=0,
        max_waveforms=10
    )
    assert group_wf["clustering_id"] == clustering_id
    assert group_wf["cluster_label"] == 0
    assert group_wf["sample_count"] > 0
    assert len(group_wf["mean"]) == 64
    assert len(group_wf["std"]) == 64
    assert len(group_wf["waveforms"]) > 0
    assert np.isclose(group_wf["upper_std"][0], group_wf["mean"][0] + group_wf["std"][0])

    # 4. API endpoint test
    api_res = client.get(f"/api/clustering/group_waveforms/{clustering_id}/0")
    assert api_res.status_code == 200
    assert api_res.json()["sample_count"] == group_wf["sample_count"]


def test_signals_group_waveform_analysis_api(tmp_path):
    # Register dataset
    csv_file = tmp_path / "group_analysis.csv"
    df = pd.DataFrame(np.random.randn(15, 32))
    df["label"] = [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 1, 1, 2, 2]
    df.to_csv(csv_file, index=False)

    reg = DataManager.register_dataset(
        project_id="default-project",
        name="Group Analysis Dataset",
        source_path=csv_file,
        has_header=True,
        label_col="label"
    )
    dataset_id = reg["id"]

    # Call /api/signals/group_waveform_analysis
    res = client.post("/api/signals/group_waveform_analysis", json={
        "dataset_id": dataset_id,
        "label_value": "1"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["sample_count"] == 5
    assert len(data["mean"]) == 32
    assert len(data["waveforms"]) == 5


def test_waveform_internal_normalization():
    # Signals with very different scales
    signals = np.array([
        [10.0, 20.0, 30.0, 40.0, 50.0],
        [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
    ], dtype=np.float32)

    # 1. Per-signal (internal) Z-Score
    norm_internal = normalize_zscore(signals, per_signal=True)
    # Both signals have same relative trend, so per-signal z-scores must be identical!
    assert np.allclose(norm_internal[0], norm_internal[1], atol=1e-5)
    assert np.isclose(np.mean(norm_internal[0]), 0.0, atol=1e-5)
    assert np.isclose(np.std(norm_internal[0]), 1.0, atol=1e-4)

    # 2. Per-signal (internal) Min-Max
    minmax_internal = normalize_minmax(signals, feature_range=(0.0, 1.0), per_signal=True)
    assert np.allclose(minmax_internal[0], minmax_internal[1], atol=1e-5)
    assert np.isclose(np.min(minmax_internal[0]), 0.0)
    assert np.isclose(np.max(minmax_internal[0]), 1.0)
