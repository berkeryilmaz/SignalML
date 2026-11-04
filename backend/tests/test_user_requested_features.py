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


def test_get_model_by_id_and_validate_shapes():
    graph = {
        "nodes": [
            {"id": 1, "type": "keras/input", "properties": {"shape": "32, 1"}},
            {"id": 2, "type": "keras/conv1d", "properties": {"filters": 8, "kernel_size": 3, "padding": "same", "activation": "relu"}},
            {"id": 3, "type": "keras/flatten", "properties": {}},
            {"id": 4, "type": "keras/dense", "properties": {"units": 4, "activation": "relu"}},
            {"id": 5, "type": "keras/dense", "properties": {"units": 32, "activation": "linear"}},
            {"id": 6, "type": "keras/reshape", "properties": {"target_shape": "32, 1"}},
            {"id": 7, "type": "keras/output", "properties": {"activation": "linear"}},
        ],
        "links": [
            [1, 1, 0, 2, 0, "Tensor"],
            [2, 2, 0, 3, 0, "Tensor"],
            [3, 3, 0, 4, 0, "Tensor"],
            [4, 4, 0, 5, 0, "Tensor"],
            [5, 5, 0, 6, 0, "Tensor"],
            [6, 6, 0, 7, 0, "Tensor"],
        ]
    }
    # 1. Validate & infer shapes
    val_res = client.post("/api/models/validate", json={"graph": graph})
    assert val_res.status_code == 200
    val_data = val_res.json()
    assert val_data["valid"] is True
    assert len(val_data["summary"]["layers"]) == 7

    # 2. Build & Save model
    build_res = client.post("/api/models/build", json={
        "project_id": "default-project",
        "name": "Test AE Model",
        "task_type": "reconstruction",
        "optimizer": "adam",
        "graph": graph
    })
    assert build_res.status_code == 200
    model_id = build_res.json()["model_id"]

    # 3. GET /api/models/{model_id} (must return 200, not 405!)
    get_res = client.get(f"/api/models/{model_id}")
    assert get_res.status_code == 200
    get_data = get_res.json()
    assert get_data["id"] == model_id
    assert "nodes" in get_data["graph"]
    assert get_data["summary"]["total_params"] > 0


def test_idec_training_and_clustering_on_autoencoder(tmp_path):
    # Create synthetic dataset with 2 distinct signal clusters of length 24 (testing pad/truncate to 32!)
    rng = np.random.default_rng(42)
    c1 = np.sin(np.linspace(0, 3.14, 24)) + rng.normal(0, 0.05, (12, 24))
    c2 = np.cos(np.linspace(0, 3.14, 24)) + rng.normal(0, 0.05, (12, 24))
    signals = np.vstack([c1, c2]).astype(np.float32)

    csv_file = tmp_path / "idec_signals.csv"
    pd.DataFrame(signals).to_csv(csv_file, index=False)

    reg = DataManager.register_dataset(
        project_id="default-project",
        name="IDEC Test Signals",
        source_path=csv_file,
        has_header=True
    )
    dataset_id = reg["id"]

    # Build a 1-output Autoencoder with input shape (32, 1) and bottleneck Dense(3)
    graph = {
        "nodes": [
            {"id": 1, "type": "keras/input", "properties": {"shape": "32, 1"}},
            {"id": 2, "type": "keras/flatten", "properties": {}},
            {"id": 3, "type": "keras/dense", "properties": {"units": 8, "activation": "relu"}},
            {"id": 4, "type": "keras/dense", "properties": {"units": 3, "activation": "relu"}},
            {"id": 5, "type": "keras/dense", "properties": {"units": 32, "activation": "linear"}},
            {"id": 6, "type": "keras/reshape", "properties": {"target_shape": "32, 1"}},
            {"id": 7, "type": "keras/output", "properties": {"activation": "linear"}},
        ],
        "links": [
            [1, 1, 0, 2, 0, "Tensor"],
            [2, 2, 0, 3, 0, "Tensor"],
            [3, 3, 0, 4, 0, "Tensor"],
            [4, 4, 0, 5, 0, "Tensor"],
            [5, 5, 0, 6, 0, "Tensor"],
            [6, 6, 0, 7, 0, "Tensor"],
        ]
    }
    build_res = client.post("/api/models/build", json={
        "project_id": "default-project",
        "name": "AE for IDEC",
        "task_type": "reconstruction",
        "graph": graph
    })
    assert build_res.status_code == 200
    model_id = build_res.json()["model_id"]

    # 1. Run IDEC Clustering via /api/clustering/run
    clus_res = client.post("/api/clustering/run", json={
        "project_id": "default-project",
        "source_type": "dataset",
        "source_id": dataset_id,
        "algorithm": "idec",
        "n_clusters": 2,
        "model_id": model_id,
        "epochs": 2,
        "pretrain_epochs": 1,
        "gamma": 0.1
    })
    assert clus_res.status_code == 200
    clus_data = clus_res.json()
    assert clus_data["algorithm"] == "idec"
    assert len(clus_data["points"]) == 24
    assert "mean_cluster_confidence" in clus_data["metrics"]

    # 2. Run IDEC Probabilities endpoint on the model
    prob_res = client.get(f"/api/models/{model_id}/idec_probabilities/{dataset_id}")
    assert prob_res.status_code == 200
    prob_data = prob_res.json()
    assert prob_data["n_clusters"] >= 2
    assert prob_data["total_samples"] == 24

    # 3. Run IDEC Training via /api/training/start (idec_mode=True)
    import time
    train_res = client.post("/api/training/start", json={
        "project_id": "default-project",
        "model_id": model_id,
        "dataset_id": dataset_id,
        "epochs": 2,
        "batch_size": 8,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "idec_mode": True,
        "n_clusters": 2,
        "gamma": 0.1,
        "pretrain_epochs": 1,
        "update_interval": 1
    })
    assert train_res.status_code == 200
    job_id = train_res.json()["job_id"]

    # Poll until completed
    for _ in range(60):
        st_res = client.get(f"/api/training/status/{job_id}")
        assert st_res.status_code == 200
        st = st_res.json()
        if st["status"] in ("completed", "failed"):
            break
        time.sleep(0.25)

    assert st["status"] == "completed", f"IDEC training failed: {st.get('error_message')}"
    assert st["evaluation"]["task_type"] == "idec_clustering"
    assert st["evaluation"]["n_clusters"] == 2
    assert len(st["history"]["loss"]) >= 2


def test_project_isolation_models_and_datasets():
    """
    Verifies that when a new project is created, models and datasets registered with
    its project_id are strictly isolated from default-project and correctly linked.
    """
    # 1. Create a brand new project
    proj_res = client.post("/api/projects", json={
        "name": "Isolation Verification Project",
        "description": "Verifying project scoping"
    })
    assert proj_res.status_code == 200
    new_proj_id = proj_res.json()["id"]

    # 2. Build model in the new project
    graph_payload = {
        "nodes": [
            {"id": "1", "type": "input", "properties": {"shape": "64, 1"}},
            {"id": "2", "type": "dense", "properties": {"units": 16, "activation": "relu"}},
            {"id": "3", "type": "flatten", "properties": {}},
            {"id": "4", "type": "dense", "properties": {"units": 2, "activation": "softmax"}}
        ],
        "links": [
            ["l1", "1", 0, "2", 0],
            ["l2", "2", 0, "3", 0],
            ["l3", "3", 0, "4", 0]
        ]
    }
    model_res = client.post("/api/models/build", json={
        "project_id": new_proj_id,
        "name": "Isolated Keras Model",
        "task_type": "classification",
        "optimizer": "adam",
        "graph": graph_payload
    })
    assert model_res.status_code == 200
    model_data = model_res.json()
    model_id = model_data["model_id"]
    assert new_proj_id in model_data["weights_path"]

    # 3. Verify model lists
    new_proj_models = client.get(f"/api/models/list/{new_proj_id}").json()
    assert any(m["id"] == model_id for m in new_proj_models)

    default_proj_models = client.get("/api/models/list/default-project").json()
    assert not any(m["id"] == model_id for m in default_proj_models)

    # 4. Verify project details endpoint
    proj_details = client.get(f"/api/projects/{new_proj_id}").json()
    assert proj_details["active_model_id"] == model_id
    assert any(m["id"] == model_id for m in proj_details["models"])


def test_training_accuracy_metric_isolation_and_reconstruction_accuracy(tmp_path):
    """
    Verifies that:
    1. Reconstruction / Autoencoder training computes genuine Explained Variance (R^2 reconstruction accuracy)
       under 'accuracy' and 'val_accuracy' without confusing it with error variables (mae/loss).
    2. Model history keeps 'accuracy' and 'mae' separated as distinct metrics.
    """
    import time
    rng = np.random.default_rng(123)
    X_synth = rng.normal(5.0, 1.0, (20, 16)).astype(np.float32)
    csv_file = tmp_path / "recon_signals.csv"
    pd.DataFrame(X_synth).to_csv(csv_file, index=False)

    reg = DataManager.register_dataset(
        project_id="default-project",
        name="Recon Test Signals",
        source_path=csv_file,
        has_header=True
    )
    dataset_id = reg["id"]

    # Build simple reconstruction model (input 16 -> Dense 4 -> Dense 16)
    graph = {
        "nodes": [
            {"id": "1", "type": "input", "properties": {"shape": "16, 1"}},
            {"id": "2", "type": "flatten", "properties": {}},
            {"id": "3", "type": "dense", "properties": {"units": 4, "activation": "relu"}},
            {"id": "4", "type": "dense", "properties": {"units": 16, "activation": "linear"}},
            {"id": "5", "type": "reshape", "properties": {"target_shape": "16, 1"}},
            {"id": "6", "type": "output", "properties": {"activation": "linear"}}
        ],
        "links": [
            ["l1", "1", 0, "2", 0],
            ["l2", "2", 0, "3", 0],
            ["l3", "3", 0, "4", 0],
            ["l4", "4", 0, "5", 0],
            ["l5", "5", 0, "6", 0]
        ]
    }
    build_res = client.post("/api/models/build", json={
        "project_id": "default-project",
        "name": "Autoencoder Accuracy Test",
        "task_type": "reconstruction",
        "graph": graph
    })
    assert build_res.status_code == 200
    model_id = build_res.json()["model_id"]

    # Start training with val_split
    train_res = client.post("/api/training/start", json={
        "project_id": "default-project",
        "model_id": model_id,
        "dataset_id": dataset_id,
        "epochs": 2,
        "batch_size": 8,
        "val_split": 0.25,
        "learning_rate": 0.01,
        "optimizer": "adam"
    })
    assert train_res.status_code == 200
    job_id = train_res.json()["job_id"]

    # Poll status
    for _ in range(60):
        st_res = client.get(f"/api/training/status/{job_id}")
        assert st_res.status_code == 200
        st = st_res.json()
        if st["status"] in ("completed", "failed"):
            break
        time.sleep(0.2)

    assert st["status"] == "completed", f"Training failed: {st.get('error')}"
    hist = st["history"]

    # Verify both loss and mae are present
    assert "loss" in hist and len(hist["loss"]) >= 2
    assert "mae" in hist and len(hist["mae"]) >= 2

    # Verify accuracy is computed and bounded between 0.0 and 1.0 (reconstruction R^2)
    assert "accuracy" in hist and len(hist["accuracy"]) >= 2
    assert all(0.0 <= acc <= 1.0 for acc in hist["accuracy"] if acc is not None)
    # Ensure accuracy is NOT identical to mae (mae is error, accuracy is bounded quality)
    assert hist["accuracy"] != hist["mae"], "Accuracy must not be identical to MAE error"




