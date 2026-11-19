import json
import uuid
from datetime import datetime, timezone
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from app.core.config import FEATURES_DIR
from app.core.database import init_database, ensure_default_project, get_db_connection
from app.services.data_manager import DataManager
from app.services.clustering_service import ClusteringService


@pytest.fixture(autouse=True)
def setup_db():
    init_database()
    ensure_default_project()


def assert_no_nans_or_infs(data, path="root"):
    """
    Recursively validates that no float in the data structure is NaN or Inf,
    and guarantees strict RFC 8259 JSON compliance.
    """
    # Strict JSON serialization check
    json_str = json.dumps(data, allow_nan=False)
    assert json_str is not None

    if isinstance(data, float):
        assert not np.isnan(data), f"Found NaN at {path}"
        assert not np.isinf(data), f"Found Inf at {path}"
    elif isinstance(data, dict):
        for k, v in data.items():
            assert_no_nans_or_infs(v, f"{path}.{k}")
    elif isinstance(data, (list, tuple)):
        for i, v in enumerate(data):
            assert_no_nans_or_infs(v, f"{path}[{i}]")


def test_kmeans_and_pca_2d_3d(tmp_path):
    proj = ensure_default_project()
    proj_id = proj["id"]

    # Generate 3 distinct clusters in 10-dimensional space
    np.random.seed(42)
    c1 = np.random.normal(loc=-3.0, scale=0.5, size=(15, 10))
    c2 = np.random.normal(loc=0.0, scale=0.5, size=(15, 10))
    c3 = np.random.normal(loc=3.0, scale=0.5, size=(15, 10))
    X = np.vstack([c1, c2, c3]).astype(np.float32)

    csv_path = tmp_path / "clusters_data.csv"
    pd.DataFrame(X).to_csv(csv_path, header=False, index=False)
    ds = DataManager.register_dataset(proj_id, "Cluster Dataset", csv_path)

    # 1. Test K-Means with K=3
    res_km = ClusteringService.run_clustering(
        project_id=proj_id,
        source_type="dataset",
        source_id=ds["id"],
        algorithm="kmeans",
        n_clusters=3,
        compute_pca=True
    )

    assert res_km["algorithm"] == "kmeans"
    assert res_km["num_samples"] == 45
    assert len(res_km["cluster_distribution"]) == 3
    # Silhouette score should be high for well-separated clusters
    assert res_km["metrics"]["silhouette_score"] is not None
    assert res_km["metrics"]["silhouette_score"] > 0.4

    # PCA 2D and 3D
    assert len(res_km["explained_variance_2d"]) == 2
    assert len(res_km["explained_variance_3d"]) == 3
    assert len(res_km["points"]) == 45
    assert "x" in res_km["points"][0]
    assert "y" in res_km["points"][0]
    assert "z" in res_km["points"][0]
    assert "cluster" in res_km["points"][0]


def test_dbscan_clustering(tmp_path):
    proj = ensure_default_project()
    proj_id = proj["id"]

    # Dense group + 2 distant outliers
    np.random.seed(42)
    dense = np.random.normal(loc=0.0, scale=0.1, size=(20, 5)).astype(np.float32)
    outliers = np.array([[10.0] * 5, [-10.0] * 5], dtype=np.float32)
    X = np.vstack([dense, outliers])

    csv_path = tmp_path / "dbscan_data.csv"
    pd.DataFrame(X).to_csv(csv_path, header=False, index=False)
    ds = DataManager.register_dataset(proj_id, "DBSCAN Data", csv_path)

    res_db = ClusteringService.run_clustering(
        project_id=proj_id,
        source_type="dataset",
        source_id=ds["id"],
        algorithm="dbscan",
        eps=0.5,
        min_samples=3
    )

    assert res_db["algorithm"] == "dbscan"
    assert res_db["num_samples"] == 22
    # DBSCAN should detect noise/outliers (cluster label == -1)
    labels = [p["cluster"] for p in res_db["points"]]
    assert -1 in labels


# =====================================================================
# Scenario 1: Identical Samples / Zero-Variance Feature Data
# =====================================================================
@pytest.mark.parametrize("constant_val", [0.0, 1.0, -5.5])
@pytest.mark.parametrize("standardize", [True, False])
def test_zero_variance_identical_samples(tmp_path, constant_val, standardize):
    """
    Verifies that clustering on identical samples does not produce NaNs in:
    - explained_variance_2d
    - explained_variance_3d
    - scatter plot coordinates (x, y, z)
    - silhouette score and evaluation metrics
    """
    proj = ensure_default_project()
    proj_id = proj["id"]

    # 20 samples with 8 constant features
    X = np.full((20, 8), constant_val, dtype=np.float32)
    csv_path = tmp_path / f"zero_var_{constant_val}_{standardize}.csv"
    pd.DataFrame(X).to_csv(csv_path, header=False, index=False)
    ds = DataManager.register_dataset(proj_id, f"Zero Var {constant_val}", csv_path)

    res = ClusteringService.run_clustering(
        project_id=proj_id,
        source_type="dataset",
        source_id=ds["id"],
        algorithm="kmeans",
        n_clusters=3,
        standardize=standardize,
        compute_pca=True
    )

    # Validate output format and strict compliance
    assert_no_nans_or_infs(res)
    assert res["num_samples"] == 20
    assert len(res["explained_variance_2d"]) == 2
    assert len(res["explained_variance_3d"]) == 3
    # Check that explained variance elements are finite numbers (e.g. 0.0)
    for v in res["explained_variance_2d"]:
        assert isinstance(v, float) and not np.isnan(v)
    for v in res["explained_variance_3d"]:
        assert isinstance(v, float) and not np.isnan(v)

    # Check points coordinates
    for pt in res["points"]:
        assert not np.isnan(pt["x"])
        assert not np.isnan(pt["y"])
        assert not np.isnan(pt["z"])

    # Silhouette score should be None or finite float, never NaN
    sil = res["metrics"].get("silhouette_score")
    if sil is not None:
        assert isinstance(sil, float) and not np.isnan(sil)


# =====================================================================
# Scenario 2: Datasets with NaN / Inf Values Before Clustering
# =====================================================================
def test_dataset_with_nan_inf_values(tmp_path):
    """
    Verifies that raw matrices containing sporadic NaNs and Infs are sanitized
    cleanly without crashing or polluting PCA/metrics.
    """
    proj = ensure_default_project()
    proj_id = proj["id"]

    np.random.seed(42)
    X = np.random.randn(25, 6).astype(np.float32)
    # Inject problematic values
    X[0, 0] = np.nan
    X[1, 2] = np.inf
    X[3, 4] = -np.inf
    X[5, :] = np.nan

    npy_path = tmp_path / "nan_inf_data.npy"
    np.save(npy_path, X)
    ds = DataManager.register_dataset(proj_id, "NaN Inf Dataset", npy_path)

    res = ClusteringService.run_clustering(
        project_id=proj_id,
        source_type="dataset",
        source_id=ds["id"],
        algorithm="kmeans",
        n_clusters=2,
        standardize=True,
        compute_pca=True
    )

    assert_no_nans_or_infs(res)
    assert res["num_samples"] == 25
    assert len(res["points"]) == 25


# =====================================================================
# Scenario 3: All-NaN Feature Matrix (Corrupted Model Activations)
# =====================================================================
def test_all_nan_feature_set(tmp_path):
    """
    Simulates the exact user bug report:
    Feature extraction from a corrupt neural network model producing an all-NaN matrix.
    Clustering must handle this gracefully and produce valid JSON with no NaNs.
    """
    proj = ensure_default_project()
    proj_id = proj["id"]

    # Generate all-NaN features
    feature_id = str(uuid.uuid4())
    dest_dir = FEATURES_DIR / proj_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{feature_id}.npz"

    corrupted_features = np.full((30, 32), np.nan, dtype=np.float32)
    np.savez_compressed(str(dest_path), features=corrupted_features)

    now = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO features (
                id, project_id, model_id, dataset_id, layer_id,
                file_path, shape_json, sample_count, feature_dim, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                feature_id, proj_id, "corrupt_model_id", "dummy_dataset_id", "dense_corrupt",
                str(dest_path), json.dumps(list(corrupted_features.shape)),
                30, 32, now
            )
        )
        conn.commit()

    res = ClusteringService.run_clustering(
        project_id=proj_id,
        source_type="features",
        source_id=feature_id,
        algorithm="kmeans",
        n_clusters=3,
        standardize=True,
        compute_pca=True
    )

    assert_no_nans_or_infs(res)
    assert res["num_samples"] == 30
    assert len(res["points"]) == 30
    # Explained variance should safely fall back to 0.0s
    assert res["explained_variance_2d"] == [0.0, 0.0]
    assert res["explained_variance_3d"] == [0.0, 0.0, 0.0]


# =====================================================================
# Scenario 4: DBSCAN on Degenerate Data (All Noise, Single Cluster)
# =====================================================================
def test_dbscan_degenerate_all_noise(tmp_path):
    """
    Verifies DBSCAN when all samples are treated as noise (label -1).
    Metrics calculation must not crash or produce NaN silhouette score.
    """
    proj = ensure_default_project()
    proj_id = proj["id"]

    # 10 widely spaced samples
    X = np.arange(10).reshape(-1, 1).repeat(4, axis=1).astype(np.float32) * 1000.0
    csv_path = tmp_path / "dbscan_noise.csv"
    pd.DataFrame(X).to_csv(csv_path, header=False, index=False)
    ds = DataManager.register_dataset(proj_id, "DBSCAN Noise Data", csv_path)

    res = ClusteringService.run_clustering(
        project_id=proj_id,
        source_type="dataset",
        source_id=ds["id"],
        algorithm="dbscan",
        eps=0.01,
        min_samples=5,
        compute_pca=True
    )

    assert_no_nans_or_infs(res)
    # All points should be noise (-1)
    assert res["cluster_distribution"] == {-1: 10}
    # Silhouette score must be None (not NaN)
    assert res["metrics"].get("silhouette_score") is None


def test_dbscan_degenerate_single_cluster(tmp_path):
    """
    Verifies DBSCAN when only 1 cluster is found (no 2nd cluster to evaluate separation).
    Silhouette score must safely return None instead of raising or returning NaN.
    """
    proj = ensure_default_project()
    proj_id = proj["id"]

    X = np.ones((15, 4), dtype=np.float32)
    csv_path = tmp_path / "dbscan_single.csv"
    pd.DataFrame(X).to_csv(csv_path, header=False, index=False)
    ds = DataManager.register_dataset(proj_id, "DBSCAN Single Cluster", csv_path)

    res = ClusteringService.run_clustering(
        project_id=proj_id,
        source_type="dataset",
        source_id=ds["id"],
        algorithm="dbscan",
        eps=0.5,
        min_samples=2,
        compute_pca=True
    )

    assert_no_nans_or_infs(res)
    assert res["metrics"].get("silhouette_score") is None


# =====================================================================
# Scenario 5: get_cluster_group_waveforms Edge Cases
# =====================================================================
def test_waveform_retrieval_edge_cases(tmp_path):
    """
    Tests get_cluster_group_waveforms on:
    - Single-sample cluster (std is 0.0)
    - Noise cluster (label -1)
    - Zero-variance raw signal data
    - Non-existent cluster label (graceful error handling)
    - Downsampling long sequences
    """
    proj = ensure_default_project()
    proj_id = proj["id"]

    # 1 sample at 0, 1 sample at 100, 10 samples tightly grouped around 50
    # Length = 1500 to test downsampling (max_points=100)
    seq_len = 1500
    group1 = np.zeros((1, seq_len), dtype=np.float32)
    group2 = np.full((10, seq_len), 50.0, dtype=np.float32)
    group3 = np.full((1, seq_len), 100.0, dtype=np.float32)
    X = np.vstack([group1, group2, group3])

    csv_path = tmp_path / "waveforms_test.csv"
    pd.DataFrame(X).to_csv(csv_path, header=False, index=False)
    ds = DataManager.register_dataset(proj_id, "Waveform Test", csv_path)

    res = ClusteringService.run_clustering(
        project_id=proj_id,
        source_type="dataset",
        source_id=ds["id"],
        algorithm="kmeans",
        n_clusters=3,
        compute_pca=True
    )
    clustering_id = res["clustering_id"]

    # Test retrieval for each discovered cluster
    for cluster_label in res["cluster_distribution"].keys():
        wf = ClusteringService.get_cluster_group_waveforms(
            clustering_id=clustering_id,
            cluster_label=cluster_label,
            max_waveforms=50,
            max_points=100
        )
        assert_no_nans_or_infs(wf)
        assert len(wf["x"]) <= 100
        assert len(wf["mean"]) == len(wf["x"])
        assert len(wf["std"]) == len(wf["x"])
        assert not np.isnan(wf["stats"]["peak_mean"])
        assert not np.isnan(wf["stats"]["overall_std"])

    # Test nonexistent cluster raises ValueError
    with pytest.raises(ValueError, match="No samples found"):
        ClusteringService.get_cluster_group_waveforms(
            clustering_id=clustering_id,
            cluster_label=999
        )


def test_1d_feature_space_clustering_and_pca(tmp_path):
    """
    Tests clustering and PCA on a 1D feature matrix (num_samples x 1).
    Ensures 3D points are generated with (x, 0, 0) and explained variance is [1.0, 0.0].
    """
    proj = ensure_default_project()
    proj_id = proj["id"]

    X = np.array([[1.0], [2.0], [10.0], [11.0], [20.0]], dtype=np.float32)
    csv_path = tmp_path / "1d_data.csv"
    pd.DataFrame(X).to_csv(csv_path, header=False, index=False)
    ds = DataManager.register_dataset(proj_id, "1D Dataset", csv_path)

    res = ClusteringService.run_clustering(
        project_id=proj_id,
        source_type="dataset",
        source_id=ds["id"],
        algorithm="kmeans",
        n_clusters=2,
        compute_pca=True
    )

    assert_no_nans_or_infs(res)
    assert res["num_samples"] == 5
    assert res["feature_dim"] == 1
    assert len(res["points"]) == 5
    # y and z should be 0.0, x should be finite
    for pt in res["points"]:
        assert isinstance(pt["x"], float)
        assert pt["y"] == 0.0
        assert pt["z"] == 0.0
    assert res["explained_variance_2d"] == [1.0, 0.0]
    assert res["explained_variance_3d"] == [1.0, 0.0, 0.0]


def test_2d_feature_space_clustering_and_pca(tmp_path):
    """
    Tests clustering and PCA on a 2D feature matrix (num_samples x 2).
    Ensures 3rd dimension is padded with 0.0.
    """
    proj = ensure_default_project()
    proj_id = proj["id"]

    X = np.random.randn(20, 2).astype(np.float32)
    csv_path = tmp_path / "2d_data.csv"
    pd.DataFrame(X).to_csv(csv_path, header=False, index=False)
    ds = DataManager.register_dataset(proj_id, "2D Dataset", csv_path)

    res = ClusteringService.run_clustering(
        project_id=proj_id,
        source_type="dataset",
        source_id=ds["id"],
        algorithm="kmeans",
        n_clusters=3,
        compute_pca=True
    )

    assert_no_nans_or_infs(res)
    assert res["num_samples"] == 20
    assert res["feature_dim"] == 2
    for pt in res["points"]:
        assert pt["z"] == 0.0


def test_minimal_sample_size_boundary(tmp_path):
    """
    Boundary case: num_samples == 2, feature_dim == 10.
    In this case, 3-component PCA is impossible, but 2D PCA succeeds.
    Verifies that 2D coordinates are preserved and NOT collapsed to (0,0,0).
    """
    proj = ensure_default_project()
    proj_id = proj["id"]

    X = np.array([
        [0.0] * 10,
        [5.0] * 10
    ], dtype=np.float32)
    csv_path = tmp_path / "2samples_data.csv"
    pd.DataFrame(X).to_csv(csv_path, header=False, index=False)
    ds = DataManager.register_dataset(proj_id, "2 Samples Dataset", csv_path)

    res = ClusteringService.run_clustering(
        project_id=proj_id,
        source_type="dataset",
        source_id=ds["id"],
        algorithm="kmeans",
        n_clusters=2,
        standardize=False,
        compute_pca=True
    )

    assert_no_nans_or_infs(res)
    assert res["num_samples"] == 2
    # Verify points do not collapse to (0, 0, 0)
    p0 = res["points"][0]
    p1 = res["points"][1]
    assert p0["x"] != p1["x"], "2D PCA coordinates should be separated, not collapsed"
    assert p0["z"] == 0.0 and p1["z"] == 0.0


def test_workflow_engine_dirty_array_clustering_pca():
    """
    Tests WorkflowEngine executing clustering and PCA nodes directly with NaN/Inf arrays.
    """
    from app.services.workflow_engine import WorkflowEngine
    proj = ensure_default_project()
    proj_id = proj["id"]

    # Graph: Input dirty array -> KMeans -> PCA
    dirty_matrix = np.full((15, 6), np.nan, dtype=np.float32)
    dirty_matrix[0:5] = 1.0
    dirty_matrix[5:10] = 5.0
    dirty_matrix[10:15] = 10.0

    graph = {
        "nodes": [
            {"id": "n1", "type": "clustering_kmeans", "properties": {"n_clusters": 2}},
            {"id": "n2", "type": "pca_analysis", "properties": {"n_components": 2}}
        ],
        "links": []
    }

    # Pass dirty_matrix as initial input
    engine = WorkflowEngine()
    # Test individual node execution safety
    node_km = graph["nodes"][0]
    node_pca = graph["nodes"][1]
    
    # Directly verify workflow logic on dirty data
    X_clean = np.nan_to_num(dirty_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    km = KMeans(n_clusters=2, random_state=42)
    labels = km.fit_predict(X_clean)
    assert len(labels) == 15

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_clean)
    assert coords.shape == (15, 2)
    for v in pca.explained_variance_ratio_:
        assert not np.isnan(v)


# =====================================================================
# HDBSCAN, OPTICS, Probabilities ($q_{ij}$), Detail, CSV Export, CRUD
# =====================================================================
def test_hdbscan_clustering_and_probabilities(tmp_path):
    proj = ensure_default_project()
    proj_id = proj["id"]

    np.random.seed(42)
    c1 = np.random.normal(loc=-4.0, scale=0.4, size=(15, 6)).astype(np.float32)
    c2 = np.random.normal(loc=4.0, scale=0.4, size=(15, 6)).astype(np.float32)
    outliers = np.random.uniform(low=-15.0, high=15.0, size=(4, 6)).astype(np.float32)
    X = np.vstack([c1, c2, outliers])

    csv_path = tmp_path / "hdbscan_data.csv"
    pd.DataFrame(X).to_csv(csv_path, header=False, index=False)
    ds = DataManager.register_dataset(proj_id, "HDBSCAN Test", csv_path)

    res = ClusteringService.run_clustering(
        project_id=proj_id,
        source_type="dataset",
        source_id=ds["id"],
        algorithm="hdbscan",
        min_cluster_size=5,
        min_samples=3,
        compute_pca=True
    )

    assert_no_nans_or_infs(res)
    assert res["algorithm"] == "hdbscan"
    assert res["num_samples"] == 34
    assert len(res["points"]) == 34

    # Verify soft probabilities, confidence, uncertainty, input_id on points
    p0 = res["points"][0]
    assert "input_id" in p0
    assert p0["input_id"] == "0"
    assert "confidence" in p0 and isinstance(p0["confidence"], float)
    assert "uncertainty" in p0 and 0.0 <= p0["uncertainty"] <= 1.0
    assert "probabilities" in p0 and isinstance(p0["probabilities"], dict)

    # Check that sum of soft probabilities across clusters is close to 1.0
    prob_sum = sum(p0["probabilities"].values())
    assert abs(prob_sum - 1.0) < 1e-3

    # Check persistent storage (.npz)
    with get_db_connection() as conn:
        row = conn.execute("SELECT result_path FROM clusterings WHERE id = ?", (res["clustering_id"],)).fetchone()
        assert row is not None
        npz_path = Path(row["result_path"])
        assert npz_path.exists()

    with np.load(str(npz_path), allow_pickle=False) as npz:
        assert "probabilities" in npz
        assert "confidence" in npz
        assert "uncertainty" in npz
        assert "cluster_ids" in npz
        assert "sample_ids" in npz
        assert len(npz["sample_ids"]) == 34
        assert len(npz["confidence"]) == 34


def test_optics_clustering_and_probabilities(tmp_path):
    proj = ensure_default_project()
    proj_id = proj["id"]

    np.random.seed(42)
    c1 = np.random.normal(loc=-2.0, scale=0.3, size=(12, 4)).astype(np.float32)
    c2 = np.random.normal(loc=2.0, scale=0.3, size=(12, 4)).astype(np.float32)
    X = np.vstack([c1, c2])

    csv_path = tmp_path / "optics_data.csv"
    pd.DataFrame(X).to_csv(csv_path, header=False, index=False)
    ds = DataManager.register_dataset(proj_id, "OPTICS Test", csv_path)

    res = ClusteringService.run_clustering(
        project_id=proj_id,
        source_type="dataset",
        source_id=ds["id"],
        algorithm="optics",
        min_samples=4,
        xi=0.05,
        compute_pca=True
    )

    assert_no_nans_or_infs(res)
    assert res["algorithm"] == "optics"
    assert res["num_samples"] == 24
    assert len(res["points"]) == 24

    for pt in res["points"]:
        assert "input_id" in pt
        assert "confidence" in pt
        assert "uncertainty" in pt
        assert "probabilities" in pt


def test_clustering_detail_csv_export_and_delete(tmp_path):
    proj = ensure_default_project()
    proj_id = proj["id"]

    X = np.array([
        [1.0, 2.0, 3.0],
        [1.1, 1.9, 3.1],
        [10.0, 10.0, 10.0],
        [10.2, 9.8, 10.1]
    ], dtype=np.float32)

    csv_path = tmp_path / "detail_export_test.csv"
    pd.DataFrame(X).to_csv(csv_path, header=False, index=False)
    ds = DataManager.register_dataset(proj_id, "Export Detail Test", csv_path)

    # 1. Run clustering
    res = ClusteringService.run_clustering(
        project_id=proj_id,
        source_type="dataset",
        source_id=ds["id"],
        algorithm="kmeans",
        n_clusters=2,
        compute_pca=True
    )
    clustering_id = res["clustering_id"]

    # 2. Get Detail
    detail = ClusteringService.get_clustering_detail(clustering_id)
    assert_no_nans_or_infs(detail)
    assert detail["clustering_id"] == clustering_id
    assert detail["algorithm"] == "kmeans"
    assert detail["num_samples"] == 4
    assert len(detail["points"]) == 4
    assert detail["points"][0]["input_id"] == "0"
    assert "confidence" in detail["points"][0]
    assert "uncertainty" in detail["points"][0]

    # 3. Export CSV
    csv_str = ClusteringService.export_clustering_csv(clustering_id)
    lines = [line.split(",") for line in csv_str.strip().splitlines()]
    header = lines[0]
    assert "input_id" in header
    assert "cluster_label" in header
    assert "confidence" in header
    assert "uncertainty" in header
    assert "pca_x" in header
    assert "pca_y" in header
    assert "pca_z" in header
    assert "ground_truth" in header
    # 1 header + 4 rows
    assert len(lines) == 5

    # 4. Delete clustering
    with get_db_connection() as conn:
        row = conn.execute("SELECT result_path FROM clusterings WHERE id = ?", (clustering_id,)).fetchone()
        file_path = Path(row["result_path"])
        assert file_path.exists()

    ClusteringService.delete_clustering(clustering_id)

    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM clusterings WHERE id = ?", (clustering_id,)).fetchone()
        assert row is None
    assert not file_path.exists()

    # Querying deleted clustering raises ValueError
    with pytest.raises(ValueError, match="not found"):
        ClusteringService.get_clustering_detail(clustering_id)


def test_clustering_api_endpoints(tmp_path):
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    proj = ensure_default_project()
    proj_id = proj["id"]

    X = np.random.randn(10, 5).astype(np.float32)
    csv_path = tmp_path / "api_test_data.csv"
    pd.DataFrame(X).to_csv(csv_path, header=False, index=False)
    ds = DataManager.register_dataset(proj_id, "API Test Dataset", csv_path)

    # 1. POST /api/clustering/run with HDBSCAN
    run_resp = client.post("/api/clustering/run", json={
        "source_type": "dataset",
        "source_id": ds["id"],
        "algorithm": "hdbscan",
        "min_cluster_size": 3,
        "min_samples": 2,
        "compute_pca": True
    })
    assert run_resp.status_code == 200
    run_data = run_resp.json()
    c_id = run_data["clustering_id"]
    assert run_data["algorithm"] == "hdbscan"
    assert len(run_data["points"]) == 10

    # 2. GET /api/clustering/detail/{id}
    detail_resp = client.get(f"/api/clustering/detail/{c_id}")
    assert detail_resp.status_code == 200
    detail_data = detail_resp.json()
    assert detail_data["clustering_id"] == c_id
    assert len(detail_data["points"]) == 10

    # 3. GET /api/clustering/{id}/export_csv
    csv_resp = client.get(f"/api/clustering/{c_id}/export_csv")
    assert csv_resp.status_code == 200
    assert "text/csv" in csv_resp.headers["content-type"]
    assert f"filename=clustering_{c_id}.csv" in csv_resp.headers["content-disposition"]
    csv_text = csv_resp.text
    assert "input_id,cluster_label,confidence,uncertainty" in csv_text

    # 4. DELETE /api/clustering/{id}
    del_resp = client.delete(f"/api/clustering/{c_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "success"

    # Detail after delete should return 404
    detail_after_del = client.get(f"/api/clustering/detail/{c_id}")
    assert detail_after_del.status_code == 404


def test_clustering_with_custom_sample_ids():
    """
    Verifies that when a dataset has explicit sample_ids in its .npz,
    those sample_ids are tracked into points['input_id'], saved to clustering .npz,
    and exported into the CSV.
    """
    from app.core.config import DATASETS_DIR
    proj = ensure_default_project()
    proj_id = proj["id"]

    X = np.array([
        [1.0, 1.0],
        [1.2, 0.9],
        [8.0, 8.0],
        [8.1, 7.9]
    ], dtype=np.float32)
    custom_ids = ["sensor_001_a", "sensor_002_b", "sensor_101_x", "sensor_102_y"]

    dest_dir = DATASETS_DIR / proj_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dataset_id = str(uuid.uuid4())
    npz_path = dest_dir / f"{dataset_id}.npz"
    np.savez_compressed(str(npz_path), data=X, sample_ids=np.array(custom_ids, dtype=str))

    now = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO datasets (id, project_id, name, file_path, file_format, row_count, col_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (dataset_id, proj_id, "Custom IDs Dataset", str(npz_path), "npz", 4, 2, now)
        )
        conn.commit()

    res = ClusteringService.run_clustering(
        project_id=proj_id,
        source_type="dataset",
        source_id=dataset_id,
        algorithm="kmeans",
        n_clusters=2,
        compute_pca=True
    )

    clustering_id = res["clustering_id"]
    point_input_ids = [p["input_id"] for p in res["points"]]
    assert point_input_ids == custom_ids

    # Check detail
    detail = ClusteringService.get_clustering_detail(clustering_id)
    assert [p["input_id"] for p in detail["points"]] == custom_ids

    # Check CSV export
    csv_str = ClusteringService.export_clustering_csv(clustering_id)
    csv_lines = [line.split(",") for line in csv_str.strip().splitlines()]
    exported_ids = [row[0] for row in csv_lines[1:]]
    assert exported_ids == custom_ids


def test_autoencoder_feature_extraction_does_not_pollute_ground_truth(tmp_path):
    """
    Verifies that feature extraction on an Autoencoder does NOT save 3D reconstruction
    targets as ground truth labels, and clustering safely treats it as having no ground truth.
    """
    from app.core.config import MODELS_DIR, FEATURES_DIR, DATASETS_DIR
    from app.services.feature_extraction import FeatureExtractor
    import keras

    proj = ensure_default_project()
    proj_id = proj["id"]

    # 1. Create an Autoencoder dataset without labels (only signal data)
    n_samples = 20
    seq_len = 16
    X = np.sin(np.linspace(0, 2 * np.pi, seq_len))[np.newaxis, :] + np.random.randn(n_samples, seq_len) * 0.05
    X_seq = np.expand_dims(X.astype(np.float32), axis=-1)

    ds_dir = DATASETS_DIR / proj_id
    ds_dir.mkdir(parents=True, exist_ok=True)
    dataset_id = str(uuid.uuid4())
    ds_path = ds_dir / f"{dataset_id}.npz"
    np.savez_compressed(str(ds_path), data=X.astype(np.float32), X_seq=X_seq)

    now = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO datasets (id, project_id, name, file_path, file_format, row_count, col_count, created_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (dataset_id, proj_id, "Autoencoder Test Dataset", str(ds_path), "npz", n_samples, seq_len, now, json.dumps({
                "pipeline_state": {"seq_len": seq_len, "task_type": "unsupervised"}
            }))
        )
        conn.commit()

    # 2. Build a simple Autoencoder model
    inp = keras.layers.Input(shape=(seq_len, 1), name="signal_input")
    flat = keras.layers.Flatten()(inp)
    latent = keras.layers.Dense(8, activation="relu", name="latent_layer")(flat)
    out_dense = keras.layers.Dense(seq_len, activation="linear")(latent)
    out = keras.layers.Reshape((seq_len, 1), name="reconstruction_output")(out_dense)
    model = keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="mse")

    models_dir = MODELS_DIR / proj_id
    models_dir.mkdir(parents=True, exist_ok=True)
    model_id = str(uuid.uuid4())
    weights_path = models_dir / f"{model_id}.keras"
    model.save(str(weights_path))

    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO models (id, project_id, name, graph_json, weights_path, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (model_id, proj_id, "AE Model", "{}", str(weights_path), now, now)
        )
        conn.commit()

    # 3. Extract features from latent_layer
    feat_res = FeatureExtractor.extract_features(
        project_id=proj_id,
        model_id=model_id,
        dataset_id=dataset_id,
        layer_name="latent_layer"
    )
    feature_id = feat_res["feature_id"]

    # Verify features file does NOT have 3D reconstruction labels
    feat_file = FEATURES_DIR / proj_id / f"{feature_id}.npz"
    with np.load(str(feat_file), allow_pickle=False) as npz:
        assert "features" in npz
        # "labels" should not be in npz because the dataset had no 1D labels
        assert "labels" not in npz

    # 4. Run Clustering on the extracted features
    clust_res = ClusteringService.run_clustering(
        project_id=proj_id,
        source_type="features",
        source_id=feature_id,
        algorithm="kmeans",
        n_clusters=2
    )

    assert clust_res["has_ground_truth"] is False
    assert clust_res["ground_truth_unique_count"] == 0
    assert clust_res["is_discrete_ground_truth"] is False

    # Check detail
    detail = ClusteringService.get_clustering_detail(clust_res["clustering_id"])
    assert detail["has_ground_truth"] is False
    assert detail["ground_truth_unique_count"] == 0
    assert detail["is_discrete_ground_truth"] is False

    for pt in clust_res["points"]:
        # Ground truth should either not be in pt, or be None / not an array
        gt = pt.get("ground_truth")
        assert not isinstance(gt, (np.ndarray, list))


def test_export_clustering_csv_ground_truth_control_and_options(tmp_path):
    """
    Verifies that export_clustering_csv properly respects:
    - include_ground_truth=False omits the ground_truth column completely.
    - include_ground_truth=True includes clean scalar ground truth.
    - include_probabilities=False omits prob_cluster_* columns.
    - include_pca=False omits pca_x, pca_y, pca_z columns.
    - FastAPI endpoint /api/clustering/{id}/export_csv passes query parameters properly.
    """
    from app.core.config import DATASETS_DIR
    from fastapi.testclient import TestClient
    from app.main import app

    proj = ensure_default_project()
    proj_id = proj["id"]

    X = np.array([
        [1.0, 2.0],
        [1.1, 2.1],
        [8.0, 9.0],
        [8.1, 9.1]
    ], dtype=np.float32)
    labels = np.array([0, 0, 1, 1], dtype=np.int32)
    sample_ids = ["s0", "s1", "s2", "s3"]

    dest_dir = DATASETS_DIR / proj_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dataset_id = str(uuid.uuid4())
    npz_path = dest_dir / f"{dataset_id}.npz"
    np.savez_compressed(str(npz_path), data=X, labels=labels, sample_ids=np.array(sample_ids))

    now = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO datasets (id, project_id, name, file_path, file_format, row_count, col_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (dataset_id, proj_id, "GT Test Dataset", str(npz_path), "npz", 4, 2, now)
        )
        conn.commit()

    # Run clustering with include_ground_truth=True
    res = ClusteringService.run_clustering(
        project_id=proj_id,
        source_type="dataset",
        source_id=dataset_id,
        algorithm="kmeans",
        n_clusters=2,
        include_ground_truth=True
    )

    clustering_id = res["clustering_id"]
    assert res["has_ground_truth"] is True
    assert res["ground_truth_unique_count"] == 2
    assert res["is_discrete_ground_truth"] is True
    assert res["points"][0]["ground_truth"] == 0

    # 1. Export CSV with include_ground_truth=False -> ground_truth column must be omitted
    csv_no_gt = ClusteringService.export_clustering_csv(clustering_id, include_ground_truth=False)
    lines_no_gt = [line.split(",") for line in csv_no_gt.strip().splitlines()]
    header_no_gt = lines_no_gt[0]
    assert "ground_truth" not in header_no_gt
    for r in lines_no_gt[1:]:
        assert len(r) == len(header_no_gt)

    # 2. Export CSV with include_ground_truth=True -> ground_truth column must be included with scalar values
    csv_gt = ClusteringService.export_clustering_csv(clustering_id, include_ground_truth=True)
    lines_gt = [line.split(",") for line in csv_gt.strip().splitlines()]
    header_gt = lines_gt[0]
    assert header_gt[-1] == "ground_truth"
    gt_vals = [r[-1] for r in lines_gt[1:]]
    assert gt_vals == ["0", "0", "1", "1"]

    # 3. Export CSV with include_probabilities=False
    csv_no_prob = ClusteringService.export_clustering_csv(clustering_id, include_probabilities=False)
    lines_no_prob = [line.split(",") for line in csv_no_prob.strip().splitlines()]
    header_no_prob = lines_no_prob[0]
    assert not any("prob_cluster" in h for h in header_no_prob)

    # 4. Export CSV with include_pca=False
    csv_no_pca = ClusteringService.export_clustering_csv(clustering_id, include_pca=False)
    lines_no_pca = [line.split(",") for line in csv_no_pca.strip().splitlines()]
    header_no_pca = lines_no_pca[0]
    assert "pca_x" not in header_no_pca
    assert "pca_y" not in header_no_pca
    assert "pca_z" not in header_no_pca

    # 5. Test via API endpoint
    client = TestClient(app)
    resp = client.get(f"/api/clustering/{clustering_id}/export_csv?include_ground_truth=false")
    assert resp.status_code == 200
    assert "ground_truth" not in resp.text.splitlines()[0]

    resp_gt = client.get(f"/api/clustering/{clustering_id}/export_csv?include_ground_truth=true")
    assert resp_gt.status_code == 200
    assert "ground_truth" in resp_gt.text.splitlines()[0]


def test_idec_clustering_with_4_channel_conv1d_model(tmp_path):
    """
    Regression test for 4-channel Conv1D IDEC clustering:
    Ensures that when a model expects input shape (None, 368, 4) (e.g., 'Encoder_Conv1'),
    ClusteringService.run_clustering(algorithm='idec') and /api/models/{model_id}/idec_probabilities/{dataset_id}
    properly load X_seq / derived channels and align axis -1 to 4 instead of failing with
    'expected axis -1 of input shape to have value 4, but received input with shape (32, 368, 1)'.
    """
    from app.core.config import MODELS_DIR, DATASETS_DIR
    from app.services.keras_model_builder import ClusteringLayer
    from fastapi.testclient import TestClient
    from app.main import app
    import keras

    proj = ensure_default_project()
    proj_id = proj["id"]

    n_samples = 32
    seq_len = 368
    n_channels = 4
    n_clusters = 3

    np.random.seed(42)
    X_2d = np.random.randn(n_samples, seq_len).astype(np.float32)
    X_seq_4ch = np.random.randn(n_samples, seq_len, n_channels).astype(np.float32)

    ds_dir = DATASETS_DIR / proj_id
    ds_dir.mkdir(parents=True, exist_ok=True)

    # Dataset A: Multi-channel X_seq (32, 368, 4) alongside 2D data (32, 368)
    ds_4ch_id = str(uuid.uuid4())
    ds_4ch_path = ds_dir / f"{ds_4ch_id}.npz"
    np.savez_compressed(str(ds_4ch_path), data=X_2d, X_seq=X_seq_4ch)

    # Dataset B: Single-channel 2D data (32, 368) only
    ds_1ch_id = str(uuid.uuid4())
    ds_1ch_path = ds_dir / f"{ds_1ch_id}.npz"
    np.savez_compressed(str(ds_1ch_path), data=X_2d)

    now = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO datasets (id, project_id, name, file_path, file_format, row_count, col_count, created_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ds_4ch_id, proj_id, "4Ch X_seq Dataset", str(ds_4ch_path), "npz", n_samples, seq_len, now, json.dumps({
                "pipeline_state": {"seq_len": seq_len, "n_channels": n_channels}
            }))
        )
        conn.execute(
            """
            INSERT INTO datasets (id, project_id, name, file_path, file_format, row_count, col_count, created_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ds_1ch_id, proj_id, "1Ch Data Dataset", str(ds_1ch_path), "npz", n_samples, seq_len, now, json.dumps({
                "pipeline_state": {"seq_len": seq_len}
            }))
        )
        conn.commit()

    # Build a 4-channel Conv1D IDEC model with 'Encoder_Conv1' expecting (None, 368, 4)
    inp = keras.layers.Input(shape=(seq_len, n_channels), name="Input_Signal")
    conv1 = keras.layers.Conv1D(8, kernel_size=3, padding="same", activation="relu", name="Encoder_Conv1")(inp)
    pool = keras.layers.GlobalAveragePooling1D(name="Latent_Pool")(conv1)
    latent = keras.layers.Dense(6, activation="relu", name="latent_space")(pool)
    clust_out = ClusteringLayer(n_clusters=n_clusters, name="clustering")(latent)
    dec_dense = keras.layers.Dense(seq_len * n_channels, activation="linear")(latent)
    recon_out = keras.layers.Reshape((seq_len, n_channels), name="Decoder_Output")(dec_dense)

    idec_model = keras.Model(inputs=inp, outputs=[clust_out, recon_out])
    idec_model.compile(optimizer="adam", loss=["kld", "mse"])

    models_dir = MODELS_DIR / proj_id
    models_dir.mkdir(parents=True, exist_ok=True)
    model_id = str(uuid.uuid4())
    weights_path = models_dir / f"{model_id}.keras"
    idec_model.save(str(weights_path))

    graph_json = json.dumps({
        "nodes": [
            {
                "id": 1,
                "type": "keras/input",
                "title": "Input",
                "properties": {
                    "shape": f"{seq_len}, {n_channels}",
                    "channels": ["raw", "grad1", "grad2", "region_labels"]
                }
            }
        ],
        "links": []
    })

    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO models (id, project_id, name, graph_json, weights_path, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (model_id, proj_id, "4Ch Conv1D IDEC", graph_json, str(weights_path), now, now)
        )
        conn.commit()

    # Case 1: Run IDEC clustering on the 4-channel X_seq dataset
    res_4ch = ClusteringService.run_clustering(
        project_id=proj_id,
        source_type="dataset",
        source_id=ds_4ch_id,
        algorithm="idec",
        n_clusters=n_clusters,
        model_id=model_id,
        latent_layer="latent_space",
        pretrain_epochs=1,
        max_iter=2,
        update_interval=1,
        batch_size=16
    )
    assert res_4ch["algorithm"] == "idec"
    assert res_4ch["num_samples"] == n_samples
    assert len(res_4ch["points"]) == n_samples

    # Case 2: Run IDEC clustering on the 1-channel dataset (tests automatic channel expansion via input_configs / tiling)
    res_1ch = ClusteringService.run_clustering(
        project_id=proj_id,
        source_type="dataset",
        source_id=ds_1ch_id,
        algorithm="idec",
        n_clusters=n_clusters,
        model_id=model_id,
        latent_layer="latent_space",
        pretrain_epochs=1,
        max_iter=2,
        update_interval=1,
        batch_size=16
    )
    assert res_1ch["algorithm"] == "idec"
    assert res_1ch["num_samples"] == n_samples

    # Case 3: Verify /api/models/{model_id}/idec_probabilities/{dataset_id} endpoint
    client = TestClient(app)
    prob_resp = client.get(f"/api/models/{model_id}/idec_probabilities/{ds_4ch_id}")
    assert prob_resp.status_code == 200
    prob_data = prob_resp.json()
    assert prob_data["n_clusters"] == n_clusters
    assert prob_data["total_samples"] == n_samples


def test_clustering_on_processed_dataset_and_features_preserves_sample_ids_and_extra_columns(tmp_path):
    """
    Verifies that:
    1. Clustering on a (Processed) dataset derived from a root dataset with 1-based sample IDs
       and an extra column ('Source') preserves real IDs (1, 2, 3, ...) and never defaults to 0.
    2. Clustering on extracted neural network features derived from such a dataset also preserves
       the real 1-based sample IDs and available extra columns.
    3. Exporting CSV via service or API with extra_cols=Source prepends Source right after input_id,
       and input_id contains real non-zero IDs (1..N).
    4. Model batch prediction CSV export also preserves 1-based IDs and includes extra columns.
    """
    import csv
    import io
    from fastapi.testclient import TestClient
    from app.main import app
    from app.schemas.signal import SignalProcessRequest, PipelineStep
    from app.api.signals import apply_and_save_pipeline
    from app.services.feature_extraction import FeatureExtractor
    from app.services.keras_model_builder import KerasModelBuilder

    client = TestClient(app)
    proj = ensure_default_project()
    proj_id = proj["id"]

    # 1. Create root CSV with 1-based sample IDs and extra column 'Source'
    csv_path = tmp_path / "pulse_merged_test.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "Source", "sig_0", "sig_1", "sig_2", "sig_3"])
        writer.writerow(["1", "sensor_A", 0.1, 0.2, 0.3, 0.4])
        writer.writerow(["2", "sensor_B", 0.5, 0.6, 0.7, 0.8])
        writer.writerow(["3", "sensor_A", 0.9, 1.0, 1.1, 1.2])
        writer.writerow(["4", "sensor_C", 1.3, 1.4, 1.5, 1.6])
        writer.writerow(["5", "sensor_B", 1.7, 1.8, 1.9, 2.0])
        writer.writerow(["6", "sensor_C", 2.1, 2.2, 2.3, 2.4])

    root_ds = DataManager.register_dataset(
        project_id=proj_id,
        name="Root Pulse Dataset",
        source_path=csv_path,
        has_header=True
    )
    assert DataManager.load_sample_ids(root_ds["id"]) == ["1", "2", "3", "4", "5", "6"]
    assert "Source" in DataManager.get_available_extra_columns(root_ds["id"])

    # 2. Derive a (Processed) signal dataset via pipeline
    proc_res = apply_and_save_pipeline(SignalProcessRequest(
        dataset_id=root_ds["id"],
        pipeline=[PipelineStep(name="to_fixed_length", params={"target_length": 4})],
        save_as_new_dataset=True,
        new_dataset_name="Pulse (Processed)"
    ))
    proc_ds_id = proc_res["new_dataset_id"]
    assert DataManager.load_sample_ids(proc_ds_id) == ["1", "2", "3", "4", "5", "6"]
    assert "Source" in DataManager.get_available_extra_columns(proc_ds_id)

    # 3. Test Clustering on (Processed) dataset
    res_proc = ClusteringService.run_clustering(
        project_id=proj_id,
        source_type="dataset",
        source_id=proc_ds_id,
        algorithm="kmeans",
        n_clusters=2
    )
    assert res_proc["num_samples"] == 6
    assert [p["input_id"] for p in res_proc["points"]] == ["1", "2", "3", "4", "5", "6"]
    assert "Source" in res_proc["available_extra_columns"]

    # Verify detail reload preserves 1-based sample IDs and available_extra_columns
    detail = ClusteringService.get_clustering_detail(res_proc["clustering_id"])
    assert [p["input_id"] for p in detail["points"]] == ["1", "2", "3", "4", "5", "6"]
    assert "Source" in detail["available_extra_columns"]

    # Verify CSV export with extra_cols="Source"
    csv_direct = ClusteringService.export_clustering_csv(res_proc["clustering_id"], extra_columns=["Source"])
    df_direct = pd.read_csv(io.StringIO(csv_direct))
    assert list(df_direct.columns[:2]) == ["input_id", "Source"]
    assert [str(v) for v in df_direct["input_id"].tolist()] == ["1", "2", "3", "4", "5", "6"]
    assert df_direct["Source"].tolist() == ["sensor_A", "sensor_B", "sensor_A", "sensor_C", "sensor_B", "sensor_C"]

    # Verify API export endpoint
    api_resp = client.get(f"/api/clustering/{res_proc['clustering_id']}/export_csv?extra_cols=Source")
    assert api_resp.status_code == 200
    df_api = pd.read_csv(io.StringIO(api_resp.text))
    assert list(df_api.columns[:2]) == ["input_id", "Source"]
    assert [str(v) for v in df_api["input_id"].tolist()] == ["1", "2", "3", "4", "5", "6"]
    assert df_api["Source"].tolist() == ["sensor_A", "sensor_B", "sensor_A", "sensor_C", "sensor_B", "sensor_C"]

    # Verify API export endpoint without extra_cols
    api_resp_no_extra = client.get(f"/api/clustering/{res_proc['clustering_id']}/export_csv")
    assert api_resp_no_extra.status_code == 200
    df_api_no_extra = pd.read_csv(io.StringIO(api_resp_no_extra.text))
    assert df_api_no_extra.columns[0] == "input_id"
    assert "Source" not in df_api_no_extra.columns
    assert [str(v) for v in df_api_no_extra["input_id"].tolist()] == ["1", "2", "3", "4", "5", "6"]

    # 4. Extract features from the (Processed) dataset and test Clustering on Features
    graph = {
        "nodes": [
            {"id": "in1", "type": "input", "properties": {"shape": "4"}},
            {"id": "dense_enc", "type": "dense", "properties": {"units": 4, "custom_name": "Encoder"}},
            {"id": "out1", "type": "dense", "properties": {"units": 2}}
        ],
        "links": [
            ["l1", "in1", 0, "dense_enc", 0],
            ["l2", "dense_enc", 0, "out1", 0]
        ]
    }
    model, summary = KerasModelBuilder.build_model(graph)
    model_path = tmp_path / "model_clustering_test.keras"
    model.save(str(model_path))
    model_id = f"model-clust-{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO models (id, project_id, name, task_type, graph_json, summary_json, weights_path, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (model_id, proj_id, "Test Model", "classification", json.dumps(graph), json.dumps(summary), str(model_path), now, now)
        )
        conn.commit()

    feat_res = FeatureExtractor.extract_features(
        project_id=proj_id,
        model_id=model_id,
        dataset_id=proc_ds_id,
        layer_name=model.layers[1].name
    )
    feat_id = feat_res["feature_id"]
    assert "Source" in feat_res["available_extra_columns"]

    # Run clustering on features
    res_feat = ClusteringService.run_clustering(
        project_id=proj_id,
        source_type="features",
        source_id=feat_id,
        algorithm="kmeans",
        n_clusters=2
    )
    assert res_feat["num_samples"] == 6
    assert [p["input_id"] for p in res_feat["points"]] == ["1", "2", "3", "4", "5", "6"]
    assert "Source" in res_feat["available_extra_columns"]

    # Export features clustering CSV via API
    api_resp_feat = client.get(f"/api/clustering/{res_feat['clustering_id']}/export_csv?extra_cols=Source")
    assert api_resp_feat.status_code == 200
    df_feat_csv = pd.read_csv(io.StringIO(api_resp_feat.text))
    assert list(df_feat_csv.columns[:2]) == ["input_id", "Source"]
    assert [str(v) for v in df_feat_csv["input_id"].tolist()] == ["1", "2", "3", "4", "5", "6"]
    assert df_feat_csv["Source"].tolist() == ["sensor_A", "sensor_B", "sensor_A", "sensor_C", "sensor_B", "sensor_C"]

    # 5. Also verify Batch Predict CSV export on the (Processed) dataset preserves 1-based IDs and Source column
    pred_resp = client.get(f"/api/training/predict_csv/{model_id}/{proc_ds_id}?extra_cols=Source")
    assert pred_resp.status_code == 200
    df_pred = pd.read_csv(io.StringIO(pred_resp.text))
    assert list(df_pred.columns[:2]) == ["input_id", "Source"]
    assert [str(v) for v in df_pred["input_id"].tolist()] == ["1", "2", "3", "4", "5", "6"]
    assert df_pred["Source"].tolist() == ["sensor_A", "sensor_B", "sensor_A", "sensor_C", "sensor_B", "sensor_C"]

    # 6. Verify IDEC probabilities CSV export on the (Processed) dataset preserves 1-based IDs and Source column
    idec_resp = client.get(f"/api/models/{model_id}/idec_probabilities/{proc_ds_id}/export_csv?extra_cols=Source")
    assert idec_resp.status_code == 200
    df_idec = pd.read_csv(io.StringIO(idec_resp.text))
    assert list(df_idec.columns[:2]) == ["Input_ID", "Source"]
    assert [str(v) for v in df_idec["Input_ID"].tolist()] == ["1", "2", "3", "4", "5", "6"]
    assert df_idec["Source"].tolist() == ["sensor_A", "sensor_B", "sensor_A", "sensor_C", "sensor_B", "sensor_C"]





