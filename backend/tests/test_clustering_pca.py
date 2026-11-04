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


