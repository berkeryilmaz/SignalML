import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.database import init_database, ensure_default_project


@pytest.fixture(scope="module")
def client():
    init_database()
    ensure_default_project()
    with TestClient(app) as c:
        yield c


def test_health_endpoints(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "tensorflow_version" in data
    assert "gpu_available" in data
    assert data["storage_writable"] is True


def test_project_crud(client):
    # 1. Create
    res = client.post("/api/projects", json={"name": "API Test Project", "description": "Testing CRUD"})
    assert res.status_code == 200
    p = res.json()
    proj_id = p["id"]

    # 2. Get
    res_get = client.get(f"/api/projects/{proj_id}")
    assert res_get.status_code == 200
    assert res_get.json()["name"] == "API Test Project"

    # 3. Update
    res_up = client.put(f"/api/projects/{proj_id}", json={"name": "Updated API Project"})
    assert res_up.status_code == 200
    assert res_up.json()["name"] == "Updated API Project"


def test_keras_model_validation_endpoint(client):
    valid_graph = {
        "nodes": [
            {"id": "1", "type": "input", "properties": {"shape": "256, 1"}},
            {"id": "2", "type": "conv1d", "properties": {"filters": 16, "kernel_size": 3}},
            {"id": "3", "type": "flatten", "properties": {}},
            {"id": "4", "type": "dense", "properties": {"units": 5, "activation": "softmax"}}
        ],
        "links": [
            ["l1", "1", 0, "2", 0],
            ["l2", "2", 0, "3", 0],
            ["l3", "3", 0, "4", 0]
        ]
    }

    res = client.post("/api/models/validate", json={"graph": valid_graph})
    assert res.status_code == 200
    data = res.json()
    assert data["valid"] is True
    assert data["summary"]["total_params"] > 0


def test_all_frontend_pages_render(client):
    pages = [
        "/dashboard",
        "/projects",
        "/data-manager",
        "/signal-tagger",
        "/signal-processing",
        "/workflow-builder",
        "/nn-builder",
        "/training",
        "/feature-extraction",
        "/clustering",
        "/model-evaluation",
        "/exports"
    ]
    for p in pages:
        r = client.get(p)
        assert r.status_code == 200, f"Page {p} returned status {r.status_code}"
        assert "SignalML" in r.text


def test_model_export_and_import_cycle(client, tmp_path):
    # 1. Build a model via API
    graph = {
        "nodes": [
            {"id": "1", "type": "input", "properties": {"shape": "128, 1"}},
            {"id": "2", "type": "conv1d", "properties": {"filters": 8, "kernel_size": 3, "activation": "relu"}},
            {"id": "3", "type": "flatten", "properties": {}},
            {"id": "4", "type": "dense", "properties": {"units": 2, "activation": "softmax"}}
        ],
        "links": [
            ["l1", "1", 0, "2", 0],
            ["l2", "2", 0, "3", 0],
            ["l3", "3", 0, "4", 0]
        ]
    }
    build_res = client.post("/api/models/build", json={
        "project_id": "default-project",
        "name": "Export Test Model",
        "graph": graph,
        "task_type": "classification"
    })
    assert build_res.status_code == 200
    orig_id = build_res.json()["model_id"]

    # 2. Download exported .keras file
    download_res = client.get(f"/api/exports/model/{orig_id}")
    assert download_res.status_code == 200
    keras_bytes = download_res.content
    assert len(keras_bytes) > 0

    # 3. Download exported graph JSON
    json_export_res = client.get(f"/api/models/{orig_id}/export_json")
    assert json_export_res.status_code == 200
    json_bytes = json_export_res.content
    assert len(json_bytes) > 0

    # 4. Import the .keras file into the project
    import_keras_res = client.post(
        "/api/models/import",
        data={"project_id": "default-project", "name": "Imported Keras Model"},
        files={"file": ("my_exported_model.keras", keras_bytes, "application/octet-stream")}
    )
    assert import_keras_res.status_code == 200
    imported_keras = import_keras_res.json()
    assert imported_keras["name"] == "Imported Keras Model"
    assert imported_keras["total_params"] > 0
    assert imported_keras["model_id"] != orig_id
    assert "graph" in imported_keras
    assert len(imported_keras["graph"]["nodes"]) >= 4
    k_types = [n["type"] for n in imported_keras["graph"]["nodes"]]
    assert "keras/input" in k_types
    assert "keras/conv1d" in k_types

    # 5. Import the .json graph file into the project
    import_json_res = client.post(
        "/api/models/import",
        data={"project_id": "default-project", "name": "Imported JSON Model"},
        files={"file": ("my_exported_graph.json", json_bytes, "application/json")}
    )
    assert import_json_res.status_code == 200
    imported_json = import_json_res.json()
    assert imported_json["name"] == "Imported JSON Model"
    assert imported_json["total_params"] > 0


def assert_api_json_compliant(response):
    """Asserts HTTP 200 and strict RFC 8259 JSON compliance with no NaNs/Infs."""
    import json
    assert response.status_code == 200, f"Expected 200 but got {response.status_code}: {response.text}"
    assert "application/json" in response.headers["content-type"]

    # Must be parseable by standard json.loads
    payload = json.loads(response.text)

    # Must re-serialize with allow_nan=False without raising ValueError
    dumped = json.dumps(payload, allow_nan=False)
    assert dumped is not None
    return payload


def test_api_clustering_run_zero_variance_dataset(client, tmp_path):
    """
    Tests POST /api/clustering/run on zero-variance feature data (all 0s).
    Asserts HTTP 200 (not 500) and no NaNs in explained variance or coordinates.
    """
    import numpy as np
    import pandas as pd
    from app.services.data_manager import DataManager

    proj = ensure_default_project()
    proj_id = proj["id"]

    X = np.zeros((20, 10), dtype=np.float32)
    csv_path = tmp_path / "api_zero_var.csv"
    pd.DataFrame(X).to_csv(csv_path, header=False, index=False)
    ds = DataManager.register_dataset(proj_id, "API Zero Var", csv_path)

    res = client.post(
        "/api/clustering/run",
        json={
            "source_type": "dataset",
            "source_id": ds["id"],
            "algorithm": "kmeans",
            "n_clusters": 3,
            "standardize": True,
            "compute_pca": True
        }
    )

    data = assert_api_json_compliant(res)
    assert data["num_samples"] == 20
    assert data["explained_variance_2d"] == [0.0, 0.0]
    assert data["explained_variance_3d"] == [0.0, 0.0, 0.0]
    for pt in data["points"]:
        assert isinstance(pt["x"], float) and not np.isnan(pt["x"])
        assert isinstance(pt["y"], float) and not np.isnan(pt["y"])
        assert isinstance(pt["z"], float) and not np.isnan(pt["z"])


def test_api_clustering_run_all_nan_features(client):
    """
    Tests POST /api/clustering/run on all-NaN features extracted from a corrupted model.
    Asserts HTTP 200 (not 500) and proper fallback values.
    """
    import uuid
    import json
    from datetime import datetime, timezone
    import numpy as np
    from app.core.config import FEATURES_DIR
    from app.core.database import get_db_connection

    proj = ensure_default_project()
    proj_id = proj["id"]

    feature_id = str(uuid.uuid4())
    dest_dir = FEATURES_DIR / proj_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{feature_id}.npz"

    corrupted_features = np.full((25, 16), np.nan, dtype=np.float32)
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
                feature_id, proj_id, "corrupt_model", "dummy_ds", "layer_nan",
                str(dest_path), json.dumps(list(corrupted_features.shape)),
                25, 16, now
            )
        )
        conn.commit()

    res = client.post(
        "/api/clustering/run",
        json={
            "source_type": "features",
            "source_id": feature_id,
            "algorithm": "kmeans",
            "n_clusters": 2,
            "standardize": True,
            "compute_pca": True
        }
    )

    data = assert_api_json_compliant(res)
    assert data["num_samples"] == 25
    assert data["explained_variance_2d"] == [0.0, 0.0]


def test_api_clustering_list_prevents_db_poisoning(client, tmp_path):
    """
    Ensures that running clustering on degenerate data does not poison SQLite metrics_json
    with NaNs, and that GET /api/clustering/list/{project_id} returns HTTP 200.
    """
    import numpy as np
    proj = ensure_default_project()
    proj_id = proj["id"]

    # Retrieve clustering list
    res = client.get(f"/api/clustering/list/{proj_id}")
    data = assert_api_json_compliant(res)
    assert isinstance(data, list)
    for run in data:
        metrics = run.get("metrics", {})
        for k, v in metrics.items():
            if isinstance(v, float):
                assert not np.isnan(v), f"Found NaN metric {k} in DB history"


def test_api_clustering_waveforms_endpoint(client, tmp_path):
    """
    Tests GET /api/clustering/group_waveforms/{clustering_id}/{cluster_label}
    for HTTP 200 on valid clusters, and HTTP 400 on non-existent clusters.
    """
    import numpy as np
    import pandas as pd
    from app.services.data_manager import DataManager

    proj = ensure_default_project()
    proj_id = proj["id"]

    X = np.random.randn(10, 50).astype(np.float32)
    csv_path = tmp_path / "api_waveforms.csv"
    pd.DataFrame(X).to_csv(csv_path, header=False, index=False)
    ds = DataManager.register_dataset(proj_id, "API Waveforms", csv_path)

    run_res = client.post(
        "/api/clustering/run",
        json={
            "source_type": "dataset",
            "source_id": ds["id"],
            "algorithm": "kmeans",
            "n_clusters": 2,
            "compute_pca": True
        }
    )
    assert run_res.status_code == 200
    clustering_id = run_res.json()["clustering_id"]

    # 1. Valid cluster
    wf_res = client.get(f"/api/clustering/group_waveforms/{clustering_id}/0")
    wf_data = assert_api_json_compliant(wf_res)
    assert wf_data["cluster_label"] == 0

    # 2. Non-existent cluster returns 400 (not 500)
    err_res = client.get(f"/api/clustering/group_waveforms/{clustering_id}/999")
    assert err_res.status_code == 400
    assert "detail" in err_res.json()


def test_import_wrapped_and_keras_native_json(client):
    """
    Tests importing:
    1. Wrapped graph JSON: {"graph": {"nodes": [...], "links": [...]}}
    2. Native Keras Model JSON architecture (from model.to_json())
    Verifies that returned payload contains visual LiteGraph nodes and links.
    """
    import json
    import keras
    from keras import layers

    # 1. Test wrapped graph JSON import
    wrapped_graph = {
        "graph": {
            "nodes": [
                {"id": "1", "type": "input", "properties": {"shape": "64, 1"}},
                {"id": "2", "type": "conv1d", "properties": {"filters": 8, "kernel_size": 3, "activation": "relu"}},
                {"id": "3", "type": "global_maxpooling1d", "properties": {}},
                {"id": "4", "type": "dense", "properties": {"units": 3, "activation": "softmax"}}
            ],
            "links": [
                ["l1", "1", 0, "2", 0],
                ["l2", "2", 0, "3", 0],
                ["l3", "3", 0, "4", 0]
            ]
        }
    }
    wrapped_json_bytes = json.dumps(wrapped_graph).encode("utf-8")

    res_wrapped = client.post(
        "/api/models/import",
        data={"project_id": "default-project", "name": "Wrapped Graph Model"},
        files={"file": ("wrapped_model.json", wrapped_json_bytes, "application/json")}
    )
    assert res_wrapped.status_code == 200
    data_wrapped = res_wrapped.json()
    assert data_wrapped["name"] == "Wrapped Graph Model"
    assert "graph" in data_wrapped
    assert len(data_wrapped["graph"]["nodes"]) == 4

    # 2. Test native Keras model.to_json() export import
    inp = layers.Input(shape=(64, 1), name="signal_input")
    x = layers.Conv1D(16, 5, padding="same", activation="relu", name="conv_layer")(inp)
    x = layers.BatchNormalization(name="bn_layer")(x)
    x = layers.GlobalAveragePooling1D(name="gap_layer")(x)
    out = layers.Dense(2, activation="softmax", name="output_layer")(x)
    keras_model = keras.Model(inputs=inp, outputs=out, name="SignalClassifier")

    keras_json_str = keras_model.to_json()
    keras_json_bytes = keras_json_str.encode("utf-8")

    res_keras_json = client.post(
        "/api/models/import",
        data={"project_id": "default-project", "name": "Native Keras Imported Model"},
        files={"file": ("signal_classifier.json", keras_json_bytes, "application/json")}
    )
    assert res_keras_json.status_code == 200
    data_keras = res_keras_json.json()
    assert data_keras["name"] == "Native Keras Imported Model"
    assert "graph" in data_keras
    # LiteGraph nodes should be generated with mapped types
    nodes = data_keras["graph"]["nodes"]
    assert len(nodes) == 5
    types = [n["type"] for n in nodes]
    assert "keras/input" in types
    assert "keras/conv1d" in types
    assert "keras/batchnorm" in types
    assert "keras/global_averagepooling1d" in types
    assert "keras/dense" in types



