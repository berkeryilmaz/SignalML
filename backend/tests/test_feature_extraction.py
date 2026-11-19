import pytest
import json
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app.core.database import init_database, ensure_default_project, get_db_connection
from app.services.data_manager import DataManager
from app.services.keras_model_builder import KerasModelBuilder
from app.services.feature_extraction import FeatureExtractor, FeatureExtractionError


@pytest.fixture(autouse=True)
def setup_db():
    init_database()
    ensure_default_project()


def test_intermediate_layer_feature_extraction(tmp_path):
    proj = ensure_default_project()
    proj_id = proj["id"]

    # 1. Dataset
    X = np.random.randn(10, 64).astype(np.float32)
    csv_path = tmp_path / "feat_data.csv"
    pd.DataFrame(X).to_csv(csv_path, header=False, index=False)
    ds = DataManager.register_dataset(proj_id, "Feat Dataset", csv_path)

    # 2. Model: Input -> Conv1D -> LSTM -> Dense
    graph = {
        "nodes": [
            {"id": "in1", "type": "input", "properties": {"shape": "64, 1"}},
            {"id": "conv1", "type": "conv1d", "properties": {"filters": 16, "kernel_size": 3}},
            {"id": "lstm1", "type": "lstm", "properties": {"units": 32, "return_sequences": False}},
            {"id": "out1", "type": "dense", "properties": {"units": 2}}
        ],
        "links": [
            ["l1", "in1", 0, "conv1", 0],
            ["l2", "conv1", 0, "lstm1", 0],
            ["l3", "lstm1", 0, "out1", 0]
        ]
    }

    model, _ = KerasModelBuilder.build_model(graph)
    model_path = tmp_path / "feat_model.keras"
    model.save(str(model_path))

    model_id = "feat-model-test"
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO models (
                id, project_id, name, task_type, graph_json, summary_json,
                weights_path, trainable_params, non_trainable_params, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            """,
            (model_id, proj_id, "Feat Model", "classification", "{}", "{}", str(model_path), 50, 0)
        )
        conn.commit()

    # 3. Inspect layers
    layers_info = FeatureExtractor.get_model_layers(model_id)
    assert len(layers_info) >= 3
    layer_names = [l["name"] for l in layers_info]
    assert any("lstm" in name.lower() for name in layer_names)

    # Find the LSTM layer name
    target_layer_name = next(name for name in layer_names if "lstm" in name.lower())

    # 4. Extract features from the LSTM layer
    result = FeatureExtractor.extract_features(
        project_id=proj_id,
        model_id=model_id,
        dataset_id=ds["id"],
        layer_name=target_layer_name,
        batch_size=4
    )

    assert result["sample_count"] == 10
    assert result["feature_dim"] == 32
    assert result["shape"] == [10, 32]

    # 5. Load feature matrix and verify
    feat_matrix, labels, record = FeatureExtractor.load_feature_matrix(result["feature_id"])
    assert feat_matrix.shape == (10, 32)
    assert not np.isnan(feat_matrix).any()


def test_dataset_switch_row_count_correctness(tmp_path):
    """
    Verifies that when a model was originally created/trained with Dataset A (10 rows),
    and then feature extraction is requested on Dataset B (35 rows), the extracted
    feature matrix correctly has 35 rows (not the old 10 rows!).
    """
    proj = ensure_default_project()
    proj_id = proj["id"]

    # 1. Dataset A: 10 rows
    X_a = np.random.randn(10, 32).astype(np.float32)
    csv_a = tmp_path / "dataset_a.csv"
    pd.DataFrame(X_a).to_csv(csv_a, header=False, index=False)
    ds_a = DataManager.register_dataset(proj_id, "Dataset A (10 rows)", csv_a)

    # 2. Dataset B: 35 rows
    X_b = np.random.randn(35, 32).astype(np.float32)
    csv_b = tmp_path / "dataset_b.csv"
    pd.DataFrame(X_b).to_csv(csv_b, header=False, index=False)
    ds_b = DataManager.register_dataset(proj_id, "Dataset B (35 rows)", csv_b)

    # 3. Model created with Dataset A explicitly bound in its input node properties
    graph = {
        "nodes": [
            {"id": "in1", "type": "input", "properties": {"shape": "32, 1", "dataset_id": ds_a["id"]}},
            {"id": "conv1", "type": "conv1d", "properties": {"filters": 8, "kernel_size": 3, "custom_name": "My Conv1D"}},
            {"id": "dense1", "type": "dense", "properties": {"units": 2}}
        ],
        "links": [
            ["l1", "in1", 0, "conv1", 0],
            ["l2", "conv1", 0, "dense1", 0]
        ]
    }
    model, summary = KerasModelBuilder.build_model(graph)
    model_path = tmp_path / "bound_model.keras"
    model.save(str(model_path))

    model_id = "model-bound-to-ds-a"
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO models (
                id, project_id, name, task_type, graph_json, summary_json,
                weights_path, trainable_params, non_trainable_params, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            """,
            (model_id, proj_id, "Bound Model", "classification", json.dumps(graph), json.dumps(summary), str(model_path), 50, 0)
        )
        conn.commit()

    # 4. Extract features on Dataset B (35 rows)
    conv_layer_name = model.layers[1].name
    result = FeatureExtractor.extract_features(
        project_id=proj_id,
        model_id=model_id,
        dataset_id=ds_b["id"],
        layer_name=conv_layer_name,
        batch_size=8
    )

    # Row count MUST match Dataset B (35 rows), NOT Dataset A (10 rows)
    assert result["sample_count"] == 35, f"Expected 35 samples from Dataset B, got {result['sample_count']}"
    assert result["shape"][0] == 35

    feat_matrix, _, record = FeatureExtractor.load_feature_matrix(result["feature_id"])
    assert feat_matrix.shape[0] == 35
    assert record["sample_count"] == 35
    assert record["dataset_id"] == ds_b["id"]


def test_save_extracted_features_as_dataset(tmp_path):
    """
    Tests converting extracted features into a registered project dataset,
    verifying row_count and col_count integrity for downstream training.
    """
    proj = ensure_default_project()
    proj_id = proj["id"]

    # 1. Dataset: 20 rows
    X = np.random.randn(20, 16).astype(np.float32)
    csv_path = tmp_path / "data_for_save.csv"
    pd.DataFrame(X).to_csv(csv_path, header=False, index=False)
    ds = DataManager.register_dataset(proj_id, "Source Dataset 20", csv_path)

    # 2. Simple Dense Model
    graph = {
        "nodes": [
            {"id": "in1", "type": "input", "properties": {"shape": "16"}},
            {"id": "dense1", "type": "dense", "properties": {"units": 12}},
            {"id": "out1", "type": "dense", "properties": {"units": 2}}
        ],
        "links": [
            ["l1", "in1", 0, "dense1", 0],
            ["l2", "dense1", 0, "out1", 0]
        ]
    }
    model, summary = KerasModelBuilder.build_model(graph)
    model_path = tmp_path / "save_model.keras"
    model.save(str(model_path))

    model_id = "save-ds-model"
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO models (
                id, project_id, name, task_type, graph_json, summary_json,
                weights_path, trainable_params, non_trainable_params, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            """,
            (model_id, proj_id, "Save DS Model", "classification", json.dumps(graph), json.dumps(summary), str(model_path), 50, 0)
        )
        conn.commit()

    dense_layer = model.layers[1].name
    feat_res = FeatureExtractor.extract_features(
        project_id=proj_id,
        model_id=model_id,
        dataset_id=ds["id"],
        layer_name=dense_layer
    )
    assert feat_res["sample_count"] == 20
    assert feat_res["feature_dim"] == 12

    # 3. Save as project dataset
    saved_ds = FeatureExtractor.save_as_dataset(feat_res["feature_id"], custom_name="Extracted_Features_DS")
    assert saved_ds["row_count"] == 20
    assert saved_ds["col_count"] == 12
    assert saved_ds["name"] == "Extracted_Features_DS"

    # Verify retrieved dataset record in DataManager
    db_ds = DataManager.get_dataset(saved_ds["dataset_id"])
    assert db_ds is not None
    assert db_ds["row_count"] == 20
    assert db_ds["col_count"] == 12
    assert db_ds["metadata"].get("source") == "feature_extraction"
    assert db_ds["metadata"].get("feature_id") == feat_res["feature_id"]


def test_feature_extraction_api_endpoints(tmp_path):
    """
    Tests full API lifecycle: layer inspection, extraction with dataset switch,
    listing with dataset metadata, save as dataset, CSV export, rename, and delete.
    """
    client = TestClient(app)
    proj = ensure_default_project()
    proj_id = proj["id"]

    # Datasets
    X1 = np.random.randn(8, 16).astype(np.float32)
    csv1 = tmp_path / "ds1.csv"
    pd.DataFrame(X1).to_csv(csv1, header=False, index=False)
    ds1 = DataManager.register_dataset(proj_id, "API DS 1", csv1)

    X2 = np.random.randn(18, 16).astype(np.float32)
    csv2 = tmp_path / "ds2.csv"
    pd.DataFrame(X2).to_csv(csv2, header=False, index=False)
    ds2 = DataManager.register_dataset(proj_id, "API DS 2", csv2)

    # Model bound to ds1
    graph = {
        "nodes": [
            {"id": "in1", "type": "input", "properties": {"shape": "16", "dataset_id": ds1["id"]}},
            {"id": "dense1", "type": "dense", "properties": {"units": 6, "custom_name": "Encoder Dense"}},
            {"id": "out1", "type": "dense", "properties": {"units": 2}}
        ],
        "links": [
            ["l1", "in1", 0, "dense1", 0],
            ["l2", "dense1", 0, "out1", 0]
        ]
    }
    model, summary = KerasModelBuilder.build_model(graph)
    model_path = tmp_path / "api_model.keras"
    model.save(str(model_path))

    model_id = "api-model-test"
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO models (
                id, project_id, name, task_type, graph_json, summary_json,
                weights_path, trainable_params, non_trainable_params, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            """,
            (model_id, proj_id, "API Test Model", "classification", json.dumps(graph), json.dumps(summary), str(model_path), 50, 0)
        )
        conn.commit()

    # 1. GET layers
    resp_layers = client.get(f"/api/features/layers/{model_id}")
    assert resp_layers.status_code == 200
    layers_data = resp_layers.json()
    assert len(layers_data["layers"]) >= 2
    # Verify custom_name from graph is resolved
    dense_layer_info = next(l for l in layers_data["layers"] if "dense" in l["name"].lower())
    assert dense_layer_info["custom_name"] in ("Encoder Dense", "Encoder_Dense")

    # 2. POST extract on ds2 (18 rows) despite model being bound to ds1 (8 rows)
    extract_payload = {
        "model_id": model_id,
        "dataset_id": ds2["id"],
        "layer_name": dense_layer_info["name"],
        "batch_size": 4
    }
    resp_extract = client.post("/api/features/extract", json=extract_payload)
    assert resp_extract.status_code == 200
    extract_data = resp_extract.json()
    assert extract_data["sample_count"] == 18, f"Expected 18 rows, got {extract_data['sample_count']}"
    assert extract_data["feature_dim"] == 6
    feat_id = extract_data["feature_id"]

    # 3. GET list
    resp_list = client.get(f"/api/features/list/{proj_id}")
    assert resp_list.status_code == 200
    list_data = resp_list.json()
    assert len(list_data) >= 1
    item = next(x for x in list_data if x["id"] == feat_id)
    assert item["sample_count"] == 18
    assert item["dataset_name"] == "API DS 2"
    assert item["dataset_row_count"] == 18

    # 4. POST save_dataset
    resp_save_ds = client.post(f"/api/features/{feat_id}/save_dataset", json={"name": "API_Extracted_Features"})
    assert resp_save_ds.status_code == 200
    save_data = resp_save_ds.json()
    assert save_data["row_count"] == 18
    assert save_data["col_count"] == 6

    # 5. GET export_csv
    resp_csv = client.get(f"/api/features/{feat_id}/export_csv")
    assert resp_csv.status_code == 200
    csv_lines = [l for l in resp_csv.text.strip().split("\n") if l.strip()]
    # 1 header line + 18 sample rows
    assert len(csv_lines) == 19

    # 6. PUT rename
    resp_put = client.put(f"/api/features/{feat_id}", json={"name": "Renamed_Layer"})
    assert resp_put.status_code == 200
    assert resp_put.json()["custom_name"] == "Renamed_Layer"

    # 7. DELETE
    resp_del = client.delete(f"/api/features/{feat_id}")
    assert resp_del.status_code == 200
    assert resp_del.json()["status"] == "success"


def test_feature_extraction_non_zero_sample_ids_and_extra_columns(tmp_path):
    """
    Verifies that:
    1. Real non-zero-based sample IDs (e.g. 1, 2, 3, ...) and extra columns (e.g. Source)
       are preserved in FeatureExtractor.extract_features and saved into feature .npz.
    2. FeatureExtractor.save_as_dataset preserves real IDs and extra columns in new dataset .npz and metadata.
    3. GET /api/features/{feature_id}/export_csv exports CSV starting at '1' (not '0'),
       and correctly prepends extra columns when requested via query param ?extra_cols=Source.
    4. Lineage walking resolves IDs and extra columns through (Processed) signal datasets.
    """
    from app.services.signal_processing import run_pipeline
    from app.schemas.signal import SignalProcessRequest, PipelineStep
    from app.api.signals import apply_and_save_pipeline
    import csv
    import io

    client = TestClient(app)
    proj = ensure_default_project()
    proj_id = proj["id"]

    # 1. Create source dataset with real non-zero IDs (1..5) and an extra column 'Source'
    csv_path = tmp_path / "pulse_merged_sim.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "Source", "sig_0", "sig_1", "sig_2", "sig_3"])
        writer.writerow(["1", "sensor_A", 0.1, 0.2, 0.3, 0.4])
        writer.writerow(["2", "sensor_B", 0.5, 0.6, 0.7, 0.8])
        writer.writerow(["3", "sensor_A", 0.9, 1.0, 1.1, 1.2])
        writer.writerow(["4", "sensor_C", 1.3, 1.4, 1.5, 1.6])
        writer.writerow(["5", "sensor_B", 1.7, 1.8, 1.9, 2.0])

    ds = DataManager.register_dataset(
        project_id=proj_id,
        name="Pulse Sim Dataset",
        source_path=csv_path,
        has_header=True
    )

    # Verify DataManager sees available extra columns and real sample IDs starting at 1
    extra_cols = DataManager.get_available_extra_columns(ds["id"])
    assert "Source" in extra_cols
    sids = DataManager.load_sample_ids(ds["id"])
    assert sids == ["1", "2", "3", "4", "5"]

    # 2. Build simple model: Input(4) -> Dense(4) -> Dense(2)
    graph = {
        "nodes": [
            {"id": "in1", "type": "input", "properties": {"shape": "4"}},
            {"id": "dense_feat", "type": "dense", "properties": {"units": 4, "custom_name": "Encoder"}},
            {"id": "out1", "type": "dense", "properties": {"units": 2}}
        ],
        "links": [
            ["l1", "in1", 0, "dense_feat", 0],
            ["l2", "dense_feat", 0, "out1", 0]
        ]
    }
    model, summary = KerasModelBuilder.build_model(graph)
    model_path = tmp_path / "model_extra_cols.keras"
    model.save(str(model_path))

    model_id = "model-extra-cols-test"
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO models (
                id, project_id, name, task_type, graph_json, summary_json,
                weights_path, trainable_params, non_trainable_params, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            """,
            (model_id, proj_id, "Model Extra Cols", "classification", json.dumps(graph), json.dumps(summary), str(model_path), 50, 0)
        )
        conn.commit()

    # 3. Extract features directly from ds
    layer_name = model.layers[1].name
    res = FeatureExtractor.extract_features(
        project_id=proj_id,
        model_id=model_id,
        dataset_id=ds["id"],
        layer_name=layer_name
    )

    assert "available_extra_columns" in res
    assert "Source" in res["available_extra_columns"]
    assert res["sample_count"] == 5
    feat_id = res["feature_id"]

    # Inspect feature .npz directly
    with get_db_connection() as conn:
        f_row = conn.execute("SELECT file_path FROM features WHERE id = ?", (feat_id,)).fetchone()
    with np.load(f_row["file_path"], allow_pickle=False) as npz:
        assert "sample_ids" in npz
        npz_sids = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in npz["sample_ids"]]
        assert npz_sids == ["1", "2", "3", "4", "5"]
        assert "extra_columns_json" in npz
        loaded_extra = json.loads(str(npz["extra_columns_json"].item()))
        assert loaded_extra.get("Source") == ["sensor_A", "sensor_B", "sensor_A", "sensor_C", "sensor_B"]

    # 4. Save extracted features as dataset
    saved_ds = FeatureExtractor.save_as_dataset(feat_id, custom_name="Saved_Feats_With_Source")
    assert saved_ds["row_count"] == 5
    assert saved_ds["metadata"]["original_dataset_id"] == ds["id"]
    assert "Source" in saved_ds["metadata"]["extra_columns"]

    # Verify sample IDs in saved dataset .npz
    saved_db_ds = DataManager.get_dataset(saved_ds["dataset_id"])
    with np.load(saved_db_ds["file_path"], allow_pickle=False) as snpz:
        saved_sids = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in snpz["sample_ids"]]
        assert saved_sids == ["1", "2", "3", "4", "5"]

    # 5. Test CSV export with extra_cols="Source"
    resp_csv = client.get(f"/api/features/{feat_id}/export_csv?extra_cols=Source")
    assert resp_csv.status_code == 200
    df_exported = pd.read_csv(io.StringIO(resp_csv.text))
    assert list(df_exported.columns[:2]) == ["input_id", "Source"]
    assert [str(v) for v in df_exported["input_id"].tolist()] == ["1", "2", "3", "4", "5"]
    assert df_exported["Source"].tolist() == ["sensor_A", "sensor_B", "sensor_A", "sensor_C", "sensor_B"]

    # Test CSV export without extra_cols
    resp_csv_no_extra = client.get(f"/api/features/{feat_id}/export_csv")
    assert resp_csv_no_extra.status_code == 200
    df_no_extra = pd.read_csv(io.StringIO(resp_csv_no_extra.text))
    assert df_no_extra.columns[0] == "input_id"
    assert "Source" not in df_no_extra.columns
    assert [str(v) for v in df_no_extra["input_id"].tolist()] == ["1", "2", "3", "4", "5"]

    # 6. Test lineage through a (Processed) signal dataset
    proc_res = apply_and_save_pipeline(SignalProcessRequest(
        dataset_id=ds["id"],
        pipeline=[PipelineStep(name="to_fixed_length", params={"target_length": 4})],
        save_as_new_dataset=True,
        new_dataset_name="Pulse Sim (Processed)"
    ))
    proc_ds_id = proc_res["new_dataset_id"]

    # Verify proc dataset preserved IDs and extra columns
    proc_sids = DataManager.load_sample_ids(proc_ds_id)
    assert proc_sids == ["1", "2", "3", "4", "5"]
    assert DataManager.get_available_extra_columns(proc_ds_id) == ["Source"]

    # Now extract features from the (Processed) dataset
    res_proc_feats = FeatureExtractor.extract_features(
        project_id=proj_id,
        model_id=model_id,
        dataset_id=proc_ds_id,
        layer_name=layer_name
    )
    assert "Source" in res_proc_feats["available_extra_columns"]

    # Export CSV from the features extracted on the (Processed) dataset
    resp_proc_csv = client.get(f"/api/features/{res_proc_feats['feature_id']}/export_csv?extra_cols=Source")
    assert resp_proc_csv.status_code == 200
    df_proc_feats = pd.read_csv(io.StringIO(resp_proc_csv.text))
    assert list(df_proc_feats.columns[:2]) == ["input_id", "Source"]
    assert [str(v) for v in df_proc_feats["input_id"].tolist()] == ["1", "2", "3", "4", "5"]
    assert df_proc_feats["Source"].tolist() == ["sensor_A", "sensor_B", "sensor_A", "sensor_C", "sensor_B"]
