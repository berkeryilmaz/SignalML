import time
import json
import csv
import io
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app.core.database import init_database, ensure_default_project, get_db_connection
from app.core.config import DATASETS_DIR, MODELS_DIR
from app.services.data_manager import DataManager
from app.services.keras_model_builder import KerasModelBuilder
from app.services.training_service import TrainingService
from app.services.feature_extraction import FeatureExtractor


@pytest.fixture(autouse=True)
def setup_db():
    init_database()
    ensure_default_project()


def test_real_keras_training_on_dataset(tmp_path):
    # 1. Create a reproducible mini dataset (20 samples x 64 points, binary classification)
    np.random.seed(42)
    X = np.random.randn(24, 64).astype(np.float32)
    # class 1 has positive mean, class 0 has negative mean
    X[12:] += 2.0
    y = np.array([0] * 12 + [1] * 12, dtype=np.int32)

    csv_path = tmp_path / "train_data.csv"
    df = pd.DataFrame(X)
    df["label"] = y
    df.to_csv(csv_path, index=False)

    proj = ensure_default_project()
    proj_id = proj["id"]

    ds_record = DataManager.register_dataset(
        project_id=proj_id,
        name="Mini Train Set",
        source_path=csv_path,
        has_header=True,
        label_col="label"
    )

    # 2. Build a real Keras model (Input -> Dense -> Dense)
    graph = {
        "nodes": [
            {"id": "1", "type": "input", "properties": {"shape": "64"}},
            {"id": "2", "type": "dense", "properties": {"units": 16, "activation": "relu"}},
            {"id": "3", "type": "dense", "properties": {"units": 2, "activation": "softmax"}}
        ],
        "links": [
            ["l1", "1", 0, "2", 0],
            ["l2", "2", 0, "3", 0]
        ]
    }

    model, summary = KerasModelBuilder.build_model(graph)
    compiled_model = KerasModelBuilder.compile_model(
        model, optimizer="adam", learning_rate=0.01, loss="sparse_categorical_crossentropy"
    )

    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "test_model.keras"
    compiled_model.save(str(model_path))

    # Register in DB
    model_id = "test-model-id"
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO models (
                id, project_id, name, task_type, graph_json, summary_json,
                weights_path, trainable_params, non_trainable_params, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            """,
            (model_id, proj_id, "Test Model", "classification", "{}", "{}", str(model_path), 100, 0)
        )
        conn.commit()

    # 3. Launch training
    config = {
        "epochs": 5,
        "batch_size": 8,
        "learning_rate": 0.01,
        "val_split": 0.25,
        "shuffle": True,
        "early_stopping": False
    }

    job = TrainingService.start_training(
        project_id=proj_id,
        model_id=model_id,
        dataset_id=ds_record["id"],
        config=config
    )

    assert job.status in ("pending", "running")

    # Wait for completion (worker thread join)
    if job.thread is not None:
        job.thread.join(timeout=30.0)

    assert job.status == "completed"
    assert "loss" in job.history
    assert len(job.history["loss"]) == 5
    # Check that training loss decreased
    assert job.history["loss"][-1] <= job.history["loss"][0] + 0.1

    # Check evaluation metrics
    assert job.evaluation is not None
    assert "accuracy" in job.evaluation
    assert "confusion_matrix" in job.evaluation


def test_training_with_nan_data_and_json_safety(tmp_path):
    # Create dataset with NaNs in row 0
    np.random.seed(42)
    X = np.random.randn(20, 32).astype(np.float32)
    X[0, :] = np.nan  # Corrupted row 0

    csv_path = tmp_path / "nan_data.csv"
    pd.DataFrame(X).to_csv(csv_path, index=False)

    proj = ensure_default_project()
    proj_id = proj["id"]

    ds_record = DataManager.register_dataset(
        project_id=proj_id,
        name="NaN Test Dataset",
        source_path=csv_path,
        has_header=True
    )

    graph = {
        "nodes": [
            {"id": "1", "type": "input", "properties": {"shape": "32"}},
            {"id": "2", "type": "dense", "properties": {"units": 16, "activation": "relu"}},
            {"id": "3", "type": "dense", "properties": {"units": 32, "activation": "linear"}}
        ],
        "links": [
            ["l1", "1", 0, "2", 0],
            ["l2", "2", 0, "3", 0]
        ]
    }
    model, _ = KerasModelBuilder.build_model(graph)
    compiled = KerasModelBuilder.compile_model(model, optimizer="adam", learning_rate=0.01, loss="mse")
    model_path = tmp_path / "nan_model.keras"
    compiled.save(str(model_path))

    import json
    model_id = "nan-test-model"
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO models (
                id, project_id, name, task_type, graph_json, summary_json,
                weights_path, trainable_params, non_trainable_params, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            """,
            (model_id, proj_id, "NaN Model", "regression", json.dumps(graph), "{}", str(model_path), 500, 0)
        )
        conn.commit()

    job = TrainingService.start_training(
        project_id=proj_id,
        model_id=model_id,
        dataset_id=ds_record["id"],
        config={"epochs": 3, "batch_size": 8, "learning_rate": 0.01}
    )
    if job.thread is not None:
        job.thread.join(timeout=30.0)

    # Job must complete without crashing, and job.to_dict() must be 100% JSON compliant
    dict_repr = job.to_dict()
    assert dict_repr is not None
    # Verify strict JSON compliance without allow_nan
    json_str = json.dumps(dict_repr, allow_nan=False)
    assert json_str is not None
    assert dict_repr["status"] in ("completed", "failed")


def test_hybrid_model_multi_input_training_resume_freeze_and_predict_csv(tmp_path):
    # 1. Setup multi-input dataset (Sequence + Tabular)
    np.random.seed(42)
    n_samples = 24
    X_seq = np.random.randn(n_samples, 64, 1).astype(np.float32)
    X_tab = np.random.randn(n_samples, 8).astype(np.float32)
    # class 1 has higher values
    X_seq[12:] += 1.5
    X_tab[12:] += 1.0
    y = np.array([0] * 12 + [1] * 12, dtype=np.int32)
    sample_ids = [f"sample_pt_{i:03d}" for i in range(n_samples)]

    proj = ensure_default_project()
    proj_id = proj["id"]

    ds_dir = tmp_path / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)
    npz_path = ds_dir / "hybrid_test.npz"
    np.savez_compressed(
        str(npz_path),
        X_seq=X_seq,
        X_tabular=X_tab,
        labels=y,
        sample_ids=np.array(sample_ids)
    )

    dataset_id = "hybrid-dataset-01"
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO datasets (
                id, project_id, name, file_path, file_format, row_count, col_count,
                has_header, label_col, created_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?)
            """,
            (dataset_id, proj_id, "Hybrid Multi-Input Dataset", str(npz_path), "npz", n_samples, 72, 0, "labels", "{}")
        )
        conn.commit()

    # 2. Build Multi-Branch Hybrid Model with custom layer names
    graph = {
        "nodes": [
            {"id": "seq_in", "type": "input", "properties": {"shape": "64, 1", "input_source": "sequence", "custom_name": "Seq_Input"}},
            {"id": "cnn", "type": "conv1d", "properties": {"filters": 8, "kernel_size": 3, "custom_name": "Conv1D_Ozellik_Cikarici"}},
            {"id": "pool", "type": "maxpooling1d", "properties": {"pool_size": 2, "custom_name": "Havuzlama"}},
            {"id": "tab_in", "type": "input", "properties": {"shape": "8", "input_source": "tabular", "custom_name": "Tab_Input"}},
            {"id": "dense_tab", "type": "dense", "properties": {"units": 8, "custom_name": "Tabular_Dense"}},
            {"id": "concat", "type": "concatenate", "properties": {"axis": -1, "custom_name": "Birlesim_Katmani"}},
            {"id": "out", "type": "dense", "properties": {"units": 2, "activation": "softmax", "custom_name": "Siniflandirici"}}
        ],
        "links": [
            ["l1", "seq_in", 0, "cnn", 0],
            ["l2", "cnn", 0, "pool", 0],
            ["l3", "tab_in", 0, "dense_tab", 0],
            ["l4", "pool", 0, "concat", 0],
            ["l5", "dense_tab", 0, "concat", 1],
            ["l6", "concat", 0, "out", 0]
        ]
    }

    model, summary = KerasModelBuilder.build_model(graph)
    compiled = KerasModelBuilder.compile_model(model, optimizer="adam", learning_rate=0.01, loss="sparse_categorical_crossentropy")

    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "hybrid_model.keras"
    compiled.save(str(model_path))

    model_id = "hybrid-model-01"
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO models (
                id, project_id, name, task_type, graph_json, summary_json,
                weights_path, trainable_params, non_trainable_params, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            """,
            (model_id, proj_id, "Hybrid CNN-Tabular Model", "classification", json.dumps(graph), json.dumps(summary), str(model_path), summary["trainable_params"], summary["non_trainable_params"])
        )
        conn.commit()

    # 3. Test TrainingService.prepare_model_inputs helper
    prep_X, prep_y, prep_rec = TrainingService.prepare_model_inputs(model, dataset_id)
    assert isinstance(prep_X, list)
    assert len(prep_X) == 2
    assert prep_X[0].shape == (n_samples, 64, 1)
    assert prep_X[1].shape == (n_samples, 8)
    assert prep_y.shape == (n_samples,)
    assert prep_rec["sample_ids"] == sample_ids

    # 4. Phase 1 Training (Initial 3 Epochs)
    initial_config = {
        "epochs": 3,
        "batch_size": 8,
        "learning_rate": 0.01,
        "val_split": 0.25,
        "shuffle": True,
        "early_stopping": False
    }

    job1 = TrainingService.start_training(
        project_id=proj_id,
        model_id=model_id,
        dataset_id=dataset_id,
        config=initial_config
    )
    if job1.thread is not None:
        job1.thread.join(timeout=30.0)

    assert job1.status == "completed"
    assert "loss" in job1.history
    assert len(job1.history["loss"]) == 3

    # 5. Layer Freezing Endpoint test
    client = TestClient(app)
    freeze_resp = client.post(
        f"/api/models/{model_id}/freeze",
        json={"frozen_layers": ["Conv1D_Ozellik_Cikarici"], "learning_rate": 0.001, "optimizer": "adam"}
    )
    assert freeze_resp.status_code == 200
    freeze_data = freeze_resp.json()
    assert freeze_data["success"] is True
    assert freeze_data["non_trainable_params"] > 0

    # Load frozen model to verify layer trainable state
    import keras
    frozen_model = keras.models.load_model(str(model_path))
    cnn_l = frozen_model.get_layer("Conv1D_Ozellik_Cikarici")
    assert cnn_l.trainable is False
    cnn_weights_before_resume = [w.copy() for w in cnn_l.get_weights()]

    dense_l = frozen_model.get_layer("Siniflandirici")
    assert dense_l.trainable is True
    dense_weights_before_resume = [w.copy() for w in dense_l.get_weights()]

    # 6. Resume Training Test (resume=True for 2 additional epochs)
    resume_config = {
        "epochs": 2,
        "batch_size": 8,
        "learning_rate": 0.005,
        "val_split": 0.25,
        "resume": True,
        "frozen_layers": ["Conv1D_Ozellik_Cikarici"]
    }

    job2 = TrainingService.start_training(
        project_id=proj_id,
        model_id=model_id,
        dataset_id=dataset_id,
        config=resume_config
    )
    if job2.thread is not None:
        job2.thread.join(timeout=30.0)

    assert job2.status == "completed"
    assert "loss" in job2.history
    # Continuous history curve: 3 original + 2 resumed = 5 epochs!
    assert len(job2.history["loss"]) == 5

    # Verify weights on frozen layer remained untouched while trainable layer changed
    reloaded_model = keras.models.load_model(str(model_path))
    cnn_weights_after = reloaded_model.get_layer("Conv1D_Ozellik_Cikarici").get_weights()
    for w_b, w_a in zip(cnn_weights_before_resume, cnn_weights_after):
        assert np.allclose(w_b, w_a), "Frozen layer weights must not change during training!"

    dense_weights_after = reloaded_model.get_layer("Siniflandirici").get_weights()
    # At least some trainable weights should differ after training
    assert not all(np.allclose(wb, wa) for wb, wa in zip(dense_weights_before_resume, dense_weights_after)), "Trainable layer weights should update!"

    # 7. Batch Prediction CSV Endpoint Test
    csv_resp = client.get(f"/api/training/predict_csv/{model_id}/{dataset_id}")
    assert csv_resp.status_code == 200
    assert "text/csv" in csv_resp.headers["content-type"]

    csv_reader = csv.reader(io.StringIO(csv_resp.text))
    rows = list(csv_reader)
    assert len(rows) == n_samples + 1  # 1 header + 24 sample rows

    header = rows[0]
    assert header[0] == "input_id"
    assert "predicted_class" in header
    assert "ground_truth" in header
    assert "prob_class_0" in header
    assert "prob_class_1" in header

    for i, row in enumerate(rows[1:]):
        assert row[0] == sample_ids[i]
        assert row[1] in ("0", "1")
        assert row[2] in ("0", "1")
        prob0 = float(row[3])
        prob1 = float(row[4])
        assert np.isclose(prob0 + prob1, 1.0, atol=1e-3)

    # 8. FeatureExtractor on Multi-Input Hybrid Model
    feat_cnn = FeatureExtractor.extract_features(
        project_id=proj_id,
        model_id=model_id,
        dataset_id=dataset_id,
        layer_name="Conv1D_Ozellik_Cikarici"
    )
    assert feat_cnn["sample_count"] == n_samples
    features_mat, _, _ = FeatureExtractor.load_feature_matrix(feat_cnn["feature_id"])
    assert features_mat.shape[0] == n_samples

    feat_concat = FeatureExtractor.extract_features(
        project_id=proj_id,
        model_id=model_id,
        dataset_id=dataset_id,
        layer_name="Birlesim_Katmani"
    )
    assert feat_concat["sample_count"] == n_samples
    features_concat_mat, _, _ = FeatureExtractor.load_feature_matrix(feat_concat["feature_id"])
    assert features_concat_mat.shape[0] == n_samples


def test_dataset_io_schema_endpoint(tmp_path):
    # Setup dataset
    np.random.seed(42)
    n_samples, seq_len = 20, 128
    X = np.sin(np.linspace(0, 4 * np.pi, seq_len))[np.newaxis, :] + np.random.randn(n_samples, seq_len) * 0.1
    y = np.array([0] * 10 + [1] * 10, dtype=np.int32)

    proj = ensure_default_project()
    proj_id = proj["id"]

    ds_dir = tmp_path / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)
    npz_path = ds_dir / "schema_test.npz"
    np.savez_compressed(str(npz_path), data=X.astype(np.float32), labels=y, sample_ids=np.array([f"s_{i}" for i in range(n_samples)]))

    dataset_id = "schema-test-dataset-01"
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO datasets (
                id, project_id, name, file_path, file_format, row_count, col_count,
                has_header, label_col, created_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?)
            """,
            (dataset_id, proj_id, "IO Schema Test Dataset", str(npz_path), "npz", n_samples, seq_len, 0, "labels", "{}")
        )
        conn.commit()

    client = TestClient(app)
    resp = client.get(f"/api/data/{dataset_id}/io_schema")
    assert resp.status_code == 200
    data = resp.json()

    assert data["dataset_id"] == dataset_id
    assert data["row_count"] == n_samples
    assert data["seq_len"] == seq_len
    assert data["has_labels"] is True
    assert data["n_classes"] == 2
    assert len(data["available_signal_channels"]) == 7
    channel_ids = [c["id"] for c in data["available_signal_channels"]]
    assert "raw" in channel_ids
    assert "derivative_1" in channel_ids
    assert "derivative_2" in channel_ids
    assert "region_mask" in channel_ids
    assert "envelope_rms" in channel_ids
    assert "rolling_mean" in channel_ids
    assert "fft_magnitude" in channel_ids

    assert len(data["available_target_sources"]) == 5
    target_ids = [t["id"] for t in data["available_target_sources"]]
    assert "reconstruction" in target_ids
    assert "region_labels" in target_ids
    assert "label_class" in target_ids


def test_multichannel_1d_cnn_derived_training_and_region_mask(tmp_path):
    # Setup test signal with clear events/peaks
    np.random.seed(42)
    n_samples, seq_len = 24, 64
    t = np.linspace(0, 2 * np.pi, seq_len)
    X = np.zeros((n_samples, seq_len), dtype=np.float32)
    for i in range(n_samples):
        X[i] = np.sin(t * (i % 3 + 1))
        # Add peak event in center
        X[i, 28:36] += 3.0

    y = np.array([i % 2 for i in range(n_samples)], dtype=np.int32)

    proj = ensure_default_project()
    proj_id = proj["id"]

    ds_dir = tmp_path / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)
    npz_path = ds_dir / "derived_channels_test.npz"
    np.savez_compressed(str(npz_path), data=X, labels=y, sample_ids=np.array([f"s_{i}" for i in range(n_samples)]))

    dataset_id = "derived-channels-ds-01"
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO datasets (
                id, project_id, name, file_path, file_format, row_count, col_count,
                has_header, label_col, created_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?)
            """,
            (dataset_id, proj_id, "Derived Channels Dataset", str(npz_path), "npz", n_samples, seq_len, 0, "labels", "{}")
        )
        conn.commit()

    # Model with 3 channels: raw, derivative_1, region_mask
    graph = {
        "nodes": [
            {
                "id": "1",
                "type": "input",
                "properties": {
                    "shape": f"{seq_len}, 1",
                    "channels": ["raw", "derivative_1", "region_mask"],
                    "normalize_channels": True
                }
            },
            {"id": "2", "type": "conv1d", "properties": {"filters": 8, "kernel_size": 3, "activation": "relu"}},
            {"id": "3", "type": "global_maxpooling1d", "properties": {}},
            {"id": "4", "type": "dense", "properties": {"units": 2, "activation": "softmax"}},
            {"id": "5", "type": "output", "properties": {"activation": "softmax", "target_source": "label_class"}}
        ],
        "links": [
            ["l1", "1", 0, "2", 0],
            ["l2", "2", 0, "3", 0],
            ["l3", "3", 0, "4", 0],
            ["l4", "4", 0, "5", 0]
        ]
    }

    model, summary = KerasModelBuilder.build_model(graph)
    compiled = KerasModelBuilder.compile_model(model, optimizer="adam", learning_rate=0.01)

    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "derived_model.keras"
    compiled.save(str(model_path))

    model_id = "derived-channels-model-01"
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO models (
                id, project_id, name, task_type, graph_json, summary_json,
                weights_path, trainable_params, non_trainable_params, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            """,
            (model_id, proj_id, "Derived 1D-CNN Model", "classification", json.dumps(graph), json.dumps(summary), str(model_path), 500, 0)
        )
        conn.commit()

    # Test prepare_model_inputs generates (24, 64, 3)
    prep = TrainingService.prepare_model_inputs(model, dataset_id)
    assert prep.X.shape == (n_samples, seq_len, 3)

    # Launch training
    job = TrainingService.start_training(
        project_id=proj_id,
        model_id=model_id,
        dataset_id=dataset_id,
        config={"epochs": 3, "batch_size": 8, "learning_rate": 0.01}
    )
    if job.thread is not None:
        job.thread.join(timeout=30.0)

    assert job.status == "completed"
    assert "loss" in job.history
    assert len(job.history["loss"]) == 3


def test_separate_datasets_per_input_and_output_training(tmp_path):
    # Setup 3 distinct datasets:
    # Dataset A (Raw signal): shape (20, 64)
    # Dataset B (1st derivative): shape (20, 64)
    # Dataset C (Regional labels target): shape (20, 64)
    np.random.seed(42)
    n_samples, seq_len = 20, 64
    raw_sig = np.random.randn(n_samples, seq_len).astype(np.float32)
    d1_sig = np.gradient(raw_sig, axis=1).astype(np.float32)
    mask_sig = (np.abs(raw_sig) > 1.2).astype(np.float32)

    proj = ensure_default_project()
    proj_id = proj["id"]
    ds_dir = tmp_path / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)

    path_a = ds_dir / "ds_a.npz"
    path_b = ds_dir / "ds_b.npz"
    path_c = ds_dir / "ds_c.npz"

    np.savez_compressed(str(path_a), data=raw_sig, sample_ids=np.array([f"s_{i}" for i in range(n_samples)]))
    np.savez_compressed(str(path_b), data=d1_sig, sample_ids=np.array([f"s_{i}" for i in range(n_samples)]))
    np.savez_compressed(str(path_c), data=mask_sig, sample_ids=np.array([f"s_{i}" for i in range(n_samples)]))

    ds_id_a, ds_id_b, ds_id_c = "ds-raw-a", "ds-d1-b", "ds-mask-c"
    with get_db_connection() as conn:
        for ds_id, path_f, name_f in [
            (ds_id_a, path_a, "Raw Dataset"),
            (ds_id_b, path_b, "D1 Dataset"),
            (ds_id_c, path_c, "Mask Dataset")
        ]:
            conn.execute(
                """
                INSERT OR REPLACE INTO datasets (
                    id, project_id, name, file_path, file_format, row_count, col_count,
                    has_header, label_col, created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?)
                """,
                (ds_id, proj_id, name_f, str(path_f), "npz", n_samples, seq_len, 0, None, "{}")
            )
        conn.commit()

    # Build 2-Input model: Input A (ds_id_a) + Input B (ds_id_b) -> Concatenate(axis=-1) -> Conv1D -> Output (ds_id_c)
    graph = {
        "nodes": [
            {"id": "in_a", "type": "input", "properties": {"shape": f"{seq_len}, 1", "dataset_id": ds_id_a, "custom_name": "In_Raw"}},
            {"id": "in_b", "type": "input", "properties": {"shape": f"{seq_len}, 1", "dataset_id": ds_id_b, "custom_name": "In_D1"}},
            {"id": "concat", "type": "concatenate", "properties": {"axis": -1, "custom_name": "Channel_Merge"}},
            {"id": "conv", "type": "conv1d", "properties": {"filters": 8, "kernel_size": 3, "activation": "relu"}},
            {"id": "out_conv", "type": "conv1d", "properties": {"filters": 1, "kernel_size": 1, "activation": "sigmoid"}},
            {"id": "out", "type": "output", "properties": {"activation": "sigmoid", "target_source": "region_labels", "dataset_id": ds_id_c}}
        ],
        "links": [
            ["l1", "in_a", 0, "concat", 0],
            ["l2", "in_b", 0, "concat", 1],
            ["l3", "concat", 0, "conv", 0],
            ["l4", "conv", 0, "out_conv", 0],
            ["l5", "out_conv", 0, "out", 0]
        ]
    }

    model, summary = KerasModelBuilder.build_model(graph)
    compiled = KerasModelBuilder.compile_model(model, optimizer="adam", learning_rate=0.01)

    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "separate_datasets_model.keras"
    compiled.save(str(model_path))

    model_id = "sep-datasets-model-01"
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO models (
                id, project_id, name, task_type, graph_json, summary_json,
                weights_path, trainable_params, non_trainable_params, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            """,
            (model_id, proj_id, "Separate Datasets 1D-CNN Model", "regression", json.dumps(graph), json.dumps(summary), str(model_path), 500, 0)
        )
        conn.commit()

    # Test prepare_model_inputs loads from each respective dataset
    prep = TrainingService.prepare_model_inputs(model, ds_id_a)
    assert isinstance(prep.X, list)
    assert len(prep.X) == 2
    assert prep.X[0].shape == (n_samples, seq_len, 1)
    assert prep.X[1].shape == (n_samples, seq_len, 1)
    # Output target loaded from dataset C (mask)
    assert prep.y.shape == (n_samples, seq_len, 1)

    # Launch training
    job = TrainingService.start_training(
        project_id=proj_id,
        model_id=model_id,
        dataset_id=ds_id_a,
        config={"epochs": 3, "batch_size": 8, "learning_rate": 0.01}
    )
    if job.thread is not None:
        job.thread.join(timeout=30.0)

    assert job.status == "completed"
    assert "loss" in job.history
    assert len(job.history["loss"]) == 3


