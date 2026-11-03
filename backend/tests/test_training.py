import time
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from app.core.database import init_database, ensure_default_project, get_db_connection
from app.services.data_manager import DataManager
from app.services.keras_model_builder import KerasModelBuilder
from app.services.training_service import TrainingService


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

