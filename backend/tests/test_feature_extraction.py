import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from app.core.database import init_database, ensure_default_project, get_db_connection
from app.services.data_manager import DataManager
from app.services.keras_model_builder import KerasModelBuilder
from app.services.feature_extraction import FeatureExtractor


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
