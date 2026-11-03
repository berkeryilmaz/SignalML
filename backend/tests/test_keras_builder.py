import pytest
import numpy as np
import keras
from app.services.keras_model_builder import KerasModelBuilder, KerasGraphError


def test_build_conv1d_lstm_dense_model():
    # Input -> Conv1D -> LSTM -> Dense
    graph = {
        "nodes": [
            {"id": "n1", "type": "input", "properties": {"shape": "128, 1"}},
            {"id": "n2", "type": "conv1d", "properties": {"filters": 16, "kernel_size": 3, "activation": "relu"}},
            {"id": "n3", "type": "lstm", "properties": {"units": 32, "return_sequences": False}},
            {"id": "n4", "type": "dense", "properties": {"units": 4, "activation": "softmax"}}
        ],
        "links": [
            ["l1", "n1", 0, "n2", 0],
            ["l2", "n2", 0, "n3", 0],
            ["l3", "n3", 0, "n4", 0]
        ]
    }

    model, summary = KerasModelBuilder.build_model(graph)
    assert isinstance(model, keras.Model)
    assert summary["total_params"] > 0
    assert summary["trainable_params"] > 0
    assert summary["input_shapes"][0] == [None, 128, 1]
    assert summary["output_shapes"][0] == [None, 4]

    # Verify real prediction execution
    dummy_input = np.random.randn(2, 128, 1).astype(np.float32)
    pred = model.predict(dummy_input, verbose=0)
    assert pred.shape == (2, 4)
    assert np.allclose(np.sum(pred, axis=1), 1.0, atol=1e-5)


def test_build_branching_and_concatenate_model():
    # Input -> Branch A (Conv1D) \
    #       -> Branch B (Dense)   -> Concatenate -> Dense
    graph = {
        "nodes": [
            {"id": "1", "type": "input", "properties": {"shape": "64, 1"}},
            {"id": "2", "type": "flatten", "properties": {}},
            {"id": "3", "type": "dense", "properties": {"units": 16, "activation": "relu"}},
            {"id": "4", "type": "conv1d", "properties": {"filters": 8, "kernel_size": 3}},
            {"id": "5", "type": "global_maxpooling1d", "properties": {}},
            {"id": "6", "type": "concatenate", "properties": {"axis": -1}},
            {"id": "7", "type": "dense", "properties": {"units": 2, "activation": "linear"}}
        ],
        "links": [
            ["l1", "1", 0, "2", 0],
            ["l2", "2", 0, "3", 0],
            ["l3", "1", 0, "4", 0],
            ["l4", "4", 0, "5", 0],
            ["l5", "3", 0, "6", 0],
            ["l6", "5", 0, "6", 1],
            ["l7", "6", 0, "7", 0]
        ]
    }

    model, summary = KerasModelBuilder.build_model(graph)
    assert isinstance(model, keras.Model)
    assert summary["output_shapes"][0] == [None, 2]

    # Test forward pass
    dummy = np.random.randn(4, 64, 1).astype(np.float32)
    out = model(dummy)
    assert out.shape == (4, 2)


def test_cycle_detection_in_keras_builder():
    cycle_graph = {
        "nodes": [
            {"id": "1", "type": "input", "properties": {"shape": "32"}},
            {"id": "2", "type": "dense", "properties": {"units": 16}},
            {"id": "3", "type": "dense", "properties": {"units": 16}}
        ],
        "links": [
            ["l1", "1", 0, "2", 0],
            ["l2", "2", 0, "3", 0],
            ["l3", "3", 0, "2", 0]  # Cycle
        ]
    }

    with pytest.raises(KerasGraphError) as exc:
        KerasModelBuilder.build_model(cycle_graph)
    assert "Cycle detected" in str(exc.value)


def test_missing_input_node_error():
    no_input_graph = {
        "nodes": [
            {"id": "1", "type": "dense", "properties": {"units": 10}}
        ],
        "links": []
    }

    with pytest.raises(KerasGraphError) as exc:
        KerasModelBuilder.build_model(no_input_graph)
    assert "must have at least one Input node" in str(exc.value)
