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


def test_clustering_layer_and_idec_model_compilation():
    # IDEC Graph: Input -> Conv1D -> MaxPool -> Dense (Latent) -> Branch A: ClusteringLayer -> Output (Q)
    #                                                          -> Branch B: Dense -> Reshape -> Output (X_recon)
    graph = {
        "nodes": [
            {"id": "1", "type": "input", "properties": {"shape": "64, 1"}},
            {"id": "2", "type": "conv1d", "properties": {"filters": 8, "kernel_size": 3}},
            {"id": "3", "type": "flatten", "properties": {}},
            {"id": "4", "type": "dense", "properties": {"units": 16}},
            {"id": "5", "type": "clustering", "properties": {"n_clusters": 4}},
            {"id": "6", "type": "output", "properties": {"activation": "linear"}},
            {"id": "7", "type": "dense", "properties": {"units": 64}},
            {"id": "8", "type": "reshape", "properties": {"target_shape": "64, 1"}},
            {"id": "9", "type": "output", "properties": {"activation": "linear"}}
        ],
        "links": [
            ["l1", "1", 0, "2", 0],
            ["l2", "2", 0, "3", 0],
            ["l3", "3", 0, "4", 0],
            ["l4", "4", 0, "5", 0],
            ["l5", "5", 0, "6", 0],
            ["l6", "4", 0, "7", 0],
            ["l7", "7", 0, "8", 0],
            ["l8", "8", 0, "9", 0]
        ]
    }

    model, summary = KerasModelBuilder.build_model(graph)
    assert isinstance(model, keras.Model)
    assert len(model.outputs) == 2
    assert summary["output_shapes"][0] == [None, 4]
    assert summary["output_shapes"][1] == [None, 64, 1]

    # Verify automatic IDEC multi-output compile (dual losses: kld and mse)
    compiled = KerasModelBuilder.compile_model(model)
    assert isinstance(compiled.loss, (list, tuple)) or hasattr(compiled.loss, "__iter__")
    losses_list = list(compiled.loss)
    assert "kld" in losses_list
    assert "mse" in losses_list

    # Forward pass
    dummy = np.random.randn(3, 64, 1).astype(np.float32)
    preds = model(dummy)
    assert len(preds) == 2
    q_prob, x_recon = preds[0].numpy(), preds[1].numpy()
    assert q_prob.shape == (3, 4)
    assert x_recon.shape == (3, 64, 1)
    # Probabilities should sum to 1.0 along axis=1
    assert np.allclose(np.sum(q_prob, axis=1), 1.0, atol=1e-5)


def test_custom_layer_execution():
    # Input -> Custom Layer -> Dense
    graph = {
        "nodes": [
            {"id": "1", "type": "input", "properties": {"shape": "32"}},
            {"id": "2", "type": "custom_layer", "properties": {"class_name": "CustomSignalLayer", "units": 16, "activation": "relu"}},
            {"id": "3", "type": "dense", "properties": {"units": 2, "activation": "softmax"}}
        ],
        "links": [
            ["l1", "1", 0, "2", 0],
            ["l2", "2", 0, "3", 0]
        ]
    }

    model, summary = KerasModelBuilder.build_model(graph)
    assert isinstance(model, keras.Model)
    assert summary["output_shapes"][0] == [None, 2]

    dummy = np.random.randn(2, 32).astype(np.float32)
    out = model(dummy)
    assert out.shape == (2, 2)
    assert np.allclose(np.sum(out.numpy(), axis=1), 1.0, atol=1e-5)


def test_generate_functional_python_code_idec():
    graph = {
        "nodes": [
            {"id": "1", "type": "input", "properties": {"shape": "64, 1"}},
            {"id": "2", "type": "conv1d", "properties": {"filters": 8, "kernel_size": 3}},
            {"id": "3", "type": "flatten", "properties": {}},
            {"id": "4", "type": "dense", "properties": {"units": 16}},
            {"id": "5", "type": "clustering", "properties": {"n_clusters": 4}},
            {"id": "6", "type": "dense", "properties": {"units": 64}},
            {"id": "7", "type": "reshape", "properties": {"target_shape": "64, 1"}},
            {"id": "8", "type": "output", "properties": {"activation": "linear"}}
        ],
        "links": [
            ["l1", "1", 0, "2", 0],
            ["l2", "2", 0, "3", 0],
            ["l3", "3", 0, "4", 0],
            ["l4", "4", 0, "5", 0],
            ["l5", "4", 0, "6", 0],
            ["l6", "6", 0, "7", 0],
            ["l7", "7", 0, "8", 0]
        ]
    }

    code = KerasModelBuilder.generate_functional_python_code(
        graph_data=graph,
        model_name="Generated_IDEC_Model",
        task_type="classification"
    )
    assert "class ClusteringLayer" in code
    assert "build_Generated_IDEC_Model" in code
    assert "load_and_preprocess_dataset" in code
    assert "Input ID" in code
    assert "target_distribution" in code
    assert "KMeans" in code

    # Execute code in safe scope to verify no syntax or runtime errors
    exec_scope = {}
    exec(code, exec_scope)
    assert "build_Generated_IDEC_Model" in exec_scope
    built_model = exec_scope["build_Generated_IDEC_Model"]()
    assert isinstance(built_model, keras.Model)
    assert len(built_model.outputs) == 2


def test_generate_functional_python_code_hybrid():
    hybrid_graph = {
        "nodes": [
            {"id": "1", "type": "input", "properties": {"shape": "128, 1"}},
            {"id": "2", "type": "conv1d", "properties": {"filters": 8, "kernel_size": 5}},
            {"id": "3", "type": "global_averagepooling1d", "properties": {}},
            {"id": "4", "type": "dense", "properties": {"units": 16}},
            {"id": "5", "type": "input", "properties": {"shape": "8"}},
            {"id": "6", "type": "dense", "properties": {"units": 16}},
            {"id": "7", "type": "concatenate", "properties": {"axis": -1}},
            {"id": "8", "type": "dense", "properties": {"units": 4, "activation": "softmax"}},
            {"id": "9", "type": "output", "properties": {"activation": "softmax"}}
        ],
        "links": [
            ["l1", "1", 0, "2", 0],
            ["l2", "2", 0, "3", 0],
            ["l3", "3", 0, "4", 0],
            ["l4", "5", 0, "6", 0],
            ["l5", "4", 0, "7", 0],
            ["l6", "6", 0, "7", 1],
            ["l7", "7", 0, "8", 0],
            ["l8", "8", 0, "9", 0]
        ]
    }

    code = KerasModelBuilder.generate_functional_python_code(
        graph_data=hybrid_graph,
        model_name="Generated_Hybrid_Model",
        task_type="classification"
    )
    assert "build_Generated_Hybrid_Model" in code
    assert "Concatenate" in code

    exec_scope = {}
    exec(code, exec_scope)
    built_model = exec_scope["build_Generated_Hybrid_Model"]()
    assert isinstance(built_model, keras.Model)
    assert len(built_model.inputs) == 2

