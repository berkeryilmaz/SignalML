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


def test_sanitize_layer_name():
    used = set()
    name1 = KerasModelBuilder.sanitize_layer_name("Conv1D Özellik Çıkarıcı", "conv_fallback", used)
    assert name1 == "Conv1D_Ozellik_Cikarici"
    assert "Conv1D_Ozellik_Cikarici" in used

    # Duplicate should get suffix
    name2 = KerasModelBuilder.sanitize_layer_name("Conv1D Özellik Çıkarıcı", "conv_fallback", used)
    assert name2 == "Conv1D_Ozellik_Cikarici_1"

    # Turkish characters
    name3 = KerasModelBuilder.sanitize_layer_name("Birleşim Katmanı (Öncelikli)", "concat_fallback", used)
    assert name3 == "Birlesim_Katmani_Oncelikli"

    # Empty raw_name uses fallback
    name4 = KerasModelBuilder.sanitize_layer_name("", "fallback_dense", used)
    assert name4 == "fallback_dense"

    # None raw_name uses fallback
    name5 = KerasModelBuilder.sanitize_layer_name(None, "fallback_dense", used)
    assert name5 == "fallback_dense_1"


def test_custom_layer_naming_and_trainable_in_model_builder():
    graph = {
        "nodes": [
            {
                "id": "1",
                "type": "input",
                "properties": {"shape": "64, 1", "custom_name": "Giris_Katmani", "input_source": "sequence"}
            },
            {
                "id": "2",
                "type": "conv1d",
                "properties": {"filters": 8, "kernel_size": 3, "custom_name": "Conv1D_Ozellik_Cikarici", "trainable": False}
            },
            {
                "id": "3",
                "type": "global_maxpooling1d",
                "title": "Havuzlama_Katmani",
                "properties": {"trainable": True}
            },
            {
                "id": "4",
                "type": "dense",
                "properties": {"units": 2, "custom_name": "Siniflandirici_Cikisi", "activation": "softmax", "trainable": True}
            }
        ],
        "links": [
            ["l1", "1", 0, "2", 0],
            ["l2", "2", 0, "3", 0],
            ["l3", "3", 0, "4", 0]
        ]
    }

    model, summary = KerasModelBuilder.build_model(graph)
    assert isinstance(model, keras.Model)

    # Verify custom layer names in Keras
    layer_names = [l.name for l in model.layers]
    assert "Giris_Katmani" in layer_names
    assert "Conv1D_Ozellik_Cikarici" in layer_names
    assert "Havuzlama_Katmani" in layer_names
    assert "Siniflandirici_Cikisi" in layer_names

    # Verify trainable flags
    conv_layer = model.get_layer("Conv1D_Ozellik_Cikarici")
    assert conv_layer.trainable is False

    dense_layer = model.get_layer("Siniflandirici_Cikisi")
    assert dense_layer.trainable is True

    # Check summary items
    layers_dict = {l["layer_name"]: l for l in summary["layers"]}
    assert layers_dict["Conv1D_Ozellik_Cikarici"]["trainable"] is False
    assert layers_dict["Siniflandirici_Cikisi"]["trainable"] is True
    assert layers_dict["Giris_Katmani"]["input_source"] == "sequence"


def test_multi_branch_hybrid_model_with_concatenate_auto_flatten():
    # Multi-branch:
    # Branch 1 (Sequence): Input (64, 1) -> Conv1D (16 filters) -> MaxPooling1D (3D tensor output)
    # Branch 2 (Tabular): Input (8) -> Dense (16 units) (2D tensor output)
    # Merge: Concatenate (mixed ranks! 3D and 2D -> auto_flatten on Branch 1) -> Dense (2)
    graph = {
        "nodes": [
            {"id": "seq_in", "type": "input", "properties": {"shape": "64, 1", "input_source": "sequence", "custom_name": "Seq_In"}},
            {"id": "conv", "type": "conv1d", "properties": {"filters": 8, "kernel_size": 3, "custom_name": "CNN_Branch"}},
            {"id": "pool", "type": "maxpooling1d", "properties": {"pool_size": 2, "custom_name": "Pool_Branch"}},
            {"id": "tab_in", "type": "input", "properties": {"shape": "8", "input_source": "tabular", "custom_name": "Tab_In"}},
            {"id": "dense_tab", "type": "dense", "properties": {"units": 16, "custom_name": "Tab_Dense"}},
            {"id": "concat", "type": "concatenate", "properties": {"axis": -1, "custom_name": "Birlesim_Katmani"}},
            {"id": "out", "type": "dense", "properties": {"units": 2, "activation": "softmax", "custom_name": "Hybrid_Output"}}
        ],
        "links": [
            ["l1", "seq_in", 0, "conv", 0],
            ["l2", "conv", 0, "pool", 0],
            ["l3", "tab_in", 0, "dense_tab", 0],
            ["l4", "pool", 0, "concat", 0],
            ["l5", "dense_tab", 0, "concat", 1],
            ["l6", "concat", 0, "out", 0]
        ]
    }

    model, summary = KerasModelBuilder.build_model(graph)
    assert isinstance(model, keras.Model)
    assert len(model.inputs) == 2
    assert summary["output_shapes"][0] == [None, 2]

    # Verify that an auto_flatten layer was created between pool and concat
    auto_flatten_layers = [l for l in model.layers if "auto_flatten" in l.name]
    assert len(auto_flatten_layers) >= 1

    # Test forward pass with multi-input tensors
    x_seq = np.random.randn(4, 64, 1).astype(np.float32)
    x_tab = np.random.randn(4, 8).astype(np.float32)

    pred = model([x_seq, x_tab])
    assert pred.shape == (4, 2)
    assert np.allclose(np.sum(pred.numpy(), axis=1), 1.0, atol=1e-5)

