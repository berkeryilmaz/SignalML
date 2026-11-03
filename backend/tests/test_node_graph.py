import pytest
from app.services.node_engine import NodeEngine, are_types_compatible
from app.schemas.node import PortType, NodeGraph, NodeDefinition, GraphLink


def test_port_type_compatibility():
    assert are_types_compatible(PortType.SIGNAL, PortType.SIGNAL)
    assert are_types_compatible(PortType.SIGNAL, PortType.FEATURE_MATRIX)
    assert are_types_compatible(PortType.SIGNAL, PortType.ANY)
    assert are_types_compatible(PortType.ANY, PortType.TENSOR)
    
    # Incompatible
    assert not are_types_compatible(PortType.MODEL, PortType.SIGNAL)
    assert not are_types_compatible(PortType.CLUSTER_RESULTS, PortType.SIGNAL)


def test_valid_dag_execution_order():
    graph_dict = {
        "nodes": [
            {"id": "1", "type": "data_source", "inputs": [], "outputs": [{"name": "dataset", "type": "Dataset"}]},
            {"id": "2", "type": "csv_parser", "inputs": [{"name": "raw_data", "type": "Dataset"}], "outputs": [{"name": "signals", "type": "Signal"}]},
            {"id": "3", "type": "fixed_length", "inputs": [{"name": "signals", "type": "Signal"}], "outputs": [{"name": "signals", "type": "Signal"}]},
            {"id": "4", "type": "signal_export", "inputs": [{"name": "signals", "type": "Signal"}], "outputs": []}
        ],
        "links": [
            ["l1", "1", 0, "2", 0, "Dataset"],
            ["l2", "2", 0, "3", 0, "Signal"],
            ["l3", "3", 0, "4", 0, "Signal"]
        ]
    }

    result = NodeEngine.validate_graph(graph_dict)
    assert result.valid is True
    assert result.execution_order == ["1", "2", "3", "4"]
    assert len(result.errors) == 0


def test_cycle_detection():
    # 1 -> 2 -> 3 -> 1 (Cycle!)
    cyclic_graph = {
        "nodes": [
            {"id": "1", "type": "fixed_length", "inputs": [{"name": "in", "type": "Signal"}], "outputs": [{"name": "out", "type": "Signal"}]},
            {"id": "2", "type": "normalize", "inputs": [{"name": "in", "type": "Signal"}], "outputs": [{"name": "out", "type": "Signal"}]},
            {"id": "3", "type": "fixed_length", "inputs": [{"name": "in", "type": "Signal"}], "outputs": [{"name": "out", "type": "Signal"}]}
        ],
        "links": [
            ["l1", "1", 0, "2", 0, "Signal"],
            ["l2", "2", 0, "3", 0, "Signal"],
            ["l3", "3", 0, "1", 0, "Signal"]
        ]
    }

    result = NodeEngine.validate_graph(cyclic_graph)
    assert result.valid is False
    assert any("Cycle detected" in err for err in result.errors)


def test_invalid_type_connection():
    bad_type_graph = {
        "nodes": [
            {"id": "1", "type": "nn_model", "inputs": [], "outputs": [{"name": "model", "type": "Model"}]},
            {"id": "2", "type": "signal_export", "inputs": [{"name": "signals", "type": "Signal"}], "outputs": []}
        ],
        "links": [
            ["l1", "1", 0, "2", 0, "Model"]
        ]
    }

    result = NodeEngine.validate_graph(bad_type_graph)
    assert result.valid is False
    assert any("Type mismatch" in err for err in result.errors)
