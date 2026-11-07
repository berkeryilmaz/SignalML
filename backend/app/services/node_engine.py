from typing import Dict, List, Any, Optional, Set, Tuple, Union
from collections import defaultdict, deque
from app.schemas.node import PortType, NodeGraph, GraphValidationResult, GraphLink, NodeDefinition


class NodeValidationError(Exception):
    """Raised when node graph validation fails."""
    pass


# Comprehensive Node Catalog for SignalML Workflow Engine
NODE_CATALOG: Dict[str, Dict[str, Any]] = {
    # Data & I/O
    "data_source": {
        "display_name": "Data Source",
        "category": "Data & I/O",
        "description": "Selects a registered dataset from the project",
        "inputs": [],
        "outputs": [{"name": "dataset", "type": PortType.DATASET}],
        "parameters": {
            "dataset_id": {"type": "string", "default": "", "description": "Dataset identifier"}
        }
    },
    "csv_parser": {
        "display_name": "CSV / TXT Parser",
        "category": "Data & I/O",
        "description": "Parses raw text/csv into structured numeric signals",
        "inputs": [{"name": "raw_data", "type": PortType.DATASET}],
        "outputs": [
            {"name": "signals", "type": PortType.SIGNAL},
            {"name": "labels", "type": PortType.LABEL_ARRAY}
        ],
        "parameters": {
            "delimiter": {"type": "string", "default": ","},
            "has_header": {"type": "boolean", "default": False},
            "label_col": {"type": "string", "default": ""}
        }
    },
    # Signal Preprocessing
    "signal_truncate": {
        "display_name": "Signal Truncate",
        "category": "Signal Preprocessing",
        "description": "Truncates signals to specified length",
        "inputs": [{"name": "signals", "type": PortType.SIGNAL}],
        "outputs": [{"name": "signals", "type": PortType.SIGNAL}],
        "parameters": {
            "target_length": {"type": "int", "default": 512, "min": 1},
            "position": {"type": "select", "options": ["end", "start", "center"], "default": "end"}
        }
    },
    "signal_pad": {
        "display_name": "Signal Pad",
        "category": "Signal Preprocessing",
        "description": "Pads signals with zeros, edge, or reflect",
        "inputs": [{"name": "signals", "type": PortType.SIGNAL}],
        "outputs": [{"name": "signals", "type": PortType.SIGNAL}],
        "parameters": {
            "target_length": {"type": "int", "default": 1024, "min": 1},
            "mode": {"type": "select", "options": ["constant", "edge", "reflect"], "default": "constant"},
            "constant_value": {"type": "float", "default": 0.0}
        }
    },
    "fixed_length": {
        "display_name": "To Fixed Length",
        "category": "Signal Preprocessing",
        "description": "Truncates or pads signals to exact length N",
        "inputs": [{"name": "signals", "type": PortType.SIGNAL}],
        "outputs": [{"name": "signals", "type": PortType.SIGNAL}],
        "parameters": {
            "target_length": {"type": "int", "default": 512, "min": 1},
            "pad_mode": {"type": "select", "options": ["constant", "edge", "reflect"], "default": "constant"},
            "truncate_pos": {"type": "select", "options": ["end", "start", "center"], "default": "end"}
        }
    },
    "peak_window": {
        "display_name": "Peak Windowing",
        "category": "Signal Preprocessing",
        "description": "Cuts window of size M on left and K on right of max peak",
        "inputs": [{"name": "signals", "type": PortType.SIGNAL}],
        "outputs": [{"name": "signals", "type": PortType.SIGNAL}],
        "parameters": {
            "left_m": {"type": "int", "default": 100, "min": 0},
            "right_k": {"type": "int", "default": 200, "min": 0},
            "mode": {"type": "select", "options": ["max_abs", "max"], "default": "max_abs"}
        }
    },
    "normalize": {
        "display_name": "Normalize",
        "category": "Signal Preprocessing",
        "description": "Normalizes signals using Min-Max, Z-Score, Median, Mode, Gaussian Peak, Min Scaler, or Max Scaler",
        "inputs": [{"name": "signals", "type": PortType.SIGNAL}],
        "outputs": [{"name": "signals", "type": PortType.SIGNAL}],
        "parameters": {
            "method": {"type": "select", "options": ["zscore", "minmax", "median", "mode", "gaussian_peak", "scale_min", "scale_max"], "default": "zscore"},
            "per_signal": {"type": "boolean", "default": True}
        }
    },
    "reverse_signals": {
        "display_name": "Reverse Filter",
        "category": "Signal Preprocessing",
        "description": "Inverts signal polarity (-x) or reverses time axis (x[::-1])",
        "inputs": [{"name": "signals", "type": PortType.SIGNAL}],
        "outputs": [{"name": "signals", "type": PortType.SIGNAL}],
        "parameters": {
            "mode": {"type": "select", "options": ["polarity", "time", "both"], "default": "polarity"}
        }
    },
    "scale_min": {
        "display_name": "Min Scaler",
        "category": "Signal Preprocessing",
        "description": "Scales signal by its minimum value (x / |min|, x / min, or x - min)",
        "inputs": [{"name": "signals", "type": PortType.SIGNAL}],
        "outputs": [{"name": "signals", "type": PortType.SIGNAL}],
        "parameters": {
            "mode": {"type": "select", "options": ["divide_abs_min", "divide_min", "subtract_min"], "default": "divide_abs_min"},
            "per_signal": {"type": "boolean", "default": True}
        }
    },
    "scale_max": {
        "display_name": "Max Scaler",
        "category": "Signal Preprocessing",
        "description": "Scales signal by its maximum value (x / max or x / max|x|) preserving 0 baseline",
        "inputs": [{"name": "signals", "type": PortType.SIGNAL}],
        "outputs": [{"name": "signals", "type": PortType.SIGNAL}],
        "parameters": {
            "mode": {"type": "select", "options": ["divide_max", "divide_max_abs"], "default": "divide_max"},
            "per_signal": {"type": "boolean", "default": True}
        }
    },
    "subtract_median": {
        "display_name": "Median Baseline Subtraction",
        "category": "Signal Preprocessing",
        "description": "Subtracts signal median so baseline is centered at 0",
        "inputs": [{"name": "signals", "type": PortType.SIGNAL}],
        "outputs": [{"name": "signals", "type": PortType.SIGNAL}],
        "parameters": {
            "per_signal": {"type": "boolean", "default": True}
        }
    },
    "subtract_mode": {
        "display_name": "Mode Baseline Subtraction",
        "category": "Signal Preprocessing",
        "description": "Subtracts the most frequent amplitude (mode) so baseline is centered at 0",
        "inputs": [{"name": "signals", "type": PortType.SIGNAL}],
        "outputs": [{"name": "signals", "type": PortType.SIGNAL}],
        "parameters": {
            "bins": {"type": "int", "default": 100, "min": 0},
            "per_signal": {"type": "boolean", "default": True}
        }
    },
    "subtract_gaussian_peak": {
        "display_name": "Gaussian Peak Baseline",
        "category": "Signal Preprocessing",
        "description": "Builds amplitude histogram, fits a Gaussian to the peak lobe, and subtracts peak center",
        "inputs": [{"name": "signals", "type": PortType.SIGNAL}],
        "outputs": [{"name": "signals", "type": PortType.SIGNAL}],
        "parameters": {
            "bins": {"type": "int", "default": 100, "min": 16},
            "fit_window_bins": {"type": "int", "default": 5, "min": 2},
            "per_signal": {"type": "boolean", "default": True}
        }
    },
    # Neural Networks & Features
    "nn_model": {
        "display_name": "Neural Network Model",
        "category": "Model & Inference",
        "description": "Loads a trained neural network model",
        "inputs": [],
        "outputs": [{"name": "model", "type": PortType.MODEL}],
        "parameters": {
            "model_id": {"type": "string", "default": ""}
        }
    },
    "feature_extractor": {
        "display_name": "Feature Extractor",
        "category": "Model & Inference",
        "description": "Extracts intermediate layer activations from model",
        "inputs": [
            {"name": "model", "type": PortType.MODEL},
            {"name": "signals", "type": PortType.SIGNAL}
        ],
        "outputs": [{"name": "features", "type": PortType.FEATURE_MATRIX}],
        "parameters": {
            "layer_name": {"type": "string", "default": ""},
            "batch_size": {"type": "int", "default": 32}
        }
    },
    # Clustering & Analysis
    "clustering_kmeans": {
        "display_name": "K-Means Clustering",
        "category": "Clustering & Analysis",
        "description": "Partitions features into K clusters",
        "inputs": [{"name": "features", "type": PortType.FEATURE_MATRIX}],
        "outputs": [{"name": "cluster_results", "type": PortType.CLUSTER_RESULTS}],
        "parameters": {
            "n_clusters": {"type": "int", "default": 3, "min": 2},
            "standardize": {"type": "boolean", "default": True},
            "random_state": {"type": "int", "default": 42}
        }
    },
    "clustering_dbscan": {
        "display_name": "DBSCAN Clustering",
        "category": "Clustering & Analysis",
        "description": "Density-based spatial clustering of applications with noise",
        "inputs": [{"name": "features", "type": PortType.FEATURE_MATRIX}],
        "outputs": [{"name": "cluster_results", "type": PortType.CLUSTER_RESULTS}],
        "parameters": {
            "eps": {"type": "float", "default": 0.5, "min": 0.01},
            "min_samples": {"type": "int", "default": 5, "min": 1},
            "standardize": {"type": "boolean", "default": True}
        }
    },
    "pca_analysis": {
        "display_name": "PCA Analysis",
        "category": "Clustering & Analysis",
        "description": "Principal Component Analysis for 2D/3D visualization",
        "inputs": [{"name": "features", "type": PortType.FEATURE_MATRIX}],
        "outputs": [{"name": "pca_coords", "type": PortType.PCA_COORDINATES}],
        "parameters": {
            "n_components": {"type": "int", "default": 3, "min": 2, "max": 3},
            "standardize": {"type": "boolean", "default": True}
        }
    },
    # Export
    "signal_export": {
        "display_name": "Signal Export",
        "category": "Export",
        "description": "Exports signals to exact N length CSV or NPY",
        "inputs": [
            {"name": "signals", "type": PortType.SIGNAL},
            {"name": "labels", "type": PortType.LABEL_ARRAY}
        ],
        "outputs": [],
        "parameters": {
            "target_length": {"type": "int", "default": 512},
            "output_format": {"type": "select", "options": ["csv", "txt", "npy", "npz"], "default": "csv"},
            "filename": {"type": "string", "default": "exported_signals"}
        }
    }
}


def are_types_compatible(source_type: str, target_type: str) -> bool:
    """
    Checks if an output port type can connect to an input port type.
    """
    if source_type == PortType.ANY or target_type == PortType.ANY:
        return True
    if source_type == target_type:
        return True
    
    # Feature matrix can accept numeric arrays and signals
    if target_type in (PortType.FEATURE_MATRIX, PortType.NUMERIC_ARRAY) and source_type in (PortType.SIGNAL, PortType.NUMERIC_ARRAY, PortType.FEATURE_MATRIX):
        return True
    if target_type == PortType.SIGNAL and source_type == PortType.NUMERIC_ARRAY:
        return True
        
    return False


class NodeEngine:
    @staticmethod
    def get_node_catalog() -> Dict[str, Any]:
        """Returns the full catalog of available workflow nodes."""
        return NODE_CATALOG

    @staticmethod
    def parse_graph(graph_data: Union[Dict[str, Any], NodeGraph]) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Normalizes and parses nodes and links from graph payload.
        Handles both LiteGraph and custom graph JSON layouts.
        """
        if isinstance(graph_data, NodeGraph):
            raw_nodes = [n.model_dump() for n in graph_data.nodes]
            raw_links = graph_data.links
        else:
            raw_nodes = graph_data.get("nodes", [])
            raw_links = graph_data.get("links", [])

        nodes_by_id = {}
        for n in raw_nodes:
            n_id = str(n.get("id"))
            nodes_by_id[n_id] = n

        normalized_links = []
        for l in raw_links:
            if isinstance(l, list):
                # LiteGraph format: [id, origin_id, origin_slot, target_id, target_slot, type]
                if len(l) >= 5:
                    normalized_links.append({
                        "id": str(l[0]),
                        "origin_id": str(l[1]),
                        "origin_slot": int(l[2]),
                        "target_id": str(l[3]),
                        "target_slot": int(l[4]),
                        "type": l[5] if len(l) > 5 else "Any"
                    })
            elif isinstance(l, dict):
                normalized_links.append({
                    "id": str(l.get("id", "")),
                    "origin_id": str(l.get("origin_id")),
                    "origin_slot": int(l.get("origin_slot", 0)),
                    "target_id": str(l.get("target_id")),
                    "target_slot": int(l.get("target_slot", 0)),
                    "type": l.get("type", "Any")
                })

        return nodes_by_id, normalized_links

    @staticmethod
    def validate_graph(graph_data: Union[Dict[str, Any], NodeGraph]) -> GraphValidationResult:
        """
        Validates graph structure, checks port types, and detects cycles.
        Returns execution order if graph is an acyclic DAG.
        """
        nodes_by_id, links = NodeEngine.parse_graph(graph_data)
        errors: List[str] = []
        warnings: List[str] = []

        if not nodes_by_id:
            return GraphValidationResult(
                valid=False,
                errors=["Graph is empty. Please add nodes."],
                warnings=[]
            )

        # Adjacency and in-degree for cycle detection
        adj: Dict[str, List[str]] = defaultdict(list)
        in_degree: Dict[str, int] = {n_id: 0 for n_id in nodes_by_id}

        # Check links
        for link in links:
            origin_id = link["origin_id"]
            target_id = link["target_id"]

            if origin_id not in nodes_by_id:
                errors.append(f"Link references non-existent origin node {origin_id}")
                continue
            if target_id not in nodes_by_id:
                errors.append(f"Link references non-existent target node {target_id}")
                continue

            origin_node = nodes_by_id[origin_id]
            target_node = nodes_by_id[target_id]

            # Port boundary checks: fallback to catalog definitions if not in node dict
            origin_type = origin_node.get("type", "")
            target_type = target_node.get("type", "")

            cat_origin = NODE_CATALOG.get(origin_type, {})
            cat_target = NODE_CATALOG.get(target_type, {})

            origin_outputs = origin_node.get("outputs") or cat_origin.get("outputs", [])
            target_inputs = target_node.get("inputs") or cat_target.get("inputs", [])

            if link["origin_slot"] >= len(origin_outputs):
                errors.append(f"Node {origin_id} output slot {link['origin_slot']} out of bounds")
                continue
            if link["target_slot"] >= len(target_inputs):
                errors.append(f"Node {target_id} input slot {link['target_slot']} out of bounds")
                continue

            origin_port = origin_outputs[link["origin_slot"]]
            target_port = target_inputs[link["target_slot"]]

            src_type = origin_port.get("type", "Any")
            dst_type = target_port.get("type", "Any")

            if not are_types_compatible(src_type, dst_type):
                errors.append(
                    f"Type mismatch: cannot connect {origin_node.get('type')} [{src_type}] to {target_node.get('type')} [{dst_type}]"
                )

            adj[origin_id].append(target_id)
            in_degree[target_id] += 1

        # Cycle detection using Kahn's algorithm
        queue = deque([n_id for n_id, deg in in_degree.items() if deg == 0])
        execution_order: List[str] = []

        while queue:
            curr = queue.popleft()
            execution_order.append(curr)

            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(execution_order) < len(nodes_by_id):
            errors.append("Cycle detected in graph! Workflow must be a Directed Acyclic Graph (DAG).")
            return GraphValidationResult(
                valid=False,
                errors=errors,
                warnings=warnings,
                execution_order=[]
            )

        return GraphValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            execution_order=execution_order
        )
