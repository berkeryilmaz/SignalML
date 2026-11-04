import os
import time
import json
import uuid
from typing import Dict, Any, List, Optional
import numpy as np

from app.core.config import EXPORTS_DIR, DATASETS_DIR
from app.core.database import get_db_connection
from app.services.node_engine import NodeEngine, NodeValidationError, NODE_CATALOG, PortType

# Ensure clustering_idec node is registered in node catalog
if "clustering_idec" not in NODE_CATALOG:
    NODE_CATALOG["clustering_idec"] = {
        "display_name": "IDEC Deep Clustering",
        "category": "Clustering & Analysis",
        "description": "Deep Embedded Clustering with Autoencoder neural network",
        "inputs": [{"name": "features", "type": PortType.FEATURE_MATRIX}],
        "outputs": [{"name": "cluster_results", "type": PortType.CLUSTER_RESULTS}],
        "parameters": {
            "model_id": {"type": "string", "default": ""},
            "n_clusters": {"type": "int", "default": 5, "min": 2},
            "epochs": {"type": "int", "default": 15, "min": 0},
            "pretrain_epochs": {"type": "int", "default": 5, "min": 0},
            "gamma": {"type": "float", "default": 0.1, "min": 0.01}
        }
    }
from app.services.data_manager import DataManager
from app.services.signal_processing import (
    to_fixed_length,
    truncate_signals,
    pad_signals,
    peak_asymmetric_window,
    normalize_zscore,
    normalize_minmax,
    export_signals
)
from app.services.feature_extraction import FeatureExtractor
from app.services.clustering_service import ClusteringService
from datetime import datetime, timezone


class WorkflowEngine:
    @staticmethod
    def execute_workflow(project_id: str, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a general-purpose workflow DAG step by step.
        Resolves dependencies and passes data through typed slots.
        """
        validation = NodeEngine.validate_graph(graph_data)
        if not validation.valid:
            raise NodeValidationError("; ".join(validation.errors))

        nodes_by_id, links = NodeEngine.parse_graph(graph_data)
        exec_order = validation.execution_order

        # link_map: (target_node_id, target_slot) -> (origin_node_id, origin_slot)
        link_map: Dict[Tuple[str, int], Tuple[str, int]] = {}
        for link in links:
            t_id = link["target_id"]
            t_slot = link["target_slot"]
            o_id = link["origin_id"]
            o_slot = link["origin_slot"]
            link_map[(t_id, t_slot)] = (o_id, o_slot)

        # Runtime storage: node_id -> dict of slot_idx -> output_value
        runtime_outputs: Dict[str, Dict[int, Any]] = {}
        node_status: Dict[str, Dict[str, Any]] = {}

        total_start = time.time()

        for nid in exec_order:
            node = nodes_by_id[nid]
            ntype = node.get("type", "")
            props = node.get("properties", {})
            runtime_outputs[nid] = {}

            node_start = time.time()
            status_entry = {
                "node_id": nid,
                "node_type": ntype,
                "status": "running",
                "preview": None,
                "error": None
            }

            try:
                # Helper to fetch input for this node
                def get_input(slot_idx: int) -> Any:
                    key = (nid, slot_idx)
                    if key not in link_map:
                        return None
                    src_id, src_slot = link_map[key]
                    if src_id not in runtime_outputs or src_slot not in runtime_outputs[src_id]:
                        return None
                    return runtime_outputs[src_id][src_slot]

                # 1. Data Source Node
                if ntype in ("data_source", "DataSource"):
                    ds_id = props.get("dataset_id")
                    if not ds_id:
                        # Fallback to project's active dataset
                        with get_db_connection() as conn:
                            proj = conn.execute("SELECT active_dataset_id FROM projects WHERE id = ?", (project_id,)).fetchone()
                            if proj and proj["active_dataset_id"]:
                                ds_id = proj["active_dataset_id"]
                    if not ds_id:
                        raise ValueError("Data Source node requires a dataset selection.")

                    data, labels, record = DataManager.load_dataset_arrays(ds_id)
                    runtime_outputs[nid][0] = {"data": data, "labels": labels, "dataset_id": ds_id, "name": record["name"]}
                    status_entry["preview"] = f"Loaded {len(data)} samples ({data.shape[1]} dims)"

                # 2. CSV Parser Node
                elif ntype in ("csv_parser", "CSVParser"):
                    inp = get_input(0)
                    if inp is None:
                        raise ValueError("CSV Parser requires a Data Source connection.")
                    data = inp["data"]
                    labels = inp.get("labels")
                    runtime_outputs[nid][0] = data
                    runtime_outputs[nid][1] = labels
                    status_entry["preview"] = f"Signals: {data.shape}, Labels: {'Yes' if labels is not None else 'None'}"

                # 3. Fixed Length / Signal Pad / Truncate
                elif ntype in ("fixed_length", "to_fixed_length"):
                    signals = get_input(0)
                    if signals is None:
                        raise ValueError("Fixed Length node requires input signals.")
                    target_len = int(props.get("target_length", 512))
                    pad_mode = props.get("pad_mode", "constant")
                    trunc_pos = props.get("truncate_pos", "end")
                    processed = to_fixed_length(signals, target_len, pad_mode=pad_mode, truncate_pos=trunc_pos)
                    runtime_outputs[nid][0] = processed
                    status_entry["preview"] = f"Resized to {processed.shape}"

                elif ntype == "signal_truncate":
                    signals = get_input(0)
                    if signals is None:
                        raise ValueError("Signal Truncate requires input signals.")
                    target_len = int(props.get("target_length", 512))
                    pos = props.get("position", "end")
                    processed = truncate_signals(signals, target_len, position=pos)
                    runtime_outputs[nid][0] = processed
                    status_entry["preview"] = f"Truncated to {processed.shape}"

                elif ntype == "signal_pad":
                    signals = get_input(0)
                    if signals is None:
                        raise ValueError("Signal Pad requires input signals.")
                    target_len = int(props.get("target_length", 1024))
                    mode = props.get("mode", "constant")
                    cval = float(props.get("constant_value", 0.0))
                    processed = pad_signals(signals, target_len, mode=mode, constant_values=cval)
                    runtime_outputs[nid][0] = processed
                    status_entry["preview"] = f"Padded to {processed.shape}"

                elif ntype == "peak_window":
                    signals = get_input(0)
                    if signals is None:
                        raise ValueError("Peak Window requires input signals.")
                    left_m = int(props.get("left_m", 100))
                    right_k = int(props.get("right_k", 200))
                    mode = props.get("mode", "max_abs")
                    processed = peak_asymmetric_window(signals, left_m, right_k, mode=mode)
                    runtime_outputs[nid][0] = processed
                    status_entry["preview"] = f"Window shape: {processed.shape}"

                elif ntype == "normalize":
                    signals = get_input(0)
                    if signals is None:
                        raise ValueError("Normalize requires input signals.")
                    method = str(props.get("method", "zscore")).lower()
                    per_sig_raw = props.get("per_signal", True)
                    if isinstance(per_sig_raw, str):
                        per_sig = per_sig_raw.lower() in ("true", "1", "yes")
                    else:
                        per_sig = bool(per_sig_raw)

                    if method == "minmax":
                        processed = normalize_minmax(signals, per_signal=per_sig)
                    else:
                        processed = normalize_zscore(signals, per_signal=per_sig)
                    runtime_outputs[nid][0] = processed
                    status_entry["preview"] = f"Normalized {processed.shape} (per_waveform={per_sig})"

                # 4. Feature Extractor Node
                elif ntype in ("feature_extractor", "FeatureExtractor"):
                    signals = get_input(1)
                    model_id = props.get("model_id")
                    layer_name = props.get("layer_name")
                    if not model_id or not layer_name:
                        raise ValueError("Feature Extractor requires model_id and layer_name.")

                    # Use temporary feature extraction
                    # (Uses FeatureExtractor service)
                    ds_input = get_input(0)
                    ds_id = ds_input.get("dataset_id") if isinstance(ds_input, dict) else None
                    if not ds_id:
                        # Register temporary array or retrieve from dataset
                        raise ValueError("Feature Extractor requires a dataset source.")

                    res = FeatureExtractor.extract_features(
                        project_id=project_id,
                        model_id=model_id,
                        dataset_id=ds_id,
                        layer_name=layer_name
                    )
                    runtime_outputs[nid][0] = res["feature_id"]
                    status_entry["preview"] = f"Features extracted: shape {res['shape']}"

                # 5. Clustering Nodes
                elif ntype in ("clustering_kmeans", "clustering_dbscan", "clustering_idec"):
                    feat_input = get_input(0)
                    if feat_input is None:
                        raise ValueError("Clustering requires input feature matrix or signals.")

                    # If input is a raw numpy array, save temporary or wrap it
                    source_type = "features" if isinstance(feat_input, str) else "dataset"
                    # For direct arrays:
                    if isinstance(feat_input, np.ndarray):
                        if ntype == "clustering_idec":
                            dataset_id = str(uuid.uuid4())
                            dest_dir = DATASETS_DIR / project_id
                            dest_dir.mkdir(parents=True, exist_ok=True)
                            dest_path = dest_dir / f"{dataset_id}.npz"
                            now = datetime.now(timezone.utc).isoformat()
                            data_arr = feat_input.reshape(feat_input.shape[0], -1) if feat_input.ndim > 2 else feat_input
                            sample_ids = np.array([str(i) for i in range(len(data_arr))], dtype=str)
                            np.savez_compressed(str(dest_path), data=data_arr, sample_ids=sample_ids)
                            with get_db_connection() as conn:
                                conn.execute(
                                    """
                                    INSERT INTO datasets (
                                        id, project_id, name, file_path, file_format,
                                        row_count, col_count, created_at, metadata_json
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                    (
                                        dataset_id, project_id, f"wf_temp_{nid}", str(dest_path), "npz",
                                        int(data_arr.shape[0]), int(data_arr.shape[1]), now, "{}"
                                    )
                                )
                                conn.commit()

                            res = ClusteringService.run_clustering(
                                project_id=project_id,
                                source_type="dataset",
                                source_id=dataset_id,
                                algorithm="idec",
                                params=props
                            )
                            runtime_outputs[nid][0] = res
                            status_entry["preview"] = f"IDEC Clusters: {res['cluster_distribution']}"
                        else:
                            # Run directly on array with NaN / Inf sanitization
                            X = np.nan_to_num(feat_input.reshape(feat_input.shape[0], -1), nan=0.0, posinf=0.0, neginf=0.0)
                            if ntype == "clustering_kmeans":
                                from sklearn.cluster import KMeans
                                n_c = max(1, int(props.get("n_clusters", 3)))
                                actual_k = min(n_c, len(X)) if len(X) > 0 else 1
                                labels = KMeans(n_clusters=actual_k, random_state=42).fit_predict(X)
                            else:
                                from sklearn.cluster import DBSCAN
                                eps = float(props.get("eps", 0.5))
                                min_s = max(1, int(props.get("min_samples", 5)))
                                labels = DBSCAN(eps=eps, min_samples=min_s).fit_predict(X)

                            runtime_outputs[nid][0] = {"labels": labels, "num_clusters": len(np.unique(labels))}
                            status_entry["preview"] = f"Clustered into {len(np.unique(labels))} groups"
                    else:
                        if ntype == "clustering_kmeans":
                            algo = "kmeans"
                        elif ntype == "clustering_dbscan":
                            algo = "dbscan"
                        else:
                            algo = "idec"

                        res = ClusteringService.run_clustering(
                            project_id=project_id,
                            source_type=source_type,
                            source_id=feat_input,
                            algorithm=algo,
                            params=props
                        )
                        runtime_outputs[nid][0] = res
                        status_entry["preview"] = f"Clusters: {res['cluster_distribution']}"

                # 6. PCA Analysis Node
                elif ntype == "pca_analysis":
                    feat_input = get_input(0)
                    if isinstance(feat_input, np.ndarray):
                        from sklearn.decomposition import PCA
                        n_comp = int(props.get("n_components", 3))
                        X = np.nan_to_num(feat_input.reshape(feat_input.shape[0], -1), nan=0.0, posinf=0.0, neginf=0.0)
                        n_c = max(1, min(n_comp, X.shape[1], X.shape[0])) if (X.shape[0] > 0 and X.shape[1] > 0) else 0
                        if n_c >= 1:
                            pca = PCA(n_components=n_c)
                            coords = np.nan_to_num(pca.fit_transform(X), nan=0.0, posinf=0.0, neginf=0.0)
                            runtime_outputs[nid][0] = coords
                            status_entry["preview"] = f"PCA shape: {coords.shape}, exp var: {[round(float(v), 3) if np.isfinite(v) else 0.0 for v in pca.explained_variance_ratio_]}"
                        else:
                            runtime_outputs[nid][0] = np.zeros((X.shape[0], 1), dtype=np.float32)
                            status_entry["preview"] = "PCA completed (empty/zero components)"
                    else:
                        status_entry["preview"] = "PCA completed"

                # 7. Signal Export Node
                elif ntype in ("signal_export", "SignalExport"):
                    signals = get_input(0)
                    labels = get_input(1)
                    if signals is None:
                        raise ValueError("Signal Export node requires input signals.")

                    target_len = int(props.get("target_length", signals.shape[1]))
                    fmt = props.get("output_format", "csv")
                    filename = props.get("filename", "exported_signals")

                    content_bytes, mime, verif = export_signals(
                        signals, labels=labels, output_format=fmt, target_length=target_len
                    )

                    export_id = str(uuid.uuid4())
                    dest_file = EXPORTS_DIR / f"{filename}_{export_id}.{fmt}"
                    with open(dest_file, "wb") as f:
                        f.write(content_bytes)

                    status_entry["preview"] = f"Exported {verif['num_rows']} rows x {verif['num_cols']} cols ({fmt})"

                status_entry["status"] = "success"
                status_entry["duration_ms"] = int((time.time() - node_start) * 1000)

            except Exception as e:
                status_entry["status"] = "error"
                status_entry["error"] = str(e)
                status_entry["duration_ms"] = int((time.time() - node_start) * 1000)
                node_status[nid] = status_entry
                break

            node_status[nid] = status_entry

        total_duration = time.time() - total_start
        overall_success = all(s["status"] == "success" for s in node_status.values())

        return {
            "success": overall_success,
            "total_duration_seconds": round(total_duration, 3),
            "execution_order": exec_order,
            "node_results": node_status
        }
