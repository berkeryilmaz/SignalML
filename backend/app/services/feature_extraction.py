import os
import json
import uuid
import copy
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import keras
from app.core.config import FEATURES_DIR, MODELS_DIR, DATASETS_DIR, resolve_storage_path
from app.core.database import get_db_connection
from app.core.responses import sanitize_for_json
from app.services.data_manager import DataManager
from app.services.training_service import TrainingService
from datetime import datetime, timezone


class FeatureExtractionError(Exception):
    """Raised when layer feature extraction fails."""
    pass


class FeatureExtractor:
    @staticmethod
    def get_model_layers(model_id: str) -> List[Dict[str, Any]]:
        """
        Loads the model and inspects all layers, returning their names, types, custom titles, and output shapes.
        """
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
            if not row:
                raise FeatureExtractionError(f"Model {model_id} not found")
            weights_path = resolve_storage_path(row["weights_path"])

        if not weights_path or not weights_path.exists():
            raise FeatureExtractionError(f"Model weights file not found on disk: {weights_path}")

        try:
            model = keras.models.load_model(str(weights_path))
        except Exception as e:
            raise FeatureExtractionError(f"Failed to load Keras model: {str(e)}")

        custom_names = {}
        try:
            sum_data = json.loads(row["summary_json"] or "{}")
            for ls in sum_data.get("layer_summary", []):
                lname = ls.get("layer_name") or ls.get("name")
                cname = ls.get("custom_name")
                if lname and cname:
                    custom_names[lname] = cname
            graph_data = json.loads(row["graph_json"] or "{}")
            for n in graph_data.get("nodes", []):
                nid = n.get("id")
                props = n.get("properties") or {}
                cname = props.get("custom_name") or n.get("title")
                if nid and cname:
                    custom_names[nid] = cname
        except Exception:
            pass

        layers_info = []
        for i, layer in enumerate(model.layers):
            try:
                out_shape = layer.output_shape
                # Convert shape to serializable list
                if isinstance(out_shape, tuple):
                    shape_list = [dim if dim is not None else -1 for dim in out_shape]
                elif isinstance(out_shape, list) and len(out_shape) > 0:
                    shape_list = [dim if dim is not None else -1 for dim in out_shape[0]]
                else:
                    shape_list = []
            except Exception:
                shape_list = []

            cname = custom_names.get(layer.name) or custom_names.get(layer.name.split("/")[0]) or layer.name

            layers_info.append({
                "index": i,
                "name": layer.name,
                "custom_name": cname,
                "class_name": layer.__class__.__name__,
                "output_shape": shape_list,
                "trainable": layer.trainable
            })

        return sanitize_for_json(layers_info)

    @staticmethod
    def extract_features(
        project_id: str,
        model_id: str,
        dataset_id: str,
        layer_name: str,
        sample_limit: Optional[int] = None,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Extracts intermediate layer activations for the specified dataset.
        Ensures the selected dataset_id overrides any stale dataset bindings stored on the model.
        Saves output features into persistent storage.
        """
        with get_db_connection() as conn:
            model_row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
            if not model_row:
                raise FeatureExtractionError(f"Model {model_id} not found")
            weights_path = resolve_storage_path(model_row["weights_path"])

        if not weights_path or not weights_path.exists():
            raise FeatureExtractionError(f"Model weights file not found: {weights_path}")

        try:
            from app.services.keras_model_builder import ClusteringLayer
            full_model = keras.models.load_model(str(weights_path), custom_objects={"ClusteringLayer": ClusteringLayer})
        except Exception as e:
            raise FeatureExtractionError(f"Could not load model: {str(e)}")

        # Find target layer
        try:
            target_layer = full_model.get_layer(layer_name)
        except ValueError:
            valid_names = [l.name for l in full_model.layers]
            raise FeatureExtractionError(
                f"Layer '{layer_name}' not found. Available layers: {', '.join(valid_names)}"
            )

        # Create submodel: inputs -> target_layer.output
        try:
            extractor_model = keras.Model(inputs=full_model.inputs, outputs=target_layer.output)
        except Exception as e:
            raise FeatureExtractionError(f"Failed to create submodel for layer '{layer_name}': {str(e)}")

        # Override any stale dataset_id stored in the model summary/graph so the selected dataset is used
        model_rec = dict(model_row) if model_row else {}
        if "summary_json" in model_rec and model_rec["summary_json"]:
            try:
                s_data = json.loads(model_rec["summary_json"])
                if "input_configs" in s_data and isinstance(s_data["input_configs"], list):
                    for ic in s_data["input_configs"]:
                        if isinstance(ic, dict):
                            ic["dataset_id"] = dataset_id
                    model_rec["summary_json"] = json.dumps(s_data)
            except Exception:
                pass
        if "graph_json" in model_rec and model_rec["graph_json"]:
            try:
                g_data = json.loads(model_rec["graph_json"])
                if "nodes" in g_data and isinstance(g_data["nodes"], list):
                    for n in g_data["nodes"]:
                        if n.get("type") == "input" and "properties" in n and isinstance(n["properties"], dict):
                            n["properties"]["dataset_id"] = dataset_id
                    model_rec["graph_json"] = json.dumps(g_data)
            except Exception:
                pass

        if hasattr(full_model, "_input_configs") and full_model._input_configs:
            updated_cfgs = copy.deepcopy(full_model._input_configs)
            for ic in updated_cfgs:
                if isinstance(ic, dict):
                    ic["dataset_id"] = dataset_id
            full_model._input_configs = updated_cfgs
        else:
            try:
                s_data = json.loads(model_rec.get("summary_json") or "{}")
                if "input_configs" in s_data and isinstance(s_data["input_configs"], list):
                    full_model._input_configs = s_data["input_configs"]
            except Exception:
                pass

        # Load dataset prepared for full_model (supports single, multi-channel, and multi-input)
        data, labels, dataset_record = TrainingService.prepare_model_inputs(full_model, dataset_id, model_record=model_rec)
        num_samples = len(data[0]) if isinstance(data, list) else (len(data) if data is not None else 0)
        if num_samples == 0:
            raise FeatureExtractionError(f"Target dataset '{dataset_id}' contains no samples for feature extraction.")

        if sample_limit is not None and sample_limit > 0 and sample_limit < num_samples:
            if isinstance(data, list):
                data = [arr[:sample_limit] for arr in data]
            else:
                data = data[:sample_limit]
            if labels is not None:
                labels = labels[:sample_limit]
            num_samples = min(num_samples, sample_limit)

        # Run batch prediction
        features_list = []
        for i in range(0, num_samples, batch_size):
            if isinstance(data, list):
                batch = [arr[i : i + batch_size] for arr in data]
            else:
                batch = data[i : i + batch_size]
            batch_output = extractor_model.predict_on_batch(batch)
            if hasattr(batch_output, "numpy"):
                batch_output = batch_output.numpy()
            features_list.append(batch_output)

        if len(features_list) == 0:
            raise FeatureExtractionError("No features extracted from submodel.")

        features = np.concatenate(features_list, axis=0)

        # Sanitize any NaNs or Infs from intermediate layer activations
        if np.isnan(features).any() or np.isinf(features).any():
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Save extracted features
        feature_id = str(uuid.uuid4())
        dest_dir = FEATURES_DIR / project_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"{feature_id}.npz"

        # Resolve dataset file path to extract original clean 1D labels
        clean_labels = None
        try:
            ds_path = resolve_storage_path(dataset_record.get("file_path"))
            if not ds_path or not ds_path.exists():
                cand = DATASETS_DIR / dataset_record.get("project_id", "") / f"{dataset_id}.npz"
                if cand.exists():
                    ds_path = cand
            if ds_path and ds_path.exists():
                with np.load(str(ds_path), allow_pickle=False) as npz:
                    if "labels" in npz:
                        raw_l = npz["labels"]
                        if isinstance(raw_l, np.ndarray):
                            if raw_l.ndim == 1:
                                clean_labels = raw_l
                            elif raw_l.ndim == 2 and raw_l.shape[1] == 1:
                                clean_labels = raw_l.squeeze(1)
        except Exception:
            clean_labels = None

        if clean_labels is not None and sample_limit is not None and sample_limit > 0:
            clean_labels = clean_labels[:sample_limit]

        # Load real sample IDs and extra metadata columns via DataManager lineage walking
        try:
            sample_ids = DataManager.load_sample_ids(dataset_id, expected_count=num_samples)
        except Exception:
            sample_ids = [str(i) for i in range(num_samples)]

        extra_cols_dict = {}
        try:
            extra_cols_dict = DataManager.load_extra_columns(dataset_id, expected_count=num_samples)
        except Exception:
            extra_cols_dict = {}

        save_dict = {"features": features}
        if clean_labels is not None and len(clean_labels) == num_samples:
            if isinstance(clean_labels, np.ndarray) and (np.isnan(clean_labels).any() or np.isinf(clean_labels).any()):
                clean_labels = np.nan_to_num(clean_labels, nan=0.0, posinf=0.0, neginf=0.0)
            save_dict["labels"] = clean_labels
        if sample_ids is not None and len(sample_ids) == num_samples:
            save_dict["sample_ids"] = np.array(sample_ids, dtype=str)
        if extra_cols_dict:
            save_dict["extra_columns_json"] = np.array(json.dumps(extra_cols_dict))

        np.savez_compressed(str(dest_path), **save_dict)

        # Compute 2D flattened representation for clustering/PCA
        if features.ndim > 2:
            flat_dim = int(np.prod(features.shape[1:]))
        else:
            flat_dim = int(features.shape[1]) if features.ndim == 2 else 1

        now = datetime.now(timezone.utc).isoformat()
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO features (
                    id, project_id, model_id, dataset_id, layer_id,
                    file_path, shape_json, sample_count, feature_dim, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feature_id, project_id, model_id, dataset_id, layer_name,
                    str(dest_path), json.dumps(list(features.shape)),
                    num_samples, flat_dim, now
                )
            )
            conn.commit()

        # Generate sample preview
        preview_sample: List[Any] = []
        if len(features) > 0:
            raw_sample = features[0].tolist()
            if isinstance(raw_sample, list):
                def _sanitize_nested(item):
                    if isinstance(item, list):
                        return [_sanitize_nested(x) for x in item]
                    val = float(item)
                    return float(round(val, 6)) if np.isfinite(val) else 0.0
                preview_sample = [_sanitize_nested(x) for x in raw_sample[:min(len(raw_sample), 50)]]
            else:
                val = float(raw_sample)
                preview_sample = [float(round(val, 6)) if np.isfinite(val) else 0.0]

        return sanitize_for_json({
            "feature_id": feature_id,
            "layer_name": layer_name,
            "sample_count": num_samples,
            "shape": list(features.shape),
            "feature_dim": flat_dim,
            "preview_sample": preview_sample,
            "available_extra_columns": list(extra_cols_dict.keys())
        })

    @staticmethod
    def load_feature_matrix(feature_id: str) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """
        Loads extracted features from disk and flattens to 2D matrix (num_samples, feature_dim)
        for clustering and PCA algorithms.
        """
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM features WHERE id = ?", (feature_id,)).fetchone()
            if not row:
                raise FeatureExtractionError(f"Features {feature_id} not found")
            record = dict(row)

        file_path = resolve_storage_path(record["file_path"])
        if not file_path or not file_path.exists():
            raise FileNotFoundError(f"Feature file missing: {record['file_path']}")

        with np.load(str(file_path), allow_pickle=False) as npz:
            raw_features = npz["features"]
            labels = npz["labels"] if "labels" in npz else None

        if raw_features.ndim > 2:
            num_samples = raw_features.shape[0]
            flat_features = raw_features.reshape(num_samples, -1)
        elif raw_features.ndim == 1:
            flat_features = raw_features.reshape(-1, 1)
        else:
            flat_features = raw_features

        # Ensure flat_features has no residual NaNs or Infs
        if np.isnan(flat_features).any() or np.isinf(flat_features).any():
            flat_features = np.nan_to_num(flat_features, nan=0.0, posinf=0.0, neginf=0.0)

        if labels is not None and isinstance(labels, np.ndarray):
            if labels.ndim == 2 and labels.shape[1] == 1:
                labels = labels.squeeze(1)
            elif labels.ndim != 1 or len(labels) != len(flat_features):
                labels = None
            if labels is not None and (np.isnan(labels).any() or np.isinf(labels).any()):
                labels = np.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)

        return flat_features.astype(np.float32), labels, record

    @staticmethod
    def save_as_dataset(feature_id: str, custom_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Converts extracted features into a registered project dataset (.npz)
        with 2D matrix (num_samples, feature_dim), ground truth labels (if available),
        and original sample IDs, properly setting row_count and col_count.
        """
        features, labels, record = FeatureExtractor.load_feature_matrix(feature_id)
        project_id = record["project_id"]
        num_samples, num_cols = features.shape

        # Resolve dataset name
        if custom_name and custom_name.strip():
            ds_name = custom_name.strip()
        else:
            layer_id = (record.get("layer_id") or "layer").replace("/", "_").replace(" ", "_")
            ds_name = f"Features_{layer_id}_{feature_id[:6]}"

        new_dataset_id = str(uuid.uuid4())
        dest_dir = DATASETS_DIR / project_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"{new_dataset_id}.npz"

        # Load sample_ids and extra_columns_json from feature .npz
        sample_ids = None
        extra_columns_json = None
        fp = resolve_storage_path(record.get("file_path"))
        if fp and fp.exists():
            try:
                with np.load(str(fp), allow_pickle=False) as fnpz:
                    if "sample_ids" in fnpz:
                        raw_ids = fnpz["sample_ids"]
                        sample_ids = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in raw_ids]
                    if "extra_columns_json" in fnpz:
                        extra_columns_json = str(fnpz["extra_columns_json"].item())
            except Exception:
                pass

        # If sample_ids missing or default zero-indexed fallback, resolve via DataManager.load_sample_ids
        if (sample_ids is None or len(sample_ids) != num_samples or DataManager._is_default_zero_indexed(sample_ids)) and record.get("dataset_id"):
            try:
                resolved_sids = DataManager.load_sample_ids(record["dataset_id"], expected_count=num_samples)
                if resolved_sids and len(resolved_sids) == num_samples:
                    sample_ids = resolved_sids
            except Exception:
                pass

        if sample_ids is None or len(sample_ids) != num_samples:
            sample_ids = [str(i) for i in range(num_samples)]

        # If extra_columns_json missing, attempt to resolve via DataManager.load_extra_columns
        extra_cols_list: List[str] = []
        if not extra_columns_json and record.get("dataset_id"):
            try:
                extra_dict = DataManager.load_extra_columns(record["dataset_id"], expected_count=num_samples)
                if extra_dict:
                    extra_columns_json = json.dumps(extra_dict)
                    extra_cols_list = list(extra_dict.keys())
            except Exception:
                pass
        elif extra_columns_json:
            try:
                parsed = json.loads(extra_columns_json)
                if isinstance(parsed, dict):
                    extra_cols_list = list(parsed.keys())
            except Exception:
                pass

        save_dict = {
            "data": features.astype(np.float32),
            "sample_ids": np.array(sample_ids, dtype=str)
        }
        if extra_columns_json:
            save_dict["extra_columns_json"] = np.array(extra_columns_json)

        has_labels = False
        if labels is not None and len(labels) == num_samples:
            save_dict["labels"] = labels
            save_dict["y"] = labels
            has_labels = True

        np.savez_compressed(str(dest_path), **save_dict)

        now = datetime.now(timezone.utc).isoformat()
        metadata = {
            "source": "feature_extraction",
            "feature_id": feature_id,
            "parent_model_id": record.get("model_id"),
            "parent_dataset_id": record.get("dataset_id"),
            "original_dataset_id": record.get("dataset_id"),
            "layer_id": record.get("layer_id"),
            "row_count": num_samples,
            "col_count": num_cols,
            "num_rows": num_samples,
            "num_cols": num_cols,
            "shape": [num_samples, num_cols],
            "has_labels": has_labels,
            "extra_columns": extra_cols_list
        }

        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO datasets (
                    id, project_id, name, file_path, file_format,
                    row_count, col_count, has_header, label_col,
                    created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_dataset_id, project_id, ds_name, str(dest_path), "npz",
                    num_samples, num_cols, 1 if has_labels else 0,
                    "labels" if has_labels else None,
                    now, json.dumps(metadata)
                )
            )
            conn.commit()

        return {
            "dataset_id": new_dataset_id,
            "name": ds_name,
            "row_count": num_samples,
            "col_count": num_cols,
            "file_format": "npz",
            "has_labels": has_labels,
            "metadata": metadata
        }

