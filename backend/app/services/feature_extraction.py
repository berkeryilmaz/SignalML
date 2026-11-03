import os
import json
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import keras
from app.core.config import FEATURES_DIR, MODELS_DIR, resolve_storage_path
from app.core.database import get_db_connection
from app.core.responses import sanitize_for_json
from app.services.data_manager import DataManager
from datetime import datetime, timezone


class FeatureExtractionError(Exception):
    """Raised when layer feature extraction fails."""
    pass


class FeatureExtractor:
    @staticmethod
    def get_model_layers(model_id: str) -> List[Dict[str, Any]]:
        """
        Loads the model and inspects all layers, returning their names, types, and output shapes.
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

            layers_info.append({
                "index": i,
                "name": layer.name,
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
            full_model = keras.models.load_model(str(weights_path))
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

        # Load dataset
        data, labels, _ = DataManager.load_dataset_arrays(dataset_id)
        if sample_limit is not None and sample_limit > 0:
            data = data[:sample_limit]
            if labels is not None:
                labels = labels[:sample_limit]

        # Reshape data if needed for multi-dimensional inputs (e.g. (N, L, 1))
        input_shape = full_model.input_shape
        if isinstance(input_shape, tuple):
            if len(input_shape) == 3 and data.ndim == 2:
                # Add channel dimension: (N, L) -> (N, L, 1)
                data = np.expand_dims(data, axis=-1)

        # Run batch prediction
        num_samples = len(data)
        features_list = []
        for i in range(0, num_samples, batch_size):
            batch = data[i : i + batch_size]
            batch_output = extractor_model.predict_on_batch(batch)
            if hasattr(batch_output, "numpy"):
                batch_output = batch_output.numpy()
            features_list.append(batch_output)

        features = np.concatenate(features_list, axis=0)

        # Sanitize any NaNs or Infs from intermediate layer activations
        if np.isnan(features).any() or np.isinf(features).any():
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Save extracted features
        feature_id = str(uuid.uuid4())
        dest_dir = FEATURES_DIR / project_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"{feature_id}.npz"

        save_dict = {"features": features}
        if labels is not None:
            if isinstance(labels, np.ndarray) and (np.isnan(labels).any() or np.isinf(labels).any()):
                labels = np.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
            save_dict["labels"] = labels
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
            "preview_sample": preview_sample
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
            if np.isnan(labels).any() or np.isinf(labels).any():
                labels = np.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)

        return flat_features.astype(np.float32), labels, record
