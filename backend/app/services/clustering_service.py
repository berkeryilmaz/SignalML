import os
import json
import uuid
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, OPTICS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score,
    davies_bouldin_score, calinski_harabasz_score
)
import keras
from app.services.keras_model_builder import KerasModelBuilder, ClusteringLayer

from app.core.responses import sanitize_for_json
from app.core.config import CLUSTERS_DIR, DATASETS_DIR, resolve_storage_path
from app.core.database import get_db_connection
from app.services.data_manager import DataManager
from app.services.feature_extraction import FeatureExtractor
from datetime import datetime, timezone


def safe_explained_variance(pca_model: Optional[PCA], target_dim: int) -> List[float]:
    """
    Safely extracts explained variance ratio from a fitted PCA model.
    Guarantees no NaNs or Infs, pads with 0.0 up to target_dim if necessary.
    """
    if pca_model is None:
        return [0.0] * target_dim
    evr = getattr(pca_model, "explained_variance_ratio_", None)
    if evr is None or len(evr) == 0:
        return [0.0] * target_dim

    out: List[float] = []
    for v in evr:
        val = float(v)
        if np.isfinite(val) and val >= 0.0:
            out.append(float(round(val, 4)))
        else:
            out.append(0.0)

    while len(out) < target_dim:
        out.append(0.0)
    return out[:target_dim]


class ClusteringService:
    @staticmethod
    def load_source_matrix(
        source_type: str,
        source_id: str
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """
        Loads numerical data from either a dataset or an extracted feature set.
        Returns a 2D numpy array (num_samples, num_features) and optional labels.
        Also attempts to load sample_ids into record["sample_ids"].
        """
        sample_ids: Optional[List[str]] = None
        if source_type in ("dataset", "processed"):
            data, labels, record = DataManager.load_dataset_arrays(source_id)
            if data.ndim > 2:
                data = data.reshape(data.shape[0], -1)
            elif data.ndim == 1:
                data = data.reshape(data.shape[0], 1)
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            try:
                fpath = resolve_storage_path(record.get("file_path"))
                if not fpath or not fpath.exists():
                    cand = DATASETS_DIR / record.get("project_id", "") / f"{source_id}.npz"
                    if cand.exists():
                        fpath = cand
                if fpath and fpath.exists():
                    with np.load(str(fpath), allow_pickle=False) as npz:
                        if "sample_ids" in npz:
                            sample_ids = [str(s) for s in npz["sample_ids"]]
            except Exception:
                pass

            record["sample_ids"] = sample_ids
            return data, labels, record
        elif source_type in ("features", "feature"):
            data, labels, record = FeatureExtractor.load_feature_matrix(source_id)
            if data.ndim > 2:
                data = data.reshape(data.shape[0], -1)
            elif data.ndim == 1:
                data = data.reshape(data.shape[0], 1)
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            try:
                fpath = resolve_storage_path(record.get("file_path"))
                if fpath and fpath.exists():
                    with np.load(str(fpath), allow_pickle=False) as npz:
                        if "sample_ids" in npz:
                            sample_ids = [str(s) for s in npz["sample_ids"]]
                if sample_ids is None and record.get("dataset_id"):
                    _, _, ds_record = DataManager.load_dataset_arrays(record["dataset_id"])
                    ds_path = resolve_storage_path(ds_record.get("file_path"))
                    if ds_path and ds_path.exists():
                        with np.load(str(ds_path), allow_pickle=False) as npz:
                            if "sample_ids" in npz:
                                sample_ids = [str(s) for s in npz["sample_ids"]]
            except Exception:
                pass

            record["sample_ids"] = sample_ids
            return data, labels, record
        else:
            raise ValueError(f"Unknown source type: {source_type}")

    @staticmethod
    def run_clustering(
        project_id: str,
        source_type: str,
        source_id: str,
        algorithm: str = "kmeans",
        # KMeans parameters
        n_clusters: int = 3,
        init: str = "k-means++",
        n_init: int = 10,
        # DBSCAN parameters
        eps: float = 0.5,
        min_samples: Optional[int] = 5,
        metric: str = "euclidean",
        # HDBSCAN parameters
        min_cluster_size: int = 5,
        cluster_selection_epsilon: float = 0.0,
        cluster_selection_method: str = "eom",
        # OPTICS parameters
        max_eps: Optional[float] = None,
        xi: float = 0.05,
        # General parameters
        standardize: bool = True,
        random_state: int = 42,
        compute_pca: bool = True,
        # IDEC parameters
        model_id: Optional[str] = None,
        epochs: int = 15,
        pretrain_epochs: int = 5,
        gamma: float = 0.1,
        update_interval: int = 1,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Executes clustering and PCA dimension reduction on the input data matrix.
        Returns coordinates, cluster labels, soft probabilities, metrics, and saves persistently.
        """
        if params is not None:
            if "model_id" in params and not model_id:
                model_id = params["model_id"]
            if "n_clusters" in params:
                try:
                    n_clusters = int(params["n_clusters"])
                except Exception:
                    pass
            if "epochs" in params:
                try:
                    epochs = int(params["epochs"])
                except Exception:
                    pass
            if "pretrain_epochs" in params:
                try:
                    pretrain_epochs = int(params["pretrain_epochs"])
                except Exception:
                    pass
            if "gamma" in params:
                try:
                    gamma = float(params["gamma"])
                except Exception:
                    pass
            if "update_interval" in params:
                try:
                    update_interval = int(params["update_interval"])
                except Exception:
                    pass

        X, ground_truth, source_meta = ClusteringService.load_source_matrix(source_type, source_id)
        num_samples, feature_dim = X.shape

        if num_samples < 2:
            raise ValueError("Clustering requires at least 2 samples.")

        # Input ID Traceability: load sample_ids if present or generate fallback indices
        sample_ids_raw = source_meta.get("sample_ids")
        if sample_ids_raw and len(sample_ids_raw) == num_samples:
            sample_ids = [str(s) for s in sample_ids_raw]
        else:
            sample_ids = [str(i) for i in range(num_samples)]

        # Handle NaNs or Infs if any in raw inputs
        if np.isnan(X).any() or np.isinf(X).any():
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Standardize if requested
        if standardize:
            scaler = StandardScaler()
            X_proc = scaler.fit_transform(X)
            if np.isnan(X_proc).any() or np.isinf(X_proc).any():
                X_proc = np.nan_to_num(X_proc, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            X_proc = np.asarray(X, dtype=np.float64)

        Z: Optional[np.ndarray] = None
        probabilities: Optional[np.ndarray] = None
        confidence: Optional[np.ndarray] = None
        uncertainty: Optional[np.ndarray] = None
        cluster_labels: Optional[np.ndarray] = None

        # Run clustering algorithm
        algo_lower = algorithm.lower()
        if algo_lower == "kmeans":
            actual_clusters = min(n_clusters, num_samples)
            clusterer = KMeans(
                n_clusters=actual_clusters,
                init=init,
                n_init=n_init,
                random_state=random_state
            )
            # Catch duplicate-point ConvergenceWarning gracefully
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", message=".*Number of distinct clusters.*")
                cluster_labels = clusterer.fit_predict(X_proc)

            params = {
                "n_clusters": actual_clusters,
                "init": init,
                "n_init": n_init,
                "random_state": random_state
            }
        elif algo_lower == "dbscan":
            actual_min_samples = max(1, min(min_samples if min_samples is not None else 5, num_samples))
            clusterer = DBSCAN(eps=eps, min_samples=actual_min_samples, metric=metric)
            cluster_labels = clusterer.fit_predict(X_proc)
            params = {
                "eps": eps,
                "min_samples": actual_min_samples,
                "metric": metric
            }
        elif algo_lower == "hdbscan":
            actual_min_cluster_size = max(2, min(min_cluster_size, num_samples))
            actual_min_samples = max(1, min(min_samples, num_samples)) if min_samples is not None else None
            clusterer = HDBSCAN(
                min_cluster_size=actual_min_cluster_size,
                min_samples=actual_min_samples,
                cluster_selection_epsilon=cluster_selection_epsilon,
                cluster_selection_method=cluster_selection_method,
                metric=metric,
                copy=True
            )
            cluster_labels = clusterer.fit_predict(X_proc)
            params = {
                "min_cluster_size": actual_min_cluster_size,
                "min_samples": actual_min_samples,
                "cluster_selection_epsilon": cluster_selection_epsilon,
                "cluster_selection_method": cluster_selection_method,
                "metric": metric
            }
        elif algo_lower == "optics":
            actual_min_samples = max(2, min(min_samples if min_samples is not None else 5, num_samples))
            optics_max_eps = max_eps if (max_eps is not None and max_eps > 0) else np.inf
            clusterer = OPTICS(
                min_samples=actual_min_samples,
                max_eps=optics_max_eps,
                xi=xi,
                metric=metric
            )
            cluster_labels = clusterer.fit_predict(X_proc)
            params = {
                "min_samples": actual_min_samples,
                "max_eps": max_eps,
                "xi": xi,
                "metric": metric
            }
        elif algo_lower == "idec":
            # 1. Retrieve model_id if not explicitly provided
            if not model_id:
                with get_db_connection() as conn:
                    p_row = conn.execute("SELECT active_model_id FROM projects WHERE id = ?", (project_id,)).fetchone()
                    if p_row and p_row["active_model_id"]:
                        model_id = p_row["active_model_id"]
                    if not model_id:
                        m_row = conn.execute("SELECT id FROM models WHERE project_id = ? ORDER BY created_at DESC LIMIT 1", (project_id,)).fetchone()
                        if m_row:
                            model_id = m_row["id"]
                        else:
                            m_row = conn.execute("SELECT id FROM models ORDER BY created_at DESC LIMIT 1").fetchone()
                            if m_row:
                                model_id = m_row["id"]

            if not model_id:
                raise ValueError("No model found for IDEC clustering. Please select or train a model first.")

            with get_db_connection() as conn:
                model_row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
                if not model_row:
                    raise ValueError(f"Model '{model_id}' not found in database.")
                model_record = dict(model_row)

            # 2. Load model from weights or rebuild from graph_json
            weights_path = resolve_storage_path(model_record.get("weights_path"))
            model = None
            if weights_path and weights_path.exists():
                try:
                    model = keras.models.load_model(str(weights_path), custom_objects={"ClusteringLayer": ClusteringLayer})
                except Exception:
                    model = None

            if model is None:
                graph_json_str = model_record.get("graph_json")
                if graph_json_str:
                    graph_json = json.loads(graph_json_str)
                    model, _ = KerasModelBuilder.build_model(graph_json)
                else:
                    raise ValueError(f"Could not load model '{model_id}': weights not found and graph_json missing.")

            # 3. Prepare input array matching model.input_shape
            model_in_shape = model.input_shape
            if isinstance(model_in_shape, list):
                model_in_shape = model_in_shape[0]

            X_input = np.asarray(X, dtype=np.float32)

            if len(model_in_shape) == 3:
                # Shape is (batch_size, seq_len, channels)
                target_seq = model_in_shape[1]
                if X_input.ndim == 2:
                    if target_seq is not None:
                        if X_input.shape[1] < target_seq:
                            X_input = np.pad(X_input, ((0, 0), (0, target_seq - X_input.shape[1])), mode="constant")
                        elif X_input.shape[1] > target_seq:
                            X_input = X_input[:, :target_seq]
                    X_input = np.expand_dims(X_input, axis=-1)
                elif X_input.ndim == 3:
                    if target_seq is not None:
                        if X_input.shape[1] < target_seq:
                            X_input = np.pad(X_input, ((0, 0), (0, target_seq - X_input.shape[1]), (0, 0)), mode="constant")
                        elif X_input.shape[1] > target_seq:
                            X_input = X_input[:, :target_seq, :]
            elif len(model_in_shape) == 2:
                target_dim = model_in_shape[1]
                if X_input.ndim > 2:
                    X_input = X_input.reshape(X_input.shape[0], -1)
                if target_dim is not None:
                    if X_input.shape[1] < target_dim:
                        X_input = np.pad(X_input, ((0, 0), (0, target_dim - X_input.shape[1])), mode="constant")
                    elif X_input.shape[1] > target_dim:
                        X_input = X_input[:, :target_dim]

            # 4. Locate or attach ClusteringLayer
            clustering_layer = None
            for layer in model.layers:
                if isinstance(layer, ClusteringLayer) or layer.__class__.__name__ == "ClusteringLayer" or "clustering" in getattr(layer, "name", "").lower():
                    clustering_layer = layer
                    break

            outputs = model.outputs if isinstance(model.outputs, list) else [model.output]

            if clustering_layer is not None and getattr(clustering_layer, "n_clusters", None) == n_clusters:
                bottleneck_tensor = clustering_layer.input
                if isinstance(bottleneck_tensor, list):
                    bottleneck_tensor = bottleneck_tensor[0]

                clustering_out_idx = 0
                for idx, out_t in enumerate(outputs):
                    layer = None
                    if hasattr(out_t, "_keras_history"):
                        hist = out_t._keras_history
                        layer = hist[0] if isinstance(hist, tuple) else getattr(hist, "operation", None)
                    if layer is not None and (isinstance(layer, ClusteringLayer) or layer.__class__.__name__ == "ClusteringLayer" or "clustering" in getattr(layer, "name", "").lower()):
                        clustering_out_idx = idx
                        break

                if len(outputs) == 2:
                    idec_model = model
                    clustering_out = outputs[clustering_out_idx]
                    recon_out = outputs[1 - clustering_out_idx]
                else:
                    clustering_out = outputs[0]
                    recon_out = outputs[0]
                    idec_model = model
            else:
                # 1-output Autoencoder without ClusteringLayer (or n_clusters mismatch):
                # Find bottleneck layer: intermediate Dense layer with fewest units or middle layer
                candidate_layers = []
                for layer in model.layers[1:-1]:
                    l_cls = layer.__class__.__name__.lower()
                    if "dense" in l_cls:
                        units = getattr(layer, "units", None)
                        if units is not None:
                            candidate_layers.append((units, layer))
                        elif hasattr(layer, "output_shape") and layer.output_shape:
                            shape = layer.output_shape
                            if isinstance(shape, tuple) and len(shape) >= 2 and shape[-1] is not None:
                                candidate_layers.append((shape[-1], layer))

                if candidate_layers:
                    candidate_layers.sort(key=lambda item: item[0])
                    bottleneck_layer = candidate_layers[0][1]
                else:
                    mid_idx = len(model.layers) // 2
                    bottleneck_layer = model.layers[mid_idx]

                bottleneck_tensor = bottleneck_layer.output
                if isinstance(bottleneck_tensor, list):
                    bottleneck_tensor = bottleneck_tensor[0]
                if len(bottleneck_tensor.shape) > 2:
                    bottleneck_tensor = keras.layers.Flatten(name="latent_flatten")(bottleneck_tensor)

                clustering_layer = ClusteringLayer(n_clusters=n_clusters, name="clustering_idec")
                clustering_out = clustering_layer(bottleneck_tensor)
                recon_out = outputs[0]
                idec_model = keras.Model(inputs=model.inputs, outputs=[clustering_out, recon_out])
                clustering_out_idx = 0

            # 5. Build encoder for latent representations
            encoder = keras.Model(inputs=model.inputs, outputs=bottleneck_tensor)

            # 6. Pre-train reconstruction branch if requested
            target_recon = np.copy(X_input)
            recon_shape = recon_out.shape
            if len(recon_shape) == 2 and target_recon.ndim == 3 and target_recon.shape[-1] == 1:
                target_recon = target_recon.squeeze(-1)
            elif len(recon_shape) == 3 and target_recon.ndim == 2:
                target_recon = np.expand_dims(target_recon, axis=-1)

            if len(recon_shape) >= 2 and recon_shape[1] is not None and target_recon.shape[1] != recon_shape[1]:
                diff = recon_shape[1] - target_recon.shape[1]
                if diff > 0:
                    pad_cfg = ((0, 0), (0, diff)) if target_recon.ndim == 2 else ((0, 0), (0, diff), (0, 0))
                    target_recon = np.pad(target_recon, pad_cfg, mode="constant")
                else:
                    target_recon = target_recon[:, :recon_shape[1]]

            if pretrain_epochs > 0:
                try:
                    ae_model = keras.Model(inputs=model.inputs, outputs=recon_out)
                    ae_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
                    ae_model.fit(X_input, target_recon, batch_size=32, epochs=pretrain_epochs, verbose=0, shuffle=True)
                except Exception as e:
                    warnings.warn(f"Autoencoder pre-training skipped: {e}")

            # 7. Extract bottleneck features Z and initialize ClusteringLayer with KMeans
            Z = encoder.predict(X_input, verbose=0)
            if Z.ndim > 2:
                Z = Z.reshape(Z.shape[0], -1)

            actual_k = min(n_clusters, len(Z))
            kmeans_init = KMeans(n_clusters=actual_k, n_init=20, random_state=random_state)
            kmeans_init.fit(Z)
            if actual_k == n_clusters:
                clustering_layer.set_weights([kmeans_init.cluster_centers_])

            # 8. Iterative IDEC training loop if epochs > 0
            def compute_target_distribution(q: np.ndarray) -> np.ndarray:
                weight = q ** 2 / np.maximum(q.sum(axis=0), 1e-10)
                return (weight.T / np.maximum(weight.sum(axis=1), 1e-10)).T

            if epochs > 0:
                losses = ["kld", "mse"] if clustering_out_idx == 0 else ["mse", "kld"]
                loss_weights = [gamma, 1.0] if clustering_out_idx == 0 else [1.0, gamma]
                idec_model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss=losses,
                    loss_weights=loss_weights
                )

                current_P = None
                for ep in range(epochs):
                    if ep % max(1, update_interval) == 0 or current_P is None:
                        curr_preds = idec_model.predict(X_input, verbose=0)
                        curr_Q = curr_preds[clustering_out_idx] if isinstance(curr_preds, list) else curr_preds
                        current_P = compute_target_distribution(curr_Q)

                    y_targets = [current_P, target_recon] if clustering_out_idx == 0 else [target_recon, current_P]
                    idec_model.fit(X_input, y_targets, batch_size=32, epochs=1, verbose=0, shuffle=True)

            # 9. Compute final soft probabilities Q and cluster assignments
            final_preds = idec_model.predict(X_input, verbose=0)
            Q = final_preds[clustering_out_idx] if isinstance(final_preds, list) else final_preds
            probabilities = np.asarray(Q, dtype=np.float64)
            cluster_labels = np.argmax(probabilities, axis=1).astype(int)
            confidence = np.max(probabilities, axis=1)
            prob_safe = np.clip(probabilities, 1e-12, 1.0)
            entropy = -np.sum(probabilities * np.log(prob_safe), axis=1)
            uncertainty = np.clip(entropy / np.log(max(2, n_clusters)), 0.0, 1.0)

            # Updated bottleneck latent features Z
            Z = encoder.predict(X_input, verbose=0)
            if Z.ndim > 2:
                Z = Z.reshape(Z.shape[0], -1)
            Z = np.asarray(Z, dtype=np.float64)

            params = {
                "model_id": model_id,
                "n_clusters": n_clusters,
                "epochs": epochs,
                "pretrain_epochs": pretrain_epochs,
                "gamma": gamma,
                "update_interval": update_interval
            }
        else:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

        # Compute cluster distribution
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        distribution = {int(k): int(v) for k, v in zip(unique_labels, counts)}

        # Soft Cluster Probabilities for non-IDEC algorithms (Student's t-distribution kernel)
        if algo_lower != "idec":
            unique_valid = sorted([int(c) for c in unique_labels if c >= 0])
            num_valid_clusters = len(unique_valid)

            if num_valid_clusters > 0:
                centroids = []
                for c in unique_valid:
                    mask = (cluster_labels == c)
                    if np.any(mask):
                        centroids.append(np.mean(X_proc[mask], axis=0))
                    else:
                        centroids.append(np.zeros(feature_dim, dtype=np.float64))
                centroids_arr = np.vstack(centroids)  # (num_valid_clusters, feature_dim)

                diff = X_proc[:, np.newaxis, :] - centroids_arr[np.newaxis, :, :]  # (N, K, D)
                dist_sq = np.sum(diff ** 2, axis=2)  # (N, K)
                num = 1.0 / (1.0 + dist_sq)
                denom = np.sum(num, axis=1, keepdims=True)
                probabilities = num / np.maximum(denom, 1e-12)

                cluster_to_idx = {c: idx for idx, c in enumerate(unique_valid)}
                confidence = np.zeros(num_samples, dtype=np.float64)
                for i in range(num_samples):
                    c = int(cluster_labels[i])
                    if c in cluster_to_idx:
                        confidence[i] = float(probabilities[i, cluster_to_idx[c]])
                    else:
                        confidence[i] = float(np.max(probabilities[i]))

                if num_valid_clusters > 1:
                    prob_safe = np.clip(probabilities, 1e-12, 1.0)
                    entropy = -np.sum(probabilities * np.log(prob_safe), axis=1)
                    uncertainty = np.clip(entropy / np.log(num_valid_clusters), 0.0, 1.0)
                else:
                    uncertainty = np.zeros(num_samples, dtype=np.float64)
            else:
                probabilities = np.zeros((num_samples, 0), dtype=np.float64)
                confidence = np.zeros(num_samples, dtype=np.float64)
                uncertainty = np.zeros(num_samples, dtype=np.float64)
        else:
            unique_valid = list(range(n_clusters))
            num_valid_clusters = len(unique_valid)

        # Metrics calculation
        metrics: Dict[str, Optional[float]] = {
            "silhouette_score": None,
            "adjusted_rand_index": None,
            "normalized_mutual_info": None
        }
        if algo_lower == "idec":
            metrics["davies_bouldin_index"] = None
            metrics["calinski_harabasz_score"] = None
            metrics["mean_cluster_confidence"] = float(round(float(np.mean(confidence)), 4)) if (confidence is not None and len(confidence) > 0) else None

        # Silhouette score and cluster quality metrics
        eval_features = Z if algo_lower == "idec" and Z is not None else X_proc
        non_noise_mask = cluster_labels != -1
        valid_labels = cluster_labels[non_noise_mask]
        num_unique_valid = len(np.unique(valid_labels))

        if num_unique_valid >= 2 and np.sum(non_noise_mask) > num_unique_valid:
            try:
                sil = silhouette_score(eval_features[non_noise_mask], valid_labels)
                if sil is not None and np.isfinite(sil):
                    metrics["silhouette_score"] = float(round(float(sil), 4))
                else:
                    metrics["silhouette_score"] = None
            except Exception:
                metrics["silhouette_score"] = None

            if algo_lower == "idec":
                try:
                    db = davies_bouldin_score(eval_features[non_noise_mask], valid_labels)
                    if db is not None and np.isfinite(db):
                        metrics["davies_bouldin_index"] = float(round(float(db), 4))
                except Exception:
                    pass

                try:
                    ch = calinski_harabasz_score(eval_features[non_noise_mask], valid_labels)
                    if ch is not None and np.isfinite(ch):
                        metrics["calinski_harabasz_score"] = float(round(float(ch), 4))
                except Exception:
                    pass
        else:
            metrics["silhouette_score"] = None

        if ground_truth is not None and len(ground_truth) == num_samples:
            gt_unique = len(np.unique(ground_truth))
            cl_unique = len(np.unique(cluster_labels))
            if gt_unique >= 2 and cl_unique >= 2:
                try:
                    ari = adjusted_rand_score(ground_truth, cluster_labels)
                    if ari is not None and np.isfinite(ari):
                        metrics["adjusted_rand_index"] = float(round(float(ari), 4))
                    else:
                        metrics["adjusted_rand_index"] = None
                except Exception:
                    metrics["adjusted_rand_index"] = None

                try:
                    nmi = normalized_mutual_info_score(ground_truth, cluster_labels)
                    if nmi is not None and np.isfinite(nmi):
                        metrics["normalized_mutual_info"] = float(round(float(nmi), 4))
                    else:
                        metrics["normalized_mutual_info"] = None
                except Exception:
                    metrics["normalized_mutual_info"] = None

        # Compute PCA (both 2D and 3D for interactive visualization)
        # For IDEC: use learned bottleneck latent features Z!
        pca_input_matrix = Z if algo_lower == "idec" and Z is not None else X_proc
        pca_feature_dim = pca_input_matrix.shape[1]

        pca_2d_coords: Optional[np.ndarray] = None
        pca_3d_coords: Optional[np.ndarray] = None
        explained_variance_2d: List[float] = []
        explained_variance_3d: List[float] = []
        points: List[Dict[str, Any]] = []

        if compute_pca:
            total_variance = float(np.var(pca_input_matrix, axis=0).sum())
            has_variance = np.isfinite(total_variance) and total_variance > 1e-12

            if not has_variance:
                # Constant or zero-variance data: all samples are identical.
                # Points project to origin, explained variance is strictly 0.0.
                pca_2d_coords = np.zeros((num_samples, 2), dtype=np.float32)
                pca_3d_coords = np.zeros((num_samples, 3), dtype=np.float32)
                explained_variance_2d = [0.0, 0.0]
                explained_variance_3d = [0.0, 0.0, 0.0]
            elif pca_feature_dim == 1:
                # 1D feature space edge case
                try:
                    pca1 = PCA(n_components=1, random_state=random_state)
                    c1 = pca1.fit_transform(pca_input_matrix)
                    c1 = np.nan_to_num(c1, nan=0.0, posinf=0.0, neginf=0.0)
                    pca_2d_coords = np.column_stack([c1, np.zeros((num_samples, 1), dtype=np.float32)])
                    pca_3d_coords = np.column_stack([c1, np.zeros((num_samples, 2), dtype=np.float32)])
                    explained_variance_2d = [1.0, 0.0]
                    explained_variance_3d = [1.0, 0.0, 0.0]
                except Exception:
                    pca_2d_coords = np.zeros((num_samples, 2), dtype=np.float32)
                    pca_3d_coords = np.zeros((num_samples, 3), dtype=np.float32)
                    explained_variance_2d = [0.0, 0.0]
                    explained_variance_3d = [0.0, 0.0, 0.0]
            elif pca_feature_dim == 2:
                try:
                    pca2 = PCA(n_components=2, random_state=random_state)
                    pca_2d_coords = pca2.fit_transform(pca_input_matrix)
                    pca_2d_coords = np.nan_to_num(pca_2d_coords, nan=0.0, posinf=0.0, neginf=0.0)
                    explained_variance_2d = safe_explained_variance(pca2, 2)

                    # Pad with zeros for 3rd component if only 2 features exist
                    pca_3d_coords = np.column_stack([pca_2d_coords, np.zeros((num_samples, 1), dtype=np.float32)])
                    explained_variance_3d = explained_variance_2d + [0.0]
                except Exception:
                    pca_2d_coords = np.zeros((num_samples, 2), dtype=np.float32)
                    pca_3d_coords = np.zeros((num_samples, 3), dtype=np.float32)
                    explained_variance_2d = [0.0, 0.0]
                    explained_variance_3d = [0.0, 0.0, 0.0]
            else:
                # pca_feature_dim >= 3
                try:
                    pca2 = PCA(n_components=2, random_state=random_state)
                    pca_2d_coords = pca2.fit_transform(pca_input_matrix)
                    pca_2d_coords = np.nan_to_num(pca_2d_coords, nan=0.0, posinf=0.0, neginf=0.0)
                    explained_variance_2d = safe_explained_variance(pca2, 2)
                except Exception:
                    pca_2d_coords = np.zeros((num_samples, 2), dtype=np.float32)
                    explained_variance_2d = [0.0, 0.0]

                if min(num_samples, pca_feature_dim) < 3:
                    # When samples < 3, 3-component PCA is mathematically impossible.
                    # Pad 2D coordinates with zeros for 3rd dimension to preserve 2D visualization.
                    pca_3d_coords = np.column_stack([pca_2d_coords, np.zeros((num_samples, 1), dtype=np.float32)])
                    explained_variance_3d = explained_variance_2d + [0.0]
                else:
                    try:
                        pca3 = PCA(n_components=3, random_state=random_state)
                        pca_3d_coords = pca3.fit_transform(pca_input_matrix)
                        pca_3d_coords = np.nan_to_num(pca_3d_coords, nan=0.0, posinf=0.0, neginf=0.0)
                        explained_variance_3d = safe_explained_variance(pca3, 3)
                    except Exception:
                        pca_3d_coords = np.column_stack([pca_2d_coords, np.zeros((num_samples, 1), dtype=np.float32)])
                        explained_variance_3d = explained_variance_2d + [0.0]

            # Build points array for frontend visualization
            if pca_3d_coords is not None:
                for i in range(num_samples):
                    x_val = float(pca_3d_coords[i, 0])
                    y_val = float(pca_3d_coords[i, 1])
                    z_val = float(pca_3d_coords[i, 2])
                    if algo_lower == "idec":
                        prob_dict = {
                            str(c): float(round(float(probabilities[i, c]), 4))
                            for c in range(probabilities.shape[1])
                        }
                    else:
                        prob_dict = {
                            str(c): float(round(float(probabilities[i, idx]), 4))
                            for idx, c in enumerate(unique_valid)
                            if idx < probabilities.shape[1]
                        }
                    point = {
                        "sample_id": i,
                        "input_id": str(sample_ids[i]),
                        "cluster": int(cluster_labels[i]),
                        "confidence": float(round(float(confidence[i]), 4)) if confidence is not None else 1.0,
                        "uncertainty": float(round(float(uncertainty[i]), 4)) if uncertainty is not None else 0.0,
                        "probabilities": prob_dict,
                        "x": float(round(x_val, 4)) if np.isfinite(x_val) else 0.0,
                        "y": float(round(y_val, 4)) if np.isfinite(y_val) else 0.0,
                        "z": float(round(z_val, 4)) if np.isfinite(z_val) else 0.0,
                    }
                    if ground_truth is not None:
                        gt_val = ground_truth[i]
                        if np.issubdtype(type(gt_val), np.integer):
                            point["ground_truth"] = int(gt_val)
                        elif np.issubdtype(type(gt_val), np.floating):
                            point["ground_truth"] = float(round(float(gt_val), 4)) if np.isfinite(gt_val) else 0.0
                        else:
                            point["ground_truth"] = str(gt_val)
                    points.append(point)
        else:
            for i in range(num_samples):
                if algo_lower == "idec":
                    prob_dict = {
                        str(c): float(round(float(probabilities[i, c]), 4))
                        for c in range(probabilities.shape[1])
                    }
                else:
                    prob_dict = {
                        str(c): float(round(float(probabilities[i, idx]), 4))
                        for idx, c in enumerate(unique_valid)
                        if idx < probabilities.shape[1]
                    }
                point = {
                    "sample_id": i,
                    "input_id": str(sample_ids[i]),
                    "cluster": int(cluster_labels[i]),
                    "confidence": float(round(float(confidence[i]), 4)) if confidence is not None else 1.0,
                    "uncertainty": float(round(float(uncertainty[i]), 4)) if uncertainty is not None else 0.0,
                    "probabilities": prob_dict,
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                }
                if ground_truth is not None:
                    gt_val = ground_truth[i]
                    if np.issubdtype(type(gt_val), np.integer):
                        point["ground_truth"] = int(gt_val)
                    elif np.issubdtype(type(gt_val), np.floating):
                        point["ground_truth"] = float(round(float(gt_val), 4)) if np.isfinite(gt_val) else 0.0
                    else:
                        point["ground_truth"] = str(gt_val)
                points.append(point)

        # Save clustering results to persistent storage
        clustering_id = str(uuid.uuid4())
        dest_dir = CLUSTERS_DIR / project_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        result_file = dest_dir / f"{clustering_id}.npz"

        save_dict = {
            "cluster_labels": cluster_labels,
            "distribution": np.array(list(distribution.items())),
            "probabilities": probabilities,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "cluster_ids": np.array(unique_valid, dtype=int),
            "sample_ids": np.array(sample_ids, dtype=str)
        }
        if compute_pca and pca_2d_coords is not None and pca_3d_coords is not None:
            save_dict["pca_2d"] = pca_2d_coords
            save_dict["pca_3d"] = pca_3d_coords
        if ground_truth is not None:
            save_dict["ground_truth"] = ground_truth

        np.savez_compressed(str(result_file), **save_dict)

        now = datetime.now(timezone.utc).isoformat()
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO clusterings (
                    id, project_id, source_type, source_id, algorithm,
                    params_json, result_path, pca_path, metrics_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    clustering_id, project_id, source_type, source_id, algo_lower,
                    json.dumps(sanitize_for_json(params)),
                    str(result_file),
                    str(result_file),
                    json.dumps(sanitize_for_json(metrics)),
                    now
                )
            )
            conn.commit()

        return sanitize_for_json({
            "clustering_id": clustering_id,
            "algorithm": algo_lower,
            "num_samples": num_samples,
            "feature_dim": feature_dim,
            "cluster_distribution": distribution,
            "metrics": metrics,
            "explained_variance_2d": explained_variance_2d,
            "explained_variance_3d": explained_variance_3d,
            "points": points
        })

    @staticmethod
    def get_clustering_detail(clustering_id: str) -> Dict[str, Any]:
        """
        Reloads a saved clustering run from DB and .npz file.
        Reconstructs points, cluster distribution, PCA coordinates, soft probabilities, and metrics.
        """
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM clusterings WHERE id = ?", (clustering_id,)).fetchone()
            if not row:
                raise ValueError(f"Clustering run {clustering_id} not found")
            c_data = dict(row)

        result_path = resolve_storage_path(c_data["result_path"])
        if not result_path or not result_path.exists():
            raise FileNotFoundError(f"Clustering result file not found: {c_data['result_path']}")

        with np.load(str(result_path), allow_pickle=False) as npz:
            cluster_labels = npz["cluster_labels"]
            distribution_arr = npz["distribution"] if "distribution" in npz else None
            pca_2d = npz["pca_2d"] if "pca_2d" in npz else None
            pca_3d = npz["pca_3d"] if "pca_3d" in npz else None
            ground_truth = npz["ground_truth"] if "ground_truth" in npz else None
            probabilities = npz["probabilities"] if "probabilities" in npz else None
            confidence = npz["confidence"] if "confidence" in npz else None
            uncertainty = npz["uncertainty"] if "uncertainty" in npz else None
            cluster_ids = [int(c) for c in npz["cluster_ids"]] if "cluster_ids" in npz else sorted([int(c) for c in np.unique(cluster_labels) if c >= 0])
            sample_ids = [str(s) for s in npz["sample_ids"]] if "sample_ids" in npz else [str(i) for i in range(len(cluster_labels))]

        num_samples = len(cluster_labels)
        if distribution_arr is not None:
            distribution = {int(k): int(v) for k, v in distribution_arr}
        else:
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            distribution = {int(k): int(v) for k, v in zip(unique_labels, counts)}

        points = []
        for i in range(num_samples):
            x_val = float(pca_3d[i, 0]) if pca_3d is not None and i < len(pca_3d) else 0.0
            y_val = float(pca_3d[i, 1]) if pca_3d is not None and i < len(pca_3d) else 0.0
            z_val = float(pca_3d[i, 2]) if pca_3d is not None and i < len(pca_3d) else 0.0

            prob_dict = {}
            if probabilities is not None and i < len(probabilities):
                for idx, c in enumerate(cluster_ids):
                    if idx < probabilities.shape[1]:
                        prob_dict[str(c)] = float(round(float(probabilities[i, idx]), 4))

            conf_val = float(round(float(confidence[i]), 4)) if (confidence is not None and i < len(confidence)) else 1.0
            unc_val = float(round(float(uncertainty[i]), 4)) if (uncertainty is not None and i < len(uncertainty)) else 0.0

            pt = {
                "sample_id": i,
                "input_id": str(sample_ids[i]),
                "cluster": int(cluster_labels[i]),
                "confidence": conf_val,
                "uncertainty": unc_val,
                "probabilities": prob_dict,
                "x": float(round(x_val, 4)) if np.isfinite(x_val) else 0.0,
                "y": float(round(y_val, 4)) if np.isfinite(y_val) else 0.0,
                "z": float(round(z_val, 4)) if np.isfinite(z_val) else 0.0,
            }
            if ground_truth is not None and i < len(ground_truth):
                gt_val = ground_truth[i]
                if np.issubdtype(type(gt_val), np.integer):
                    pt["ground_truth"] = int(gt_val)
                elif np.issubdtype(type(gt_val), np.floating):
                    pt["ground_truth"] = float(round(float(gt_val), 4)) if np.isfinite(gt_val) else 0.0
                else:
                    pt["ground_truth"] = str(gt_val)
            points.append(pt)

        params = {}
        try:
            params = json.loads(c_data.get("params_json", "{}"))
        except Exception:
            pass

        metrics = {}
        try:
            metrics = json.loads(c_data.get("metrics_json", "{}"))
        except Exception:
            pass

        return sanitize_for_json({
            "clustering_id": clustering_id,
            "project_id": c_data.get("project_id"),
            "source_type": c_data.get("source_type"),
            "source_id": c_data.get("source_id"),
            "algorithm": c_data.get("algorithm"),
            "params": params,
            "metrics": metrics,
            "num_samples": num_samples,
            "cluster_distribution": distribution,
            "points": points
        })

    @staticmethod
    def export_clustering_csv(clustering_id: str) -> str:
        """
        Exports a CSV string with format:
        input_id,cluster_label,confidence,uncertainty,prob_cluster_0,...,pca_x,pca_y,pca_z,ground_truth
        """
        import io
        import csv

        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM clusterings WHERE id = ?", (clustering_id,)).fetchone()
            if not row:
                raise ValueError(f"Clustering run {clustering_id} not found")
            c_data = dict(row)

        result_path = resolve_storage_path(c_data["result_path"])
        if not result_path or not result_path.exists():
            raise FileNotFoundError(f"Clustering result file not found: {c_data['result_path']}")

        with np.load(str(result_path), allow_pickle=False) as npz:
            cluster_labels = npz["cluster_labels"]
            probabilities = npz["probabilities"] if "probabilities" in npz else None
            confidence = npz["confidence"] if "confidence" in npz else None
            uncertainty = npz["uncertainty"] if "uncertainty" in npz else None
            cluster_ids = [int(c) for c in npz["cluster_ids"]] if "cluster_ids" in npz else sorted([int(c) for c in np.unique(cluster_labels) if c >= 0])
            sample_ids = [str(s) for s in npz["sample_ids"]] if "sample_ids" in npz else [str(i) for i in range(len(cluster_labels))]
            pca_3d = npz["pca_3d"] if "pca_3d" in npz else None
            ground_truth = npz["ground_truth"] if "ground_truth" in npz else None

        num_samples = len(cluster_labels)
        output = io.StringIO()
        writer = csv.writer(output, lineterminator="\n")

        headers = ["input_id", "cluster_label", "confidence", "uncertainty"]
        for c in cluster_ids:
            headers.append(f"prob_cluster_{c}")
        headers.extend(["pca_x", "pca_y", "pca_z", "ground_truth"])
        writer.writerow(headers)

        for i in range(num_samples):
            s_id = str(sample_ids[i])
            c_lbl = int(cluster_labels[i])
            conf_val = f"{float(confidence[i]):.4f}" if (confidence is not None and i < len(confidence)) else "1.0000"
            unc_val = f"{float(uncertainty[i]):.4f}" if (uncertainty is not None and i < len(uncertainty)) else "0.0000"

            row = [s_id, c_lbl, conf_val, unc_val]

            for idx, c in enumerate(cluster_ids):
                if probabilities is not None and i < len(probabilities) and idx < probabilities.shape[1]:
                    row.append(f"{float(probabilities[i, idx]):.4f}")
                else:
                    row.append("0.0000")

            if pca_3d is not None and i < len(pca_3d):
                row.append(f"{float(pca_3d[i, 0]):.4f}")
                row.append(f"{float(pca_3d[i, 1]):.4f}")
                row.append(f"{float(pca_3d[i, 2]):.4f}")
            else:
                row.extend(["0.0000", "0.0000", "0.0000"])

            if ground_truth is not None and i < len(ground_truth):
                row.append(str(ground_truth[i]))
            else:
                row.append("")

            writer.writerow(row)

        return output.getvalue()

    @staticmethod
    def delete_clustering(clustering_id: str) -> bool:
        """
        Deletes the clustering record from DB and its .npz file from disk.
        """
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM clusterings WHERE id = ?", (clustering_id,)).fetchone()
            if not row:
                raise ValueError(f"Clustering run {clustering_id} not found")
            result_path_str = row["result_path"]
            conn.execute("DELETE FROM clusterings WHERE id = ?", (clustering_id,))
            conn.commit()

        if result_path_str:
            file_path = resolve_storage_path(result_path_str)
            if file_path and file_path.exists():
                try:
                    file_path.unlink()
                except Exception:
                    pass
        return True

    @staticmethod
    def get_cluster_group_waveforms(
        clustering_id: str,
        cluster_label: int,
        max_waveforms: int = 100,
        max_points: int = 1000
    ) -> Dict[str, Any]:
        """
        Retrieves waveforms for a specific cluster group.
        Returns:
          - group sample count
          - all (or sampled) waveforms in the cluster (for faded spaghetti plot)
          - exact mean/average waveform (prominent)
          - std deviation (for confidence envelope)
          - min and max boundaries
        """
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM clusterings WHERE id = ?", (clustering_id,)).fetchone()
            if not row:
                raise ValueError(f"Clustering run {clustering_id} not found")

            c_data = dict(row)
            source_type = c_data["source_type"]
            source_id = c_data["source_id"]
            result_path = resolve_storage_path(c_data["result_path"])

            # Find dataset to load raw waveforms from
            if source_type in ("dataset", "processed"):
                dataset_id = source_id
            else:
                feat_row = conn.execute("SELECT dataset_id FROM features WHERE id = ?", (source_id,)).fetchone()
                if not feat_row:
                    raise ValueError(f"Features {source_id} not found")
                dataset_id = feat_row["dataset_id"]

        if not result_path or not result_path.exists():
            raise FileNotFoundError(f"Clustering result file not found: {c_data['result_path']}")

        # Load cluster assignments
        with np.load(str(result_path), allow_pickle=False) as npz:
            cluster_labels = npz["cluster_labels"]

        # Load raw waveforms
        data, labels, record = DataManager.load_dataset_arrays(dataset_id)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        data = np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        num_samples, seq_len = data.shape

        # Find sample indices belonging to cluster_label
        try:
            target_label = int(cluster_label)
        except (ValueError, TypeError):
            target_label = cluster_label
        indices = np.where(cluster_labels == target_label)[0]
        if len(indices) == 0:
            raise ValueError(f"No samples found for cluster label {cluster_label}")

        group_signals = data[indices]
        group_count = len(indices)

        # Compute exact mean, std, min, max waveforms
        mean_wf = np.mean(group_signals, axis=0)
        std_wf = np.std(group_signals, axis=0)
        min_wf = np.min(group_signals, axis=0)
        max_wf = np.max(group_signals, axis=0)

        # Downsample if length exceeds max_points
        if seq_len > max_points and max_points > 0:
            step = int(np.ceil(seq_len / max_points))
            x_axis = list(range(0, seq_len, step))
            mean_y = [float(v) if np.isfinite(v) else 0.0 for v in mean_wf[::step]]
            std_y = [float(v) if np.isfinite(v) else 0.0 for v in std_wf[::step]]
            min_y = [float(v) if np.isfinite(v) else 0.0 for v in min_wf[::step]]
            max_y = [float(v) if np.isfinite(v) else 0.0 for v in max_wf[::step]]
        else:
            step = 1
            x_axis = list(range(seq_len))
            mean_y = [float(v) if np.isfinite(v) else 0.0 for v in mean_wf]
            std_y = [float(v) if np.isfinite(v) else 0.0 for v in std_wf]
            min_y = [float(v) if np.isfinite(v) else 0.0 for v in min_wf]
            max_y = [float(v) if np.isfinite(v) else 0.0 for v in max_wf]

        # Upper and lower bounds for std
        upper_std = [float(m + s) if np.isfinite(m + s) else 0.0 for m, s in zip(mean_y, std_y)]
        lower_std = [float(m - s) if np.isfinite(m - s) else 0.0 for m, s in zip(mean_y, std_y)]

        # Sample up to max_waveforms for individual faded lines
        if group_count > max_waveforms:
            sub_sample_indices = np.linspace(0, group_count - 1, max_waveforms, dtype=int)
            sampled_signals = group_signals[sub_sample_indices]
            original_sample_ids = indices[sub_sample_indices].tolist()
        else:
            sampled_signals = group_signals
            original_sample_ids = indices.tolist()

        rendered_waveforms = []
        for orig_idx, sig in zip(original_sample_ids, sampled_signals):
            rendered_waveforms.append({
                "sample_index": int(orig_idx),
                "y": [float(v) if np.isfinite(v) else 0.0 for v in sig[::step].tolist()]
            })

        return sanitize_for_json({
            "clustering_id": clustering_id,
            "cluster_label": int(cluster_label),
            "sample_count": group_count,
            "rendered_count": len(rendered_waveforms),
            "sequence_length": seq_len,
            "x": x_axis,
            "mean": [float(v) for v in mean_y],
            "std": [float(v) for v in std_y],
            "upper_std": upper_std,
            "lower_std": lower_std,
            "min": [float(v) for v in min_y],
            "max": [float(v) for v in max_y],
            "waveforms": rendered_waveforms,
            "stats": {
                "peak_mean": float(np.max(mean_wf)) if len(mean_wf) > 0 and np.isfinite(np.max(mean_wf)) else 0.0,
                "trough_mean": float(np.min(mean_wf)) if len(mean_wf) > 0 and np.isfinite(np.min(mean_wf)) else 0.0,
                "overall_std": float(np.mean(std_wf)) if len(std_wf) > 0 and np.isfinite(np.mean(std_wf)) else 0.0
            }
        })
