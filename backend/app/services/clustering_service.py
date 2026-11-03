import os
import json
import uuid
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

from app.core.responses import sanitize_for_json
from app.core.config import CLUSTERS_DIR, resolve_storage_path
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
        """
        if source_type in ("dataset", "processed"):
            data, labels, record = DataManager.load_dataset_arrays(source_id)
            if data.ndim > 2:
                data = data.reshape(data.shape[0], -1)
            elif data.ndim == 1:
                data = data.reshape(data.shape[0], 1)
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            return data, labels, record
        elif source_type in ("features", "feature"):
            data, labels, record = FeatureExtractor.load_feature_matrix(source_id)
            if data.ndim > 2:
                data = data.reshape(data.shape[0], -1)
            elif data.ndim == 1:
                data = data.reshape(data.shape[0], 1)
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
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
        min_samples: int = 5,
        metric: str = "euclidean",
        # General parameters
        standardize: bool = True,
        random_state: int = 42,
        compute_pca: bool = True
    ) -> Dict[str, Any]:
        """
        Executes clustering and PCA dimension reduction on the input data matrix.
        Returns coordinates, cluster labels, metrics, and saves the result persistently.
        """
        X, ground_truth, source_meta = ClusteringService.load_source_matrix(source_type, source_id)
        num_samples, feature_dim = X.shape

        if num_samples < 2:
            raise ValueError("Clustering requires at least 2 samples.")

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
            clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
            cluster_labels = clusterer.fit_predict(X_proc)
            params = {
                "eps": eps,
                "min_samples": min_samples,
                "metric": metric
            }
        else:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

        # Compute cluster distribution
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        distribution = {int(k): int(v) for k, v in zip(unique_labels, counts)}

        # Metrics calculation
        metrics: Dict[str, Optional[float]] = {
            "silhouette_score": None,
            "adjusted_rand_index": None,
            "normalized_mutual_info": None
        }

        # Silhouette score requires at least 2 clusters and at least 2 samples per cluster
        non_noise_mask = cluster_labels != -1
        valid_labels = cluster_labels[non_noise_mask]
        num_unique_valid = len(np.unique(valid_labels))

        if num_unique_valid >= 2 and np.sum(non_noise_mask) > num_unique_valid:
            try:
                sil = silhouette_score(X_proc[non_noise_mask], valid_labels)
                if sil is not None and np.isfinite(sil):
                    metrics["silhouette_score"] = float(round(float(sil), 4))
                else:
                    metrics["silhouette_score"] = None
            except Exception:
                metrics["silhouette_score"] = None
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
        pca_2d_coords: Optional[np.ndarray] = None
        pca_3d_coords: Optional[np.ndarray] = None
        explained_variance_2d: List[float] = []
        explained_variance_3d: List[float] = []
        points: List[Dict[str, Any]] = []

        if compute_pca:
            total_variance = float(np.var(X_proc, axis=0).sum())
            has_variance = np.isfinite(total_variance) and total_variance > 1e-12

            if not has_variance:
                # Constant or zero-variance data: all samples are identical.
                # Points project to origin, explained variance is strictly 0.0.
                pca_2d_coords = np.zeros((num_samples, 2), dtype=np.float32)
                pca_3d_coords = np.zeros((num_samples, 3), dtype=np.float32)
                explained_variance_2d = [0.0, 0.0]
                explained_variance_3d = [0.0, 0.0, 0.0]
            elif feature_dim == 1:
                # 1D feature space edge case
                try:
                    pca1 = PCA(n_components=1, random_state=random_state)
                    c1 = pca1.fit_transform(X_proc)
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
            elif feature_dim == 2:
                try:
                    pca2 = PCA(n_components=2, random_state=random_state)
                    pca_2d_coords = pca2.fit_transform(X_proc)
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
                # feature_dim >= 3
                try:
                    pca2 = PCA(n_components=2, random_state=random_state)
                    pca_2d_coords = pca2.fit_transform(X_proc)
                    pca_2d_coords = np.nan_to_num(pca_2d_coords, nan=0.0, posinf=0.0, neginf=0.0)
                    explained_variance_2d = safe_explained_variance(pca2, 2)
                except Exception:
                    pca_2d_coords = np.zeros((num_samples, 2), dtype=np.float32)
                    explained_variance_2d = [0.0, 0.0]

                if min(num_samples, feature_dim) < 3:
                    # When samples < 3, 3-component PCA is mathematically impossible.
                    # Pad 2D coordinates with zeros for 3rd dimension to preserve 2D visualization.
                    pca_3d_coords = np.column_stack([pca_2d_coords, np.zeros((num_samples, 1), dtype=np.float32)])
                    explained_variance_3d = explained_variance_2d + [0.0]
                else:
                    try:
                        pca3 = PCA(n_components=3, random_state=random_state)
                        pca_3d_coords = pca3.fit_transform(X_proc)
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
                    point = {
                        "sample_id": i,
                        "cluster": int(cluster_labels[i]),
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

        # Save clustering results to persistent storage
        clustering_id = str(uuid.uuid4())
        dest_dir = CLUSTERS_DIR / project_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        result_file = dest_dir / f"{clustering_id}.npz"

        save_dict = {
            "cluster_labels": cluster_labels,
            "distribution": np.array(list(distribution.items()))
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
