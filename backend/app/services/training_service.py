import os
import time
import json
import uuid
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from app.core.config import MODELS_DIR, CHECKPOINTS_DIR, DATASETS_DIR, resolve_storage_path
from app.core.database import get_db_connection
from app.services.data_manager import DataManager
from app.services.keras_model_builder import KerasModelBuilder, ClusteringLayer
from datetime import datetime, timezone


def _sanitize_json_floats(obj: Any) -> Any:
    """Recursively replaces NaN, +inf, -inf with None for compliant JSON serialization."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    elif isinstance(obj, dict):
        return {k: _sanitize_json_floats(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_json_floats(v) for v in obj]
    return obj


class TrainingJob:
    def __init__(self, job_id: str, project_id: str, model_id: str, dataset_id: str, config: Dict[str, Any]):
        self.job_id = job_id
        self.project_id = project_id
        self.model_id = model_id
        self.dataset_id = dataset_id
        self.config = config
        
        self.status = "pending"  # pending, running, completed, stopped, failed
        self.current_epoch = 0
        self.total_epochs = config.get("epochs", 20)
        self.history: Dict[str, List[Optional[float]]] = {}
        self.evaluation: Optional[Dict[str, Any]] = None
        self.error_message: Optional[str] = None
        
        self.start_time: float = 0.0
        self.elapsed_seconds: float = 0.0
        self.cancel_requested = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "project_id": self.project_id,
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "status": self.status,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "history": _sanitize_json_floats(self.history),
            "evaluation": _sanitize_json_floats(self.evaluation),
            "error": self.error_message,
            "elapsed_seconds": round(self.elapsed_seconds, 2)
        }


class PreparedDataset(tuple):
    def __new__(cls, X, y, record, sample_ids=None):
        return super().__new__(cls, (X, y, record))

    def __init__(self, X, y, record, sample_ids=None):
        self.X = X
        self.y = y
        self.record = record
        self.sample_ids = sample_ids or (record.get("sample_ids") if isinstance(record, dict) else [])


class CancelCallback(keras.callbacks.Callback):
    def __init__(self, job: TrainingJob, epoch_offset: int = 0):
        super().__init__()
        self.job = job
        self.epoch_offset = epoch_offset

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        actual_epoch = epoch + self.epoch_offset + 1
        self.job.current_epoch = actual_epoch
        self.job.elapsed_seconds = time.time() - self.job.start_time
        
        has_diverged = False
        for k, v in logs.items():
            if k not in self.job.history:
                self.job.history[k] = []
            if v is not None and np.isfinite(v):
                val = float(round(v, 6))
            else:
                val = None
                has_diverged = True
            self.job.history[k].append(val)

        if has_diverged:
            self.model.stop_training = True
            self.job.status = "failed"
            self.job.error_message = (
                f"Eğitim {actual_epoch}. epokta durduruldu: Kayıp (loss) değeri 'NaN' oldu (divergence). "
                "Lütfen öğrenme oranını (learning rate) düşürün veya veri kümesini kontrol edin."
            )
            return

        if self.job.cancel_requested.is_set():
            self.model.stop_training = True
            self.job.status = "stopped"


class TrainingService:
    _active_jobs: Dict[str, TrainingJob] = {}
    _lock = threading.Lock()

    @classmethod
    def get_job(cls, job_id: str) -> Optional[TrainingJob]:
        with cls._lock:
            return cls._active_jobs.get(job_id)

    @classmethod
    def cancel_job(cls, job_id: str) -> bool:
        with cls._lock:
            job = cls._active_jobs.get(job_id)
            if job and job.status == "running":
                job.cancel_requested.set()
                return True
        return False

    @classmethod
    def start_training(
        cls,
        project_id: str,
        model_id: str,
        dataset_id: str,
        config: Dict[str, Any]
    ) -> TrainingJob:
        job_id = str(uuid.uuid4())
        job = TrainingJob(job_id, project_id, model_id, dataset_id, config)

        with cls._lock:
            cls._active_jobs[job_id] = job

        # Record in DB
        now = datetime.now(timezone.utc).isoformat()
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO training_runs (
                    id, project_id, model_id, dataset_id, status,
                    config_json, history_json, evaluation_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id, project_id, model_id, dataset_id, "pending",
                    json.dumps(config), "{}", "{}", now
                )
            )
            conn.commit()

        # Launch training in worker thread
        thread = threading.Thread(target=cls._run_training_worker, args=(job,), daemon=True)
        job.thread = thread
        thread.start()

        return job

    @classmethod
    def prepare_model_inputs(
        cls,
        model: keras.Model,
        dataset_id: str
    ) -> PreparedDataset:
        """
        Loads dataset (.npz) containing data, X_seq, X_tabular, y, labels, sample_ids.
        Adapts shapes and ranks to match single or multi-input Keras models.
        """
        record = DataManager.get_dataset(dataset_id)
        if not record:
            raise ValueError(f"Dataset {dataset_id} not found")

        file_path = resolve_storage_path(record["file_path"])
        if not file_path or not file_path.exists():
            cand = DATASETS_DIR / record.get("project_id", "") / f"{dataset_id}.npz"
            if cand.exists():
                file_path = cand
            else:
                raise FileNotFoundError(f"Dataset file missing on disk: {record['file_path']}")

        with np.load(str(file_path), allow_pickle=False) as npz:
            data = npz["data"] if "data" in npz else None
            X_seq = npz["X_seq"] if "X_seq" in npz else None
            X_tabular = npz["X_tabular"] if "X_tabular" in npz else None
            labels = npz["labels"] if "labels" in npz else (npz["y"] if "y" in npz else None)
            raw_sample_ids = npz["sample_ids"] if "sample_ids" in npz else None

        # Determine total samples
        n_samples = 0
        for cand_arr in (data, X_seq, X_tabular):
            if cand_arr is not None:
                n_samples = len(cand_arr)
                break

        if raw_sample_ids is not None:
            sample_ids = [
                s.decode("utf-8") if isinstance(s, bytes) else str(s)
                for s in raw_sample_ids
            ]
        else:
            sample_ids = [f"sample_{i}" for i in range(n_samples)]

        if labels is not None:
            labels = np.nan_to_num(np.array(labels), copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Inspect model input shape
        model_input_shapes = model.input_shape
        if isinstance(model_input_shapes, list) and len(model_input_shapes) > 1:
            multi_inputs = []
            for in_shape in model_input_shapes:
                rank = len(in_shape)
                if rank == 3:
                    # 3D: (None, seq_len, channels)
                    if X_seq is not None:
                        arr = np.array(X_seq, dtype=np.float32)
                    elif data is not None:
                        arr = np.array(data, dtype=np.float32)
                    elif X_tabular is not None:
                        arr = np.array(X_tabular, dtype=np.float32)
                    else:
                        raise ValueError("No input data array found in dataset .npz for 3D input")

                    if arr.ndim == 2:
                        arr = np.expand_dims(arr, axis=-1)
                    elif arr.ndim == 1:
                        arr = arr.reshape(-1, 1, 1)

                    expected_len = in_shape[1]
                    if expected_len is not None and arr.shape[1] != expected_len:
                        if arr.shape[1] > expected_len:
                            arr = arr[:, :expected_len, :]
                        else:
                            arr = np.pad(arr, ((0, 0), (0, expected_len - arr.shape[1]), (0, 0)))

                    expected_ch = in_shape[2]
                    if expected_ch is not None and arr.shape[2] != expected_ch:
                        if arr.shape[2] > expected_ch:
                            arr = arr[:, :, :expected_ch]
                        elif arr.shape[2] == 1 and expected_ch > 1:
                            arr = np.repeat(arr, expected_ch, axis=-1)

                    arr = np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                    multi_inputs.append(arr)

                elif rank == 2:
                    # 2D: (None, feat_dim)
                    if X_tabular is not None:
                        arr = np.array(X_tabular, dtype=np.float32)
                    elif data is not None:
                        arr = np.array(data, dtype=np.float32)
                    elif X_seq is not None:
                        arr = np.array(X_seq, dtype=np.float32)
                    else:
                        raise ValueError("No input data array found in dataset .npz for 2D input")

                    if arr.ndim > 2:
                        arr = arr.reshape(len(arr), -1)
                    elif arr.ndim == 1:
                        arr = arr.reshape(-1, 1)

                    expected_dim = in_shape[1]
                    if expected_dim is not None and arr.shape[1] != expected_dim:
                        if arr.shape[1] > expected_dim:
                            arr = arr[:, :expected_dim]
                        else:
                            arr = np.pad(arr, ((0, 0), (0, expected_dim - arr.shape[1])))

                    arr = np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                    multi_inputs.append(arr)
                else:
                    arr = np.array(data if data is not None else (X_tabular if X_tabular is not None else X_seq), dtype=np.float32)
                    arr = np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                    multi_inputs.append(arr)

            X = multi_inputs
        else:
            in_shape = model_input_shapes[0] if isinstance(model_input_shapes, list) else model_input_shapes
            arr = data if data is not None else (X_seq if X_seq is not None else X_tabular)
            if arr is None:
                raise ValueError("No input array found in dataset .npz")
            arr = np.array(arr, dtype=np.float32)
            if len(in_shape) == 3 and arr.ndim == 2:
                arr = np.expand_dims(arr, axis=-1)
            elif len(in_shape) == 2 and arr.ndim > 2:
                arr = arr.reshape(len(arr), -1)
            elif len(in_shape) == 2 and arr.ndim == 1:
                arr = arr.reshape(-1, 1)

            X = np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        record["sample_ids"] = sample_ids
        return PreparedDataset(X, labels, record, sample_ids)

    @classmethod
    def _run_training_worker(cls, job: TrainingJob):
        job.status = "running"
        job.start_time = time.time()

        try:
            # Load model record
            with get_db_connection() as conn:
                row = conn.execute("SELECT * FROM models WHERE id = ?", (job.model_id,)).fetchone()
                if not row:
                    raise ValueError(f"Model {job.model_id} not found")
                model_record = dict(row)

            weights_path = resolve_storage_path(model_record.get("weights_path"))
            if not weights_path or not weights_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_record.get('weights_path')}")

            model = keras.models.load_model(str(weights_path))

            is_resumed = bool(job.config.get("resume", False))
            epoch_offset = 0

            if is_resumed:
                # Load previous run's history and compute epoch_offset
                with get_db_connection() as conn:
                    prev_run = conn.execute(
                        """
                        SELECT history_json FROM training_runs
                        WHERE model_id = ? AND id != ? AND status IN ('completed', 'stopped')
                        ORDER BY created_at DESC LIMIT 1
                        """,
                        (job.model_id, job.job_id)
                    ).fetchone()
                    if prev_run and prev_run["history_json"]:
                        try:
                            prev_hist = json.loads(prev_run["history_json"])
                            if isinstance(prev_hist, dict):
                                for k, v in prev_hist.items():
                                    if isinstance(v, list):
                                        job.history[k] = list(v)
                                if "loss" in job.history:
                                    epoch_offset = len(job.history["loss"])
                        except Exception:
                            pass
                job.total_epochs = int(job.config.get("epochs", 20)) + epoch_offset
            else:
                # Weight sanity check: if fresh training and loaded model has NaN weights, rebuild from graph
                has_corrupt_weights = any(
                    np.isnan(w).any() or np.isinf(w).any()
                    for layer in model.layers
                    for w in layer.get_weights()
                )
                if has_corrupt_weights:
                    graph_json = json.loads(model_record.get("graph_json", "{}"))
                    model, _ = KerasModelBuilder.build_model(graph_json)

            # Apply frozen_layers if provided
            frozen_layers = job.config.get("frozen_layers")
            if frozen_layers:
                frozen_set = set(frozen_layers)
                for layer in model.layers:
                    layer.trainable = (layer.name not in frozen_set)

            # Prepare dataset inputs (supports single and multi-input)
            X, y, dataset_record = cls.prepare_model_inputs(model, job.dataset_id)

            # Determine task type and targets
            task_type = model_record.get("task_type", "classification")

            # Check if model contains ClusteringLayer or is a 2-output IDEC model
            clustering_layer = None
            clustering_out_idx = 0
            is_idec = False

            outputs = model.outputs if isinstance(model.outputs, list) else [model.output]
            for layer in model.layers:
                if isinstance(layer, ClusteringLayer) or layer.__class__.__name__ == "ClusteringLayer" or "clustering" in getattr(layer, "name", "").lower():
                    clustering_layer = layer
                    break

            if clustering_layer is not None or len(outputs) == 2:
                is_idec = True
                for idx, out_tensor in enumerate(outputs):
                    layer = None
                    if hasattr(out_tensor, "_keras_history"):
                        hist = out_tensor._keras_history
                        layer = hist[0] if isinstance(hist, tuple) else getattr(hist, "operation", None)
                    if layer is not None:
                        if isinstance(layer, ClusteringLayer) or layer.__class__.__name__ == "ClusteringLayer" or "clustering" in getattr(layer, "name", "").lower():
                            clustering_out_idx = idx
                            clustering_layer = layer
                            break
                if clustering_out_idx == 0 and len(outputs) == 2:
                    output_shapes = model.output_shape if isinstance(model.output_shape, list) else [model.output_shape]
                    if len(output_shapes[1]) == 2 and len(output_shapes[0]) > 2:
                        clustering_out_idx = 1

            val_split = float(job.config.get("val_split", 0.2))
            shuffle = bool(job.config.get("shuffle", True))
            seed = job.config.get("seed", 42)
            batch_size = int(job.config.get("batch_size", 32))
            epochs = int(job.config.get("epochs", 20))
            opt_name = job.config.get("optimizer", "adam")
            lr = float(job.config.get("learning_rate", 0.001))

            callbacks = [CancelCallback(job, epoch_offset=epoch_offset)]

            if is_idec:
                # Automated IDEC Training:
                # 1. Initialize cluster centers with KMeans on bottleneck layer embeddings
                if clustering_layer is not None and hasattr(clustering_layer, "input"):
                    try:
                        encoder = keras.Model(inputs=model.inputs, outputs=clustering_layer.input)
                        bottleneck_features = encoder.predict(X, verbose=0)
                        n_clusters = getattr(clustering_layer, "n_clusters", 8)
                        kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
                        kmeans.fit(bottleneck_features)
                        clustering_layer.set_weights([kmeans.cluster_centers_])
                    except Exception:
                        pass

                # 2. Compute initial high-confidence target distribution P
                def compute_target_distribution(q: np.ndarray) -> np.ndarray:
                    weight = q ** 2 / np.maximum(q.sum(axis=0), 1e-10)
                    return (weight.T / np.maximum(weight.sum(axis=1), 1e-10)).T

                initial_preds = model.predict(X, verbose=0)
                q_initial = initial_preds[clustering_out_idx] if isinstance(initial_preds, list) else initial_preds
                P_target = compute_target_distribution(q_initial)

                recon_target = X[0] if isinstance(X, list) else X
                if clustering_out_idx == 0:
                    targets = [P_target, recon_target]
                else:
                    targets = [recon_target, P_target]

                n_samples = len(X[0]) if isinstance(X, list) else len(X)
                if val_split > 0.0 and n_samples > 4:
                    idx_train, idx_val = train_test_split(
                        np.arange(n_samples), test_size=val_split, shuffle=shuffle, random_state=seed
                    )
                    if isinstance(X, list):
                        X_train = [arr[idx_train] for arr in X]
                        X_val = [arr[idx_val] for arr in X]
                    else:
                        X_train, X_val = X[idx_train], X[idx_val]
                    y_train = [t[idx_train] for t in targets]
                    y_val = [t[idx_val] for t in targets]
                    validation_data = (X_val, y_val)
                else:
                    X_train = X
                    y_train = targets
                    validation_data = None

                # Compile with IDEC losses & loss weights
                losses = ["kld", "mse"] if clustering_out_idx == 0 else ["mse", "kld"]
                loss_weights = [0.1, 1.0] if clustering_out_idx == 0 else [1.0, 0.1]
                model = KerasModelBuilder.compile_model(
                    model,
                    optimizer=opt_name,
                    learning_rate=lr,
                    loss=losses,
                    loss_weights=loss_weights
                )

                if job.config.get("early_stopping", False):
                    patience = int(job.config.get("patience", 5))
                    callbacks.append(keras.callbacks.EarlyStopping(
                        monitor="val_loss" if validation_data is not None else "loss",
                        patience=patience,
                        restore_best_weights=True
                    ))

                # Train IDEC model
                model.fit(
                    X_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=validation_data,
                    callbacks=callbacks,
                    verbose=0
                )

                if job.status == "failed":
                    return

                # Evaluation for IDEC
                eval_X = X_val if validation_data is not None else X_train
                y_pred = model.predict(eval_X, verbose=0)
                q_eval = y_pred[clustering_out_idx] if isinstance(y_pred, list) else y_pred
                recon_eval = y_pred[1 - clustering_out_idx] if isinstance(y_pred, list) else y_pred

                pred_clusters = np.argmax(q_eval, axis=1)
                cluster_conf = float(np.mean(np.max(q_eval, axis=1)))
                unique_c, counts_c = np.unique(pred_clusters, return_counts=True)
                eval_X_arr = eval_X[0] if isinstance(eval_X, list) else eval_X
                rec_mse = float(mean_squared_error(eval_X_arr.flatten(), recon_eval.flatten()))
                rec_mae = float(mean_absolute_error(eval_X_arr.flatten(), recon_eval.flatten()))

                evaluation_results = {
                    "task_type": "idec_clustering",
                    "n_clusters": int(q_eval.shape[1]),
                    "mean_cluster_confidence": round(cluster_conf, 4),
                    "cluster_distribution": {str(k): int(v) for k, v in zip(unique_c, counts_c)},
                    "reconstruction_mse": round(rec_mse, 6),
                    "reconstruction_mae": round(rec_mae, 6)
                }
                job.evaluation = evaluation_results

            else:
                # Standard Classification / Regression Pipeline (supports single or multi-input X)
                if y is None:
                    y = X.copy() if not isinstance(X, list) else X[0].copy()
                    task_type = "regression"

                n_samples = len(X[0]) if isinstance(X, list) else len(X)
                if val_split > 0.0 and n_samples > 4:
                    indices = np.arange(n_samples)
                    train_idx, val_idx = train_test_split(indices, test_size=val_split, shuffle=shuffle, random_state=seed)
                    if isinstance(X, list):
                        X_train = [arr[train_idx] for arr in X]
                        X_val = [arr[val_idx] for arr in X]
                    else:
                        X_train = X[train_idx]
                        X_val = X[val_idx]
                    y_train = y[train_idx] if y is not None else None
                    y_val = y[val_idx] if y is not None else None
                    validation_data = (X_val, y_val)
                else:
                    X_train, y_train = X, y
                    validation_data = None

                if job.config.get("early_stopping", False):
                    patience = int(job.config.get("patience", 5))
                    monitor = "val_loss" if validation_data is not None else "loss"
                    callbacks.append(keras.callbacks.EarlyStopping(
                        monitor=monitor,
                        patience=patience,
                        restore_best_weights=True
                    ))

                # Compile if optimizer not present or if frozen_layers specified
                if frozen_layers or not getattr(model, "optimizer", None):
                    loss_name = job.config.get("loss") or ("mse" if task_type == "regression" else "sparse_categorical_crossentropy")
                    metrics = job.config.get("metrics") or (["mae"] if task_type == "regression" else ["accuracy"])
                    opt_instance = keras.optimizers.get({"class_name": opt_name, "config": {"learning_rate": lr}})
                    model.compile(optimizer=opt_instance, loss=loss_name, metrics=metrics)

                model.fit(
                    X_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=validation_data,
                    callbacks=callbacks,
                    verbose=0
                )

                if job.status == "failed":
                    return

                # Run evaluation on validation set or training set
                eval_X = X_val if validation_data is not None else X_train
                eval_y = y_val if validation_data is not None else y_train

                y_pred = model.predict(eval_X, verbose=0)
                evaluation_results = cls._compute_evaluation_metrics(eval_y, y_pred, task_type)
                job.evaluation = evaluation_results

            # Save updated trained model weights as long as weights are finite
            # Save weights even when cancelled early (job.cancel_requested.is_set())
            has_nan_weights = any(
                np.isnan(w).any() or np.isinf(w).any()
                for layer in model.layers
                for w in layer.get_weights()
            )
            if not has_nan_weights:
                dest_model_path = Path(weights_path)
                model.save(str(dest_model_path))

            if job.cancel_requested.is_set():
                job.status = "stopped"
            elif job.status != "failed":
                job.status = "completed"

        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)

        finally:
            job.elapsed_seconds = time.time() - job.start_time
            now = datetime.now(timezone.utc).isoformat()
            
            # Update SQLite database
            try:
                with get_db_connection() as conn:
                    conn.execute(
                        """
                        UPDATE training_runs
                        SET status = ?, history_json = ?, evaluation_json = ?,
                            error_message = ?, finished_at = ?
                        WHERE id = ?
                        """,
                        (
                            job.status,
                            json.dumps(_sanitize_json_floats(job.history)),
                            json.dumps(_sanitize_json_floats(job.evaluation or {})),
                            job.error_message,
                            now,
                            job.job_id
                        )
                    )
                    conn.commit()
            except Exception:
                pass

    @staticmethod
    def _compute_evaluation_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: str
    ) -> Dict[str, Any]:
        """Calculates evaluation metrics (classification or regression)."""
        eval_metrics = {}
        y_true = np.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

        if task_type == "classification":
            # For softmax or probabilities
            if y_pred.ndim > 1 and y_pred.shape[1] > 1:
                pred_classes = np.argmax(y_pred, axis=1)
            else:
                pred_classes = (y_pred.flatten() > 0.5).astype(int)

            if y_true.ndim > 1 and y_true.shape[1] > 1:
                true_classes = np.argmax(y_true, axis=1)
            else:
                true_classes = y_true.flatten().astype(int)

            try:
                unique_classes = np.unique(np.concatenate([true_classes, pred_classes]))
                cm = confusion_matrix(true_classes, pred_classes, labels=unique_classes)
                eval_metrics["confusion_matrix"] = cm.tolist()
                eval_metrics["class_labels"] = [str(c) for c in unique_classes]
                eval_metrics["classification_report"] = classification_report(
                    true_classes, pred_classes, labels=unique_classes, output_dict=True, zero_division=0
                )
                eval_metrics["accuracy"] = float(round(np.mean(true_classes == pred_classes), 4))
            except Exception as e:
                eval_metrics["error"] = str(e)
        else:
            # Regression metrics
            try:
                y_true_flat = y_true.flatten()
                y_pred_flat = y_pred.flatten()
                
                mse = float(mean_squared_error(y_true_flat, y_pred_flat))
                mae = float(mean_absolute_error(y_true_flat, y_pred_flat))
                rmse = float(np.sqrt(mse))
                r2 = float(r2_score(y_true_flat, y_pred_flat))

                eval_metrics["mse"] = round(mse, 6)
                eval_metrics["mae"] = round(mae, 6)
                eval_metrics["rmse"] = round(rmse, 6)
                eval_metrics["r2"] = round(r2, 4)
            except Exception as e:
                eval_metrics["error"] = str(e)

        return eval_metrics
