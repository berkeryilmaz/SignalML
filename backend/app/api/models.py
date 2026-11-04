import json
import uuid
import io
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Body, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse, Response
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import keras
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from app.core.config import MODELS_DIR, resolve_storage_path
from app.core.database import get_db_connection
from app.core.responses import SafeJSONResponse
from app.services.data_manager import DataManager
from app.services.keras_model_builder import KerasModelBuilder, KerasGraphError, DEFAULT_CUSTOM_LAYER_TEMPLATE, ClusteringLayer
from app.services.training_service import TrainingService

router = APIRouter(prefix="/api/models", tags=["Neural Network Builder"])


@router.post("/validate")
def validate_keras_graph(payload: Dict[str, Any] = Body(...)):
    """
    Validates neural network node graph structure, infers tensor shapes for all intermediate layers,
    and returns parameter counts and architecture summary.
    """
    graph_data = payload.get("graph", payload)
    try:
        model, summary = KerasModelBuilder.build_model(graph_data)
        return {
            "valid": True,
            "summary": summary
        }
    except KerasGraphError as e:
        return {
            "valid": False,
            "error": str(e)
        }
    except Exception as e:
        return {
            "valid": False,
            "error": f"Unexpected validation error: {str(e)}"
        }


@router.post("/build")
def build_and_save_model(payload: Dict[str, Any] = Body(...)):
    """
    Builds a genuine Keras Functional API model from graph JSON,
    compiles it, saves to disk (.keras format), and registers in database.
    Supports wrapped {graph: ...} and direct graph payloads.
    """
    graph_data = payload.get("graph", payload)
    project_id = payload.get("project_id") or (graph_data.get("project_id") if isinstance(graph_data, dict) else None)
    name = payload.get("name", (graph_data.get("name") if isinstance(graph_data, dict) else None) or "SignalML Model")
    task_type = payload.get("task_type", (graph_data.get("task_type") if isinstance(graph_data, dict) else None) or "classification")
    optimizer = payload.get("optimizer", (graph_data.get("optimizer") if isinstance(graph_data, dict) else None) or "adam")
    learning_rate = float(payload.get("learning_rate", (graph_data.get("learning_rate") if isinstance(graph_data, dict) else None) or 0.001))
    loss = payload.get("loss", graph_data.get("loss") if isinstance(graph_data, dict) else None)

    if not project_id:
        raise HTTPException(status_code=400, detail="project_id is required")

    # Detect if the graph contains a ClusteringLayer
    raw_nodes = graph_data.get("nodes", []) if isinstance(graph_data, dict) else []
    if isinstance(raw_nodes, dict):
        raw_nodes = list(raw_nodes.values())
    has_clustering = any(
        "clustering" in str(n.get("type", "")).lower()
        for n in raw_nodes if isinstance(n, dict)
    )
    if has_clustering:
        task_type = "idec_clustering"

    try:
        model, summary = KerasModelBuilder.build_model(graph_data)
        compiled_model = KerasModelBuilder.compile_model(
            model=model,
            optimizer=optimizer,
            learning_rate=learning_rate,
            loss=loss
        )
    except KerasGraphError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build Keras model: {str(e)}")

    model_id = str(uuid.uuid4())
    dest_dir = MODELS_DIR / project_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    weights_path = dest_dir / f"{model_id}.keras"

    try:
        compiled_model.save(str(weights_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save model on disk: {str(e)}")

    now = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO models (
                id, project_id, name, task_type, graph_json, summary_json,
                weights_path, trainable_params, non_trainable_params, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_id, project_id, name, task_type, json.dumps(graph_data),
                json.dumps(summary), str(weights_path), summary["trainable_params"],
                summary["non_trainable_params"], now, now
            )
        )
        conn.execute(
            "UPDATE projects SET active_model_id = ?, updated_at = ? WHERE id = ?",
            (model_id, now, project_id)
        )
        conn.commit()

    return {
        "model_id": model_id,
        "name": name,
        "task_type": task_type,
        "weights_path": str(weights_path),
        "summary": summary
    }


@router.get("/{model_id}")
def get_model(model_id: str):
    """
    Retrieves the model row from SQLite models table by id = model_id.
    Parses graph_json into graph and summary_json into summary.
    """
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Model not found")
        data = dict(row)

    data["model_id"] = data["id"]
    try:
        data["graph"] = json.loads(data["graph_json"]) if data.get("graph_json") else {}
    except Exception:
        data["graph"] = {}
    try:
        data["summary"] = json.loads(data["summary_json"]) if data.get("summary_json") else {}
    except Exception:
        data["summary"] = {}

    return SafeJSONResponse(content=data)


@router.put("/{model_id}")
def update_model(model_id: str, payload: Dict[str, Any] = Body(...)):
    """Updates an existing loaded model graph architecture and rebuilds weights."""
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Model not found")
        project_id = row["project_id"]

    name = payload.get("name", row["name"])
    graph_data = payload.get("graph", {})
    task_type = payload.get("task_type", row["task_type"])

    try:
        model, summary = KerasModelBuilder.build_model(graph_data)
        compiled_model = KerasModelBuilder.compile_model(
            model=model,
            optimizer="adam",
            learning_rate=0.001,
            loss=None
        )
    except KerasGraphError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build Keras model for update: {str(e)}")

    dest_dir = MODELS_DIR / project_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    weights_path = dest_dir / f"{model_id}.keras"

    try:
        compiled_model.save(str(weights_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save model weights on disk: {str(e)}")

    now = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        conn.execute(
            """
            UPDATE models SET
                name = ?, task_type = ?, graph_json = ?, summary_json = ?,
                weights_path = ?, trainable_params = ?, non_trainable_params = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                name, task_type, json.dumps(graph_data), json.dumps(summary),
                str(weights_path), summary["trainable_params"], summary["non_trainable_params"],
                now, model_id
            )
        )
        conn.commit()

    return {
        "model_id": model_id,
        "name": name,
        "task_type": task_type,
        "weights_path": str(weights_path),
        "summary": summary,
        "graph": graph_data
    }


@router.delete("/{model_id}")
def delete_model(model_id: str):
    """Deletes a saved model from database and disk."""
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Model not found")
        weights_path = row["weights_path"]

    # Remove weights file if exists
    if weights_path:
        p = resolve_storage_path(weights_path)
        if p and p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    with get_db_connection() as conn:
        conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
        conn.execute("UPDATE projects SET active_model_id = NULL WHERE active_model_id = ?", (model_id,))
        conn.commit()

    return {"status": "success", "deleted_model_id": model_id}


@router.get("/list/{project_id}")
def list_project_models(project_id: str):
    """Lists all trained or configured models for a project."""
    with get_db_connection() as conn:
        rows = conn.execute("SELECT id, name, task_type, trainable_params, non_trainable_params, created_at FROM models WHERE project_id = ? ORDER BY created_at DESC", (project_id,)).fetchall()
        return [dict(r) for r in rows]


LAYER_TYPE_MAP = {
    "inputlayer": "input",
    "batchnormalization": "batchnorm",
    "layernormalization": "layernorm",
    "unitnormalization": "unitnorm",
    "maxpooling1d": "maxpooling1d",
    "averagepooling1d": "averagepooling1d",
    "globalmaxpooling1d": "global_maxpooling1d",
    "globalaveragepooling1d": "global_averagepooling1d",
    "conv1dtranspose": "conv1dtranspose",
    "upsampling1d": "upsampling1d",
    "multiheadattention": "multihead_attention",
    "positionalencoding": "positional_encoding",
}


def keras_model_to_litegraph(model: keras.Model) -> Dict[str, Any]:
    nodes = []
    links = []
    link_id = 1
    layer_name_to_id = {l.name: idx + 1 for idx, l in enumerate(model.layers)}

    # Parse inbound_nodes from config to determine inbound connections and depth
    inbound_map: Dict[str, List[str]] = {}

    def find_histories(obj, found):
        if isinstance(obj, dict):
            if "keras_history" in obj:
                found.append(obj["keras_history"])
            for v in obj.values():
                find_histories(v, found)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                find_histories(item, found)

    try:
        cfg = model.get_config()
        layers_cfg = cfg.get("layers", []) if isinstance(cfg, dict) else []

        for l_cfg in layers_cfg:
            target_name = l_cfg.get("name")
            if not target_name:
                continue
            histories = []
            find_histories(l_cfg.get("inbound_nodes", []), histories)
            inbound_map[target_name] = [
                hist[0] for hist in histories
                if isinstance(hist, (list, tuple)) and len(hist) >= 1 and isinstance(hist[0], str)
            ]
    except Exception:
        inbound_map = {}

    # Compute topological depth for each layer for clean left-to-right DAG layout
    depth: Dict[str, int] = {}
    for l in model.layers:
        inbounds = inbound_map.get(l.name, [])
        valid_inbounds = [orig for orig in inbounds if orig in depth]
        if valid_inbounds:
            depth[l.name] = max([depth[orig] for orig in valid_inbounds]) + 1
        else:
            depth[l.name] = 0

    # Layout coordinates: layers flow left-to-right, branches offset vertically
    depth_counts: Dict[int, int] = {}
    positions: Dict[str, List[int]] = {}
    for l in model.layers:
        d = depth.get(l.name, 0)
        row = depth_counts.get(d, 0)
        depth_counts[d] = row + 1
        positions[l.name] = [100 + d * 280, 100 + row * 150]

    for idx, l in enumerate(model.layers):
        node_id = idx + 1
        cname = l.__class__.__name__.lower()
        mapped_type = LAYER_TYPE_MAP.get(cname, cname)
        ntype = f"keras/{mapped_type}"
        props: Dict[str, Any] = {}
        if hasattr(l, "filters"): props["filters"] = int(l.filters)
        if hasattr(l, "kernel_size"): props["kernel_size"] = int(l.kernel_size[0]) if isinstance(l.kernel_size, (list, tuple)) else int(l.kernel_size)
        if hasattr(l, "strides"): props["strides"] = int(l.strides[0]) if isinstance(l.strides, (list, tuple)) else int(l.strides)
        if hasattr(l, "padding"): props["padding"] = str(l.padding)
        if hasattr(l, "units"): props["units"] = int(l.units)
        if hasattr(l, "activation") and l.activation:
            props["activation"] = getattr(l.activation, "__name__", "linear")
        if hasattr(l, "pool_size"): props["pool_size"] = int(l.pool_size[0]) if isinstance(l.pool_size, (list, tuple)) else int(l.pool_size)
        if hasattr(l, "size"): props["size"] = int(l.size[0]) if isinstance(l.size, (list, tuple)) else int(l.size)
        if hasattr(l, "rate"): props["rate"] = float(l.rate)
        if hasattr(l, "return_sequences"): props["return_sequences"] = bool(l.return_sequences)
        if cname == "inputlayer":
            batch_shape = getattr(l, "batch_shape", None)
            if not batch_shape and hasattr(l, "input_shape"):
                batch_shape = l.input_shape
            if batch_shape and len(batch_shape) > 1:
                props["shape"] = ", ".join(str(s) for s in batch_shape[1:] if s is not None)

        pos = positions.get(l.name, [100 + idx * 280, 100])

        # Multi-input layers receive proper multiple input ports so slot 1 is preserved
        if cname in ("inputlayer", "input"):
            inputs = []
        elif mapped_type in ("multihead_attention", "mha") or cname in ("multiheadattention", "mha"):
            inputs = [{"name": "query", "type": "Tensor"}, {"name": "value", "type": "Tensor"}]
        elif mapped_type in ("concatenate", "add", "multiply", "average") or cname in ("concatenate", "add", "multiply", "average"):
            in_count = max(2, len(inbound_map.get(l.name, [])))
            inputs = [{"name": f"in{i+1}", "type": "Tensor"} for i in range(in_count)]
        else:
            inputs = [{"name": "in", "type": "Tensor"}]

        outputs = [{"name": "out", "type": "Tensor"}]

        node = {
            "id": node_id,
            "type": ntype,
            "title": l.name,
            "pos": pos,
            "size": [200, 90],
            "properties": props,
            "inputs": inputs,
            "outputs": outputs
        }
        nodes.append(node)

    # Reconstruct links from inbound_nodes
    try:
        cfg = model.get_config()
        layers_cfg = cfg.get("layers", []) if isinstance(cfg, dict) else []
        has_history_links = False

        for l_cfg in layers_cfg:
            target_name = l_cfg.get("name")
            target_id = layer_name_to_id.get(target_name)
            if not target_id:
                continue
            inbounds = l_cfg.get("inbound_nodes", [])
            histories = []
            find_histories(inbounds, histories)
            for slot_idx, hist in enumerate(histories):
                if isinstance(hist, (list, tuple)) and len(hist) >= 1:
                    origin_name = hist[0]
                    origin_id = layer_name_to_id.get(origin_name)
                    if origin_id:
                        links.append([link_id, origin_id, 0, target_id, slot_idx, "Tensor"])
                        link_id += 1
                        has_history_links = True

        if not has_history_links:
            for i in range(1, len(nodes)):
                links.append([i, i, 0, i + 1, 0, "Tensor"])
    except Exception:
        for i in range(1, len(nodes)):
            links.append([i, i, 0, i + 1, 0, "Tensor"])

    return {
        "last_node_id": len(nodes),
        "last_link_id": len(links),
        "nodes": nodes,
        "links": links
    }


@router.post("/import")
async def import_model(
    file: UploadFile = File(...),
    project_id: str = Form(...),
    name: Optional[str] = Form(None),
    task_type: Optional[str] = Form(None)
):
    """
    Imports an exported Keras model (.keras / .h5) or graph architecture (.json).
    Validates model structure, infers shapes/parameters, saves to project models,
    and registers in the database.
    """
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id is required")

    filename = file.filename or "imported_model"
    suffix = Path(filename).suffix.lower()
    if suffix not in (".keras", ".h5", ".json"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported model format. Only .keras, .h5, or .json files can be imported."
        )

    model_name = name.strip() if name and name.strip() else Path(filename).stem
    model_id = str(uuid.uuid4())
    dest_dir = MODELS_DIR / project_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_weights_path = dest_dir / f"{model_id}.keras"

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    detected_task_type = task_type or "classification"
    graph_data: Dict[str, Any] = {}
    summary: Dict[str, Any] = {}

    if suffix in (".keras", ".h5"):
        with open(dest_weights_path, "wb") as f:
            f.write(content)

        try:
            model = keras.models.load_model(str(dest_weights_path))
        except Exception as e:
            if dest_weights_path.exists():
                dest_weights_path.unlink()
            raise HTTPException(status_code=400, detail=f"Failed to load Keras model: {str(e)}")

        trainable = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
        non_trainable = int(np.sum([np.prod(v.shape) for v in model.non_trainable_weights]))
        total_params = trainable + non_trainable

        layers_info = []
        for idx, l in enumerate(model.layers):
            out_shape = getattr(l, "output_shape", None)
            if out_shape is None and hasattr(l, "output"):
                try:
                    out_shape = tuple(l.output.shape)
                except Exception:
                    out_shape = None
            layers_info.append({
                "node_id": str(idx + 1),
                "layer_type": l.__class__.__name__,
                "output_shape": list(out_shape) if isinstance(out_shape, tuple) else out_shape,
                "params": l.count_params()
            })

        summary = {
            "total_params": total_params,
            "trainable_params": trainable,
            "non_trainable_params": non_trainable,
            "layers": layers_info,
            "input_shapes": [list(model.input_shape)] if getattr(model, "input_shape", None) else [],
            "output_shapes": [list(model.output_shape)] if getattr(model, "output_shape", None) else []
        }

        # Auto-detect task type
        if not task_type:
            if getattr(model, "output_shape", None) == getattr(model, "input_shape", None):
                detected_task_type = "reconstruction"
            elif getattr(model, "output_shape", None) and len(model.output_shape) > 1 and model.output_shape[-1] == 1:
                detected_task_type = "regression"
            else:
                detected_task_type = "classification"

        # Generate visual LiteGraph nodes for the canvas using helper
        graph_data = keras_model_to_litegraph(model)

        # Compile model if needed
        if not getattr(model, "optimizer", None):
            loss = "mse" if detected_task_type in ("regression", "reconstruction") else "sparse_categorical_crossentropy"
            metrics = ["mae"] if detected_task_type in ("regression", "reconstruction") else ["accuracy"]
            model.compile(optimizer="adam", loss=loss, metrics=metrics)
            model.save(str(dest_weights_path))

    elif suffix == ".json":
        try:
            parsed = json.loads(content.decode("utf-8"))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON file: {str(e)}")

        if isinstance(parsed, dict) and ("nodes" in parsed or "last_node_id" in parsed):
            graph_data = parsed
            try:
                model, summary = KerasModelBuilder.build_model(graph_data)
                compiled_model = KerasModelBuilder.compile_model(
                    model=model,
                    optimizer="adam",
                    learning_rate=0.001,
                    loss=None
                )
                compiled_model.save(str(dest_weights_path))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to build model from graph JSON: {str(e)}")
        elif isinstance(parsed, dict) and "graph" in parsed and isinstance(parsed["graph"], dict) and ("nodes" in parsed["graph"] or "last_node_id" in parsed["graph"]):
            graph_data = parsed["graph"]
            try:
                model, summary = KerasModelBuilder.build_model(graph_data)
                compiled_model = KerasModelBuilder.compile_model(
                    model=model,
                    optimizer="adam",
                    learning_rate=0.001,
                    loss=None
                )
                compiled_model.save(str(dest_weights_path))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to build model from graph JSON: {str(e)}")
        else:
            try:
                json_str = json.dumps(parsed) if isinstance(parsed, dict) else str(parsed)
                model = keras.models.model_from_json(json_str)
                loss = "mse" if detected_task_type in ("regression", "reconstruction") else "sparse_categorical_crossentropy"
                model.compile(optimizer="adam", loss=loss)
                model.save(str(dest_weights_path))

                trainable = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
                non_trainable = int(np.sum([np.prod(v.shape) for v in model.non_trainable_weights]))
                layers_info = []
                for idx, l in enumerate(model.layers):
                    out_shape = getattr(l, "output_shape", None)
                    if out_shape is None and hasattr(l, "output"):
                        try:
                            out_shape = tuple(l.output.shape)
                        except Exception:
                            out_shape = None
                    layers_info.append({
                        "node_id": str(idx + 1),
                        "layer_type": l.__class__.__name__,
                        "output_shape": list(out_shape) if isinstance(out_shape, tuple) else out_shape,
                        "params": l.count_params()
                    })
                summary = {
                    "total_params": trainable + non_trainable,
                    "trainable_params": trainable,
                    "non_trainable_params": non_trainable,
                    "layers": layers_info,
                    "input_shapes": [list(model.input_shape)] if model.input_shape else [],
                    "output_shapes": [list(model.output_shape)] if model.output_shape else []
                }
                graph_data = keras_model_to_litegraph(model)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to parse Keras architecture JSON: {str(e)}")

    now = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO models (
                id, project_id, name, task_type, graph_json, summary_json,
                weights_path, trainable_params, non_trainable_params, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_id, project_id, model_name, detected_task_type,
                json.dumps(graph_data), json.dumps(summary), str(dest_weights_path),
                summary.get("trainable_params", 0), summary.get("non_trainable_params", 0),
                now, now
            )
        )
        conn.execute(
            "UPDATE projects SET active_model_id = ?, updated_at = ? WHERE id = ?",
            (model_id, now, project_id)
        )
        conn.commit()

    return {
        "model_id": model_id,
        "name": model_name,
        "task_type": detected_task_type,
        "trainable_params": summary.get("trainable_params", 0),
        "total_params": summary.get("total_params", 0),
        "weights_path": str(dest_weights_path),
        "summary": summary,
        "graph": graph_data
    }


@router.get("/{model_id}/export_json")
def export_model_json(model_id: str):
    """Exports model graph architecture as JSON file download."""
    with get_db_connection() as conn:
        row = conn.execute("SELECT name, graph_json FROM models WHERE id = ?", (model_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Model not found")

        graph = json.loads(row["graph_json"] or "{}")
        content = json.dumps(graph, indent=2)
        filename = f"{row['name'].replace(' ', '_')}_{model_id[:8]}_graph.json"
        return Response(
            content=content,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )


@router.post("/{model_id}/freeze")
def freeze_model_layers(
    model_id: str,
    payload: Dict[str, Any] = Body(...)
):
    """
    Selectively freezes or unfreezes model layers.
    Accepts frozen_layers (list of layer names to freeze), learning_rate, and optimizer.
    Recompiles model while preserving existing weights, saves updated .keras file,
    and updates trainable/non-trainable parameters count and summary_json in SQLite.
    """
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Model not found")
        model_record = dict(row)

    weights_path = resolve_storage_path(model_record.get("weights_path"))
    if not weights_path or not weights_path.exists():
        raise HTTPException(status_code=404, detail=f"Model weights file not found on disk: {model_record.get('weights_path')}")

    try:
        model = keras.models.load_model(str(weights_path), custom_objects={"ClusteringLayer": ClusteringLayer})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load Keras model: {str(e)}")

    frozen_layers = set(payload.get("frozen_layers", []))
    lr = float(payload.get("learning_rate", 0.001))
    opt_name = str(payload.get("optimizer", "adam"))

    # Apply trainable flags to layers
    for layer in model.layers:
        layer.trainable = (layer.name not in frozen_layers)

    task_type = model_record.get("task_type", "classification")
    is_idec = (task_type == "idec_clustering") or any(
        isinstance(l, ClusteringLayer) or "clustering" in l.__class__.__name__.lower()
        for l in model.layers
    )

    loss_weights = None
    if is_idec:
        outputs = model.outputs if isinstance(model.outputs, list) else [model.output]
        clustering_idx = 0
        for idx, out_tensor in enumerate(outputs):
            hist = getattr(out_tensor, "_keras_history", None)
            layer = hist[0] if isinstance(hist, tuple) else getattr(hist, "operation", None)
            if isinstance(layer, ClusteringLayer) or (layer and "clustering" in getattr(layer, "name", "").lower()):
                clustering_idx = idx
                break
        if clustering_idx == 0:
            loss = ["kld", "mse"]
            loss_weights = [0.1, 1.0]
            metrics = [["accuracy"], ["mae"]]
        else:
            loss = ["mse", "kld"]
            loss_weights = [1.0, 0.1]
            metrics = [["mae"], ["accuracy"]]
    else:
        loss = "mse" if task_type in ("regression", "reconstruction") else "sparse_categorical_crossentropy"
        metrics = ["mae"] if task_type in ("regression", "reconstruction") else ["accuracy"]

    # Recompile model while preserving existing weights
    try:
        compiled_model = KerasModelBuilder.compile_model(
            model=model,
            optimizer=opt_name,
            learning_rate=lr,
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights
        )
        compiled_model.save(str(weights_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to recompile/save frozen model: {str(e)}")

    total_params = int(compiled_model.count_params())
    trainable_params = int(sum(np.prod(p.shape) for p in compiled_model.trainable_weights))
    non_trainable_params = total_params - trainable_params

    try:
        summary = json.loads(model_record.get("summary_json") or "{}")
    except Exception:
        summary = {}

    summary["total_params"] = total_params
    summary["trainable_params"] = trainable_params
    summary["non_trainable_params"] = non_trainable_params

    summary_layers = summary.get("layers", [])
    for l_info in summary_layers:
        l_name = l_info.get("layer_name") or l_info.get("name")
        if l_name:
            l_info["trainable"] = (l_name not in frozen_layers)
    summary["layers"] = summary_layers

    now = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        conn.execute(
            """
            UPDATE models
            SET trainable_params = ?, non_trainable_params = ?, summary_json = ?, updated_at = ?
            WHERE id = ?
            """,
            (trainable_params, non_trainable_params, json.dumps(summary), now, model_id)
        )
        conn.commit()

    return {
        "success": True,
        "model_id": model_id,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "summary": summary
    }


# =====================================================================
# Custom Layer CRUD Endpoints
# =====================================================================
@router.get("/custom_layers/template/default")
def get_default_custom_layer_template():
    """Returns the default starter template for custom Keras layers with Turkish comments."""
    return {"template": DEFAULT_CUSTOM_LAYER_TEMPLATE}


@router.get("/custom_layers/{project_id}")
def list_custom_layers(project_id: str):
    """Lists all user-defined custom layers for a project."""
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM custom_layers WHERE project_id = ? ORDER BY created_at DESC",
            (project_id,)
        ).fetchall()
        return [dict(r) for r in rows]


@router.post("/custom_layers")
def create_custom_layer(payload: Dict[str, Any] = Body(...)):
    """Registers a new custom Keras layer for a project with Python syntax validation."""
    project_id = payload.get("project_id")
    name = payload.get("name")
    class_name = payload.get("class_name", "CustomSignalLayer")
    code = payload.get("code")
    description = payload.get("description", "")

    if not project_id:
        raise HTTPException(status_code=400, detail="project_id is required")
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="name is required")
    if not code or not code.strip():
        raise HTTPException(status_code=400, detail="code is required")

    try:
        compile(code, f"<custom_layer_{class_name}>", "exec")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Python syntax error in layer code: {str(e)}")

    layer_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO custom_layers (id, project_id, name, class_name, code, description, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (layer_id, project_id, name.strip(), class_name.strip(), code, description, now, now)
        )
        conn.commit()

    return {
        "id": layer_id,
        "project_id": project_id,
        "name": name,
        "class_name": class_name,
        "code": code,
        "description": description,
        "created_at": now,
        "updated_at": now
    }


@router.put("/custom_layers/{layer_id}")
def update_custom_layer(layer_id: str, payload: Dict[str, Any] = Body(...)):
    """Updates an existing custom Keras layer with Python syntax validation."""
    name = payload.get("name")
    class_name = payload.get("class_name")
    code = payload.get("code")
    description = payload.get("description")

    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM custom_layers WHERE id = ?", (layer_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Custom layer not found")

        curr = dict(row)
        new_name = name.strip() if name else curr["name"]
        new_class = class_name.strip() if class_name else curr["class_name"]
        new_code = code if code is not None else curr["code"]
        new_desc = description if description is not None else curr["description"]

        if code:
            try:
                compile(new_code, f"<custom_layer_{new_class}>", "exec")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Python syntax error in layer code: {str(e)}")

        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            UPDATE custom_layers
            SET name = ?, class_name = ?, code = ?, description = ?, updated_at = ?
            WHERE id = ?
            """,
            (new_name, new_class, new_code, new_desc, now, layer_id)
        )
        conn.commit()

    return {
        "id": layer_id,
        "project_id": curr["project_id"],
        "name": new_name,
        "class_name": new_class,
        "code": new_code,
        "description": new_desc,
        "updated_at": now
    }


@router.delete("/custom_layers/{layer_id}")
def delete_custom_layer(layer_id: str):
    """Deletes a custom layer from project."""
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM custom_layers WHERE id = ?", (layer_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Custom layer not found")
        conn.execute("DELETE FROM custom_layers WHERE id = ?", (layer_id,))
        conn.commit()
    return {"status": "deleted", "id": layer_id}


# =====================================================================
# IDEC Cluster Probabilities & Analysis Endpoints
# =====================================================================
def _load_model_and_dataset_inputs(model_id: str, dataset_id: str):
    """Loads model from disk, prepares dataset arrays, and aligns tensor shapes."""
    with get_db_connection() as conn:
        m_row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
        if not m_row:
            raise HTTPException(status_code=404, detail="Model not found")
        model_record = dict(m_row)

    weights_path = resolve_storage_path(model_record.get("weights_path"))
    if not weights_path or not weights_path.exists():
        raise HTTPException(status_code=404, detail="Model weights file not found on disk")

    try:
        model = keras.models.load_model(str(weights_path), custom_objects={"ClusteringLayer": ClusteringLayer})
    except Exception:
        graph_json = json.loads(model_record.get("graph_json", "{}"))
        model, _ = KerasModelBuilder.build_model(graph_json)

    rec = None
    try:
        rec = DataManager.get_dataset(dataset_id)
    except Exception:
        pass

    try:
        prep = TrainingService.prepare_model_inputs(model, dataset_id)
        X = prep.X
        input_ids = prep.sample_ids
    except Exception:
        X, y, dataset_record = DataManager.load_dataset_arrays(dataset_id)
        if not rec:
            rec = dataset_record
        input_ids = dataset_record.get("sample_ids", [])

    in_shape = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape
    if isinstance(X, np.ndarray):
        if len(in_shape) == 3:
            if X.ndim == 2:
                X = np.expand_dims(X, axis=-1)
            elif X.ndim == 1:
                X = X.reshape(-1, 1, 1)
            expected_len = in_shape[1]
            if expected_len is not None and X.shape[1] != expected_len:
                if X.shape[1] > expected_len:
                    X = X[:, :expected_len, :]
                else:
                    X = np.pad(X, ((0, 0), (0, expected_len - X.shape[1]), (0, 0)))
        elif len(in_shape) == 2:
            if X.ndim > 2:
                X = X.reshape(len(X), -1)
            elif X.ndim == 1:
                X = X.reshape(-1, 1)
            expected_dim = in_shape[1]
            if expected_dim is not None and X.shape[1] != expected_dim:
                if X.shape[1] > expected_dim:
                    X = X[:, :expected_dim]
                else:
                    X = np.pad(X, ((0, 0), (0, expected_dim - X.shape[1])))
    X = np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    n_samples = len(X[0]) if isinstance(X, list) else len(X)
    if not input_ids or len(input_ids) != n_samples:
        raw_file_path = resolve_storage_path(rec.get("file_path")) if rec else None
        if raw_file_path and raw_file_path.exists():
            try:
                df_preview = pd.read_csv(raw_file_path)
                for c in ["Input ID", "input_id", "ID", "id", "SampleID", "signal_id"]:
                    if c in df_preview.columns:
                        input_ids = df_preview[c].astype(str).tolist()
                        break
            except Exception:
                pass
    if not input_ids or len(input_ids) != n_samples:
        input_ids = [f"SIG_{i:04d}" for i in range(n_samples)]

    return model, X, input_ids, model_record, rec, weights_path


def _extract_model_idec_probabilities(
    model: keras.Model,
    X: Any,
    n_clusters_param: Optional[int] = None,
    calibrate: bool = True,
    epochs: int = 0,
    weights_path: Optional[Path] = None,
    return_details: bool = False
) -> Any:
    """
    Computes soft cluster probabilities Q for the given model and inputs.
    - Always locates the bottleneck layer and clustering output.
    - If epochs > 0: attaches ClusteringLayer if missing, initializes via KMeans,
      compiles dual-output IDEC model with loss=['kld', 'mse'], loss_weights=[0.1, 1.0],
      runs iterative IDEC training, and saves updated weights to weights_path.
    - Extracts bottleneck features Z and normalizes NaNs/Infs.
    - If ClusteringLayer exists and Q_raw is non-degenerate (and n_clusters matches): uses Q_raw.
    - If no ClusteringLayer, or n_clusters differs, or calibrate=True and Q_raw is degenerate:
      fits KMeans, updates ClusteringLayer weights if shape matches, and computes Student's t
      probabilities with distance scaling to prevent flat 1/K probabilities.
    """
    try:
        outputs = model.outputs if isinstance(model.outputs, list) else [model.output]
        clustering_layer = None
        clustering_out_idx = None

        # 1. Locate ClusteringLayer and its output index if present
        for l in model.layers:
            if isinstance(l, ClusteringLayer) or l.__class__.__name__ == "ClusteringLayer" or "clustering" in getattr(l, "name", "").lower():
                clustering_layer = l
                break

        for idx, out_tensor in enumerate(outputs):
            layer = None
            if hasattr(out_tensor, "_keras_history"):
                hist = out_tensor._keras_history
                layer = hist[0] if isinstance(hist, tuple) else getattr(hist, "operation", None)
            if layer is not None and (isinstance(layer, ClusteringLayer) or layer.__class__.__name__ == "ClusteringLayer" or "clustering" in getattr(layer, "name", "").lower()):
                clustering_out_idx = idx
                if clustering_layer is None:
                    clustering_layer = layer
                break

        if clustering_out_idx is None and len(outputs) == 2:
            output_shapes = model.output_shape if isinstance(model.output_shape, list) else [model.output_shape]
            if len(output_shapes[1]) == 2 and len(output_shapes[0]) > 2:
                clustering_out_idx = 1
            elif len(output_shapes[0]) == 2 and len(output_shapes[1]) > 2:
                clustering_out_idx = 0

        # 2. Locate bottleneck layer and tensor
        bottleneck_layer = None
        bottleneck_tensor = None

        if clustering_layer is not None:
            try:
                bottleneck_tensor = clustering_layer.input
                if isinstance(bottleneck_tensor, list):
                    bottleneck_tensor = bottleneck_tensor[0]
                if hasattr(bottleneck_tensor, "_keras_history"):
                    hist = bottleneck_tensor._keras_history
                    bottleneck_layer = hist[0] if isinstance(hist, tuple) else getattr(hist, "operation", None)
            except Exception:
                bottleneck_tensor = None

        if bottleneck_layer is None or bottleneck_tensor is None:
            dense_candidates = [
                l for l in model.layers[1:-1]
                if isinstance(l, keras.layers.Dense) or l.__class__.__name__ == "Dense"
            ]
            if dense_candidates:
                bottleneck_layer = min(dense_candidates, key=lambda l: getattr(l, "units", 999999))
            else:
                pool_candidates = [
                    l for l in model.layers[:-1]
                    if any(k in l.__class__.__name__.lower() for k in ("flatten", "globalaveragepooling", "globalmaxpooling", "latent", "bottleneck", "embedding"))
                ]
                if pool_candidates:
                    bottleneck_layer = pool_candidates[-1]
                else:
                    mid_idx = max(0, (len(model.layers) - 1) // 2)
                    bottleneck_layer = model.layers[mid_idx]

            bottleneck_tensor = bottleneck_layer.output
            if isinstance(bottleneck_tensor, list):
                bottleneck_tensor = bottleneck_tensor[0]

        bt_for_cluster = bottleneck_tensor
        if len(bt_for_cluster.shape) > 2:
            bt_for_cluster = keras.layers.Flatten(name="latent_flatten")(bt_for_cluster)

        encoder = keras.Model(inputs=model.inputs, outputs=bottleneck_tensor)

        # 3. Optional quick IDEC training if epochs > 0
        if epochs > 0:
            target_k_init = int(n_clusters_param) if n_clusters_param and n_clusters_param > 1 else (
                getattr(clustering_layer, "n_clusters", 5) if clustering_layer is not None else 5
            )
            recon_out = outputs[1 - clustering_out_idx] if (len(outputs) == 2 and clustering_out_idx is not None) else outputs[0]

            if clustering_layer is None or (n_clusters_param is not None and getattr(clustering_layer, "n_clusters", None) != n_clusters_param):
                clustering_layer = ClusteringLayer(n_clusters=target_k_init, name="clustering_idec")
                clustering_out = clustering_layer(bt_for_cluster)
                model = keras.Model(inputs=model.inputs, outputs=[clustering_out, recon_out])
                clustering_out_idx = 0
                outputs = model.outputs
            elif len(outputs) == 1:
                clustering_out = clustering_layer(bt_for_cluster)
                model = keras.Model(inputs=model.inputs, outputs=[clustering_out, recon_out])
                clustering_out_idx = 0
                outputs = model.outputs

            Z_init = encoder.predict(X, verbose=0)
            if Z_init.ndim > 2:
                Z_init = Z_init.reshape(len(Z_init), -1)
            Z_init = np.nan_to_num(Z_init, copy=False)

            actual_k_init = min(target_k_init, len(Z_init))
            km_init = KMeans(n_clusters=actual_k_init, n_init=20, random_state=42)
            km_init.fit(Z_init)
            if actual_k_init == target_k_init:
                clustering_layer.set_weights([km_init.cluster_centers_.astype(np.float32)])

            target_recon = np.copy(X)
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

            loss_list = ["kld", "mse"] if clustering_out_idx == 0 else ["mse", "kld"]
            weights_list = [0.1, 1.0] if clustering_out_idx == 0 else [1.0, 0.1]
            model.compile(
                loss=loss_list,
                loss_weights=weights_list,
                optimizer=keras.optimizers.Adam(learning_rate=0.001)
            )

            def compute_target_distribution(q: np.ndarray) -> np.ndarray:
                weight = q ** 2 / np.maximum(q.sum(axis=0), 1e-10)
                return (weight.T / np.maximum(weight.sum(axis=1), 1e-10)).T

            for ep in range(epochs):
                preds_ep = model.predict(X, verbose=0)
                Q_curr = preds_ep[clustering_out_idx] if isinstance(preds_ep, list) else preds_ep
                P = compute_target_distribution(Q_curr)
                y_targets = [P, target_recon] if clustering_out_idx == 0 else [target_recon, P]
                model.fit(X, y_targets, batch_size=min(32, len(X)), epochs=1, verbose=0, shuffle=True)

            if weights_path:
                try:
                    model.save(str(weights_path))
                except Exception:
                    pass

        # 4. Predict bottleneck latent features Z
        encoder = keras.Model(inputs=model.inputs, outputs=bottleneck_tensor)
        Z = encoder.predict(X, verbose=0)
        if Z.ndim > 2:
            Z = Z.reshape(len(Z), -1)
        Z = np.nan_to_num(Z, copy=False, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # 5. Determine target_k
        target_k = int(n_clusters_param) if n_clusters_param and n_clusters_param > 1 else (
            int(getattr(clustering_layer, "n_clusters", 5)) if clustering_layer is not None else 5
        )
        target_k = max(2, min(target_k, len(Z))) if len(Z) >= 2 else 1

        # 6. Compute soft probabilities Q
        Q = None
        need_calibration = True

        if clustering_out_idx is not None and clustering_layer is not None:
            preds = model.predict(X, verbose=0)
            Q_raw = preds[clustering_out_idx] if isinstance(preds, list) else preds
            if Q_raw.ndim == 2:
                k_raw = Q_raw.shape[1]
                if n_clusters_param is None or n_clusters_param == k_raw:
                    is_degenerate = (len(np.unique(np.argmax(Q_raw, axis=1))) < 2) or (
                        float(np.mean(np.max(Q_raw, axis=1))) <= (1.0 / k_raw) + 0.02
                    )
                    if not is_degenerate or not calibrate:
                        Q = Q_raw
                        need_calibration = False

        if need_calibration:
            kmeans = KMeans(n_clusters=target_k, n_init=20, random_state=42)
            kmeans.fit(Z)
            if clustering_layer is not None:
                try:
                    curr_w = clustering_layer.get_weights()
                    if curr_w and curr_w[0].shape == (target_k, Z.shape[1]):
                        clustering_layer.set_weights([kmeans.cluster_centers_.astype(np.float32)])
                        if weights_path:
                            try:
                                model.save(str(weights_path))
                            except Exception:
                                pass
                except Exception:
                    pass

            alpha = 1.0
            expanded_z = np.expand_dims(Z, axis=1)
            expanded_mu = np.expand_dims(kmeans.cluster_centers_, axis=0)
            dist_sq = np.sum(np.square(expanded_z - expanded_mu), axis=-1)

            scale = float(np.mean(np.min(dist_sq, axis=1)))
            dist_sq_norm = dist_sq / max(scale, 1e-6)

            q = (1.0 + (dist_sq_norm / alpha)) ** (-(alpha + 1.0) / 2.0)
            q = q / np.maximum(np.sum(q, axis=-1, keepdims=True), 1e-10)
            Q = np.clip(q, 1e-7, 1.0 - 1e-7)

        # 7. Extract reconstruction predictions if available
        recon_preds = None
        try:
            outputs_curr = model.outputs if isinstance(model.outputs, list) else [model.output]
            if len(outputs_curr) == 2 and clustering_out_idx is not None:
                preds = model.predict(X, verbose=0)
                recon_preds = preds[1 - clustering_out_idx]
            elif len(outputs_curr) == 1 and clustering_out_idx is None:
                preds = model.predict(X, verbose=0)
                recon_preds = preds if not isinstance(preds, list) else preds[0]
        except Exception:
            recon_preds = None

        if return_details:
            return {
                "Q": Q,
                "Z": Z,
                "bottleneck_layer": bottleneck_layer,
                "clustering_layer": clustering_layer,
                "clustering_out_idx": clustering_out_idx,
                "model": model,
                "recon_preds": recon_preds
            }
        return Q

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute bottleneck IDEC probabilities: {str(e)}")


@router.get("/{model_id}/idec_probabilities/{dataset_id}")
def get_idec_probabilities(
    model_id: str,
    dataset_id: str,
    n_clusters: Optional[int] = Query(default=None, alias="n_clusters"),
    calibrate: bool = Query(default=True, description="Auto-calibrate uninitialized/degenerate cluster centers via KMeans on bottleneck Z"),
    epochs: int = Query(default=0, ge=0, le=200, description="Optional quick IDEC training epochs before analysis")
):
    """
    Computes soft cluster probabilities (Q) and rich deep clustering analysis from the model
    and bottleneck embeddings on the given dataset.
    """
    model, X, input_ids, model_record, rec, weights_path = _load_model_and_dataset_inputs(model_id, dataset_id)

    details = _extract_model_idec_probabilities(
        model=model,
        X=X,
        n_clusters_param=n_clusters,
        calibrate=calibrate,
        epochs=epochs,
        weights_path=weights_path,
        return_details=True
    )

    Q = details["Q"]
    Z = details["Z"]
    bottleneck_layer = details["bottleneck_layer"]
    recon_preds = details["recon_preds"]

    n_samples = len(X)
    n_clusters_out = int(Q.shape[1])
    assigned_clusters = np.argmax(Q, axis=1)
    confidences = np.max(Q, axis=1)
    unique_c, counts_c = np.unique(assigned_clusters, return_counts=True)
    cluster_counts = {int(k): int(v) for k, v in zip(unique_c, counts_c)}

    # Clustering evaluation metrics on latent bottleneck Z
    silhouette_val = None
    if 1 < len(unique_c) < n_samples:
        try:
            silhouette_val = round(float(silhouette_score(Z, assigned_clusters)), 4)
        except Exception:
            silhouette_val = None

    davies_bouldin_val = None
    if len(unique_c) > 1:
        try:
            davies_bouldin_val = round(float(davies_bouldin_score(Z, assigned_clusters)), 4)
        except Exception:
            davies_bouldin_val = None

    calinski_harabasz_val = None
    if 1 < len(unique_c) < n_samples:
        try:
            calinski_harabasz_val = round(float(calinski_harabasz_score(Z, assigned_clusters)), 4)
        except Exception:
            calinski_harabasz_val = None

    # Reconstruction evaluation metrics if matching reconstruction output exists
    reconstruction_mse = None
    reconstruction_mae = None
    if recon_preds is not None:
        try:
            X_flat = X.reshape(n_samples, -1)
            recon_flat = np.asarray(recon_preds).reshape(len(recon_preds), -1)
            if X_flat.shape == recon_flat.shape:
                mse_val = float(np.mean(np.square(X_flat - recon_flat)))
                mae_val = float(np.mean(np.abs(X_flat - recon_flat)))
                if np.isfinite(mse_val):
                    reconstruction_mse = round(mse_val, 4)
                if np.isfinite(mae_val):
                    reconstruction_mae = round(mae_val, 4)
        except Exception:
            pass

    # Per-cluster statistics and Group Overlay waveform profiles
    cluster_stats = []
    all_indices = np.arange(n_samples)
    for c in range(n_clusters_out):
        mask = (assigned_clusters == c)
        count_c = int(np.sum(mask))
        pct = round(float(count_c / max(n_samples, 1) * 100.0), 2)
        if count_c > 0:
            conf_c = round(float(np.mean(confidences[mask])), 4)
            c_signals = X[mask]
            c_sample_indices = all_indices[mask]
            if c_signals.ndim == 3:
                c_sig_1d = c_signals[:, :, 0]
            elif c_signals.ndim == 2:
                c_sig_1d = c_signals
            else:
                c_sig_1d = c_signals.reshape(count_c, -1)

            orig_seq_len = int(c_sig_1d.shape[1])
            mean_sig = np.mean(c_sig_1d, axis=0)
            std_sig = np.std(c_sig_1d, axis=0)
            upper_sig = mean_sig + std_sig
            lower_sig = mean_sig - std_sig

            peak_mean = round(float(np.max(mean_sig)), 4)
            trough_mean = round(float(np.min(mean_sig)), 4)
            overall_std = round(float(np.mean(std_sig)), 4)

            if orig_seq_len > 240:
                sub_idx = np.linspace(0, orig_seq_len - 1, 240, dtype=int)
                x_coords = [int(i) for i in sub_idx]
                mean_sig_ds = mean_sig[sub_idx]
                std_sig_ds = std_sig[sub_idx]
                upper_sig_ds = upper_sig[sub_idx]
                lower_sig_ds = lower_sig[sub_idx]
                c_sig_ds = c_sig_1d[:, sub_idx]
            else:
                x_coords = list(range(orig_seq_len))
                mean_sig_ds = mean_sig
                std_sig_ds = std_sig
                upper_sig_ds = upper_sig
                lower_sig_ds = lower_sig
                c_sig_ds = c_sig_1d

            mean_wf = [round(float(v), 4) for v in mean_sig_ds]
            std_wf = [round(float(v), 4) for v in std_sig_ds]
            upper_wf = [round(float(v), 4) for v in upper_sig_ds]
            lower_wf = [round(float(v), 4) for v in lower_sig_ds]

            max_overlay = min(60, count_c)
            if count_c > max_overlay:
                chosen_pos = np.linspace(0, count_c - 1, max_overlay, dtype=int)
            else:
                chosen_pos = np.arange(count_c)

            indiv_waveforms = []
            for pos_i in chosen_pos:
                s_idx = int(c_sample_indices[pos_i])
                indiv_waveforms.append({
                    "sample_index": s_idx,
                    "id": str(input_ids[s_idx]),
                    "y": [round(float(v), 4) for v in c_sig_ds[pos_i]]
                })

            stats_dict = {
                "peak_mean": peak_mean,
                "trough_mean": trough_mean,
                "overall_std": overall_std
            }
        else:
            conf_c = 0.0
            orig_seq_len = int(X.shape[1]) if X.ndim >= 2 else 0
            x_coords = []
            mean_wf = []
            std_wf = []
            upper_wf = []
            lower_wf = []
            max_overlay = 0
            indiv_waveforms = []
            stats_dict = {
                "peak_mean": 0.0,
                "trough_mean": 0.0,
                "overall_std": 0.0
            }

        cluster_stats.append({
            "cluster": int(c),
            "count": count_c,
            "rendered_count": int(max_overlay),
            "sequence_length": orig_seq_len,
            "percentage": pct,
            "mean_confidence": conf_c,
            "x": x_coords,
            "mean_waveform": mean_wf,
            "std_waveform": std_wf,
            "upper_std": upper_wf,
            "lower_std": lower_wf,
            "waveforms": indiv_waveforms,
            "stats": stats_dict
        })

    # 2D PCA representation of bottleneck features Z
    coords = None
    pca_explained_variance = [0.0, 0.0]
    if Z.shape[1] == 1:
        coords = np.column_stack([Z[:, 0], np.zeros(n_samples, dtype=np.float32)])
        pca_explained_variance = [100.0, 0.0]
    elif n_samples >= 2 and Z.shape[1] >= 2:
        try:
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(Z)
            coords = np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
            pca_explained_variance = [round(float(v * 100.0), 2) for v in pca.explained_variance_ratio_]
        except Exception:
            coords = np.zeros((n_samples, 2), dtype=np.float32)
            pca_explained_variance = [0.0, 0.0]
    else:
        coords = np.zeros((n_samples, 2), dtype=np.float32)
        pca_explained_variance = [0.0, 0.0]

    max_pca_pts = min(600, n_samples)
    pca_points = [
        {
            "id": str(input_ids[i]),
            "x": round(float(coords[i, 0]), 4),
            "y": round(float(coords[i, 1]), 4),
            "cluster": int(assigned_clusters[i]),
            "confidence": round(float(confidences[i]), 4)
        }
        for i in range(max_pca_pts)
    ]

    # Sample predictions (up to 200 samples)
    max_samples = min(200, n_samples)
    samples = [
        {
            "id": str(input_ids[i]),
            "assigned_cluster": int(assigned_clusters[i]),
            "confidence": round(float(confidences[i]), 4),
            "probabilities": [round(float(p), 4) for p in Q[i]]
        }
        for i in range(max_samples)
    ]

    return {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "model_name": model_record.get("name"),
        "dataset_name": rec.get("name") if rec else dataset_id,
        "total_samples": int(n_samples),
        "n_clusters": n_clusters_out,
        "active_clusters": int(len(unique_c)),
        "latent_dim": int(Z.shape[1]),
        "bottleneck_layer_name": str(getattr(bottleneck_layer, "name", "bottleneck")),
        "mean_confidence": round(float(np.mean(confidences)), 4),
        "silhouette_score": silhouette_val,
        "davies_bouldin_score": davies_bouldin_val,
        "calinski_harabasz_score": calinski_harabasz_val,
        "reconstruction_mse": reconstruction_mse,
        "reconstruction_mae": reconstruction_mae,
        "cluster_counts": cluster_counts,
        "cluster_stats": cluster_stats,
        "pca_points": pca_points,
        "pca_explained_variance": pca_explained_variance,
        "samples": samples
    }


@router.get("/{model_id}/idec_probabilities/{dataset_id}/export_csv")
def export_idec_probabilities_csv(
    model_id: str,
    dataset_id: str,
    n_clusters: Optional[int] = Query(default=None, alias="n_clusters"),
    calibrate: bool = Query(default=True, description="Auto-calibrate uninitialized/degenerate cluster centers via KMeans on bottleneck Z"),
    epochs: int = Query(default=0, ge=0, le=200, description="Optional quick IDEC training epochs before analysis")
):
    """Exports all IDEC cluster probabilities with Input ID as a downloadable CSV."""
    model, X, input_ids, model_record, rec, weights_path = _load_model_and_dataset_inputs(model_id, dataset_id)

    Q = _extract_model_idec_probabilities(
        model=model,
        X=X,
        n_clusters_param=n_clusters,
        calibrate=calibrate,
        epochs=epochs,
        weights_path=weights_path,
        return_details=False
    )

    assigned_clusters = np.argmax(Q, axis=1)
    confidences = np.max(Q, axis=1)

    df_out = pd.DataFrame({
        "Input_ID": input_ids,
        "Assigned_Cluster": assigned_clusters,
        "Confidence": np.round(confidences, 4)
    })
    for c in range(Q.shape[1]):
        df_out[f"Cluster_{c}_Prob"] = np.round(Q[:, c], 4)

    csv_buffer = io.StringIO()
    df_out.to_csv(csv_buffer, index=False)
    csv_str = csv_buffer.getvalue()

    model_name_clean = (model_record.get("name") or "model").replace(" ", "_")
    filename = f"idec_cluster_probabilities_{model_name_clean}_{model_id[:8]}.csv"
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


# =====================================================================
# Python Functional API Exporter Endpoints
# =====================================================================
@router.post("/export_python_preview")
def export_python_preview(payload: Dict[str, Any] = Body(...)):
    """Generates Python Functional API code string from graph payload for modal preview."""
    graph_data = payload.get("graph", payload)
    name = payload.get("model_name", "SignalML_Functional_Model")
    task_type = payload.get("task_type", "classification")

    try:
        code = KerasModelBuilder.generate_functional_python_code(
            graph_data=graph_data,
            model_name=name,
            task_type=task_type
        )
        return {"code": code}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to generate Python code: {str(e)}")


@router.get("/{model_id}/export_python")
def export_model_python(model_id: str):
    """Exports saved model architecture as a runnable Python Functional API script download."""
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Model not found")

        model_record = dict(row)
        graph = json.loads(model_record.get("graph_json", "{}"))
        name = model_record.get("name", "SignalML_Functional_Model")
        task_type = model_record.get("task_type", "classification")

        try:
            code = KerasModelBuilder.generate_functional_python_code(
                graph_data=graph,
                model_name=name,
                task_type=task_type
            )
            safe_name = name.replace(" ", "_")
            filename = f"{safe_name}_{model_id[:8]}_functional_api.py"
            return Response(
                content=code,
                media_type="text/x-python",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'}
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to export Python code: {str(e)}")

