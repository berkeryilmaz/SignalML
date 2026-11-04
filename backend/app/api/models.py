import json
import uuid
import io
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Body, UploadFile, File, Form
from fastapi.responses import JSONResponse, Response
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import keras

from app.core.config import MODELS_DIR, resolve_storage_path
from app.core.database import get_db_connection
from app.services.data_manager import DataManager
from app.services.keras_model_builder import KerasModelBuilder, KerasGraphError, DEFAULT_CUSTOM_LAYER_TEMPLATE, ClusteringLayer

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
    """
    project_id = payload.get("project_id")
    name = payload.get("name", "SignalML Model")
    graph_data = payload.get("graph", {})
    task_type = payload.get("task_type", "classification")
    optimizer = payload.get("optimizer", "adam")
    learning_rate = float(payload.get("learning_rate", 0.001))
    loss = payload.get("loss")

    if not project_id:
        raise HTTPException(status_code=400, detail="project_id is required")

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
    layer_name_to_id = {}

    for idx, l in enumerate(model.layers):
        node_id = idx + 1
        layer_name_to_id[l.name] = node_id
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

        col = idx % 4
        row = idx // 4
        pos = [120 + col * 260, 100 + row * 180]

        node = {
            "id": node_id,
            "type": ntype,
            "title": l.name,
            "pos": pos,
            "size": [200, 90],
            "properties": props,
            "inputs": [{"name": "in", "type": "Tensor"}] if idx > 0 else [],
            "outputs": [{"name": "out", "type": "Tensor"}] if idx < len(model.layers) - 1 else []
        }
        nodes.append(node)

    # Reconstruct links from inbound_nodes
    try:
        cfg = model.get_config()
        layers_cfg = cfg.get("layers", [])
        has_history_links = False

        def find_histories(obj, found):
            if isinstance(obj, dict):
                if "keras_history" in obj:
                    found.append(obj["keras_history"])
                for v in obj.values():
                    find_histories(v, found)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    find_histories(item, found)

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
        model = keras.models.load_model(str(weights_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load Keras model: {str(e)}")

    frozen_layers = set(payload.get("frozen_layers", []))
    lr = float(payload.get("learning_rate", 0.001))
    opt_name = str(payload.get("optimizer", "adam"))

    # Apply trainable flags to layers
    for layer in model.layers:
        layer.trainable = (layer.name not in frozen_layers)

    task_type = model_record.get("task_type", "classification")
    loss = "mse" if task_type in ("regression", "reconstruction") else "sparse_categorical_crossentropy"
    metrics = ["mae"] if task_type in ("regression", "reconstruction") else ["accuracy"]

    # Recompile model while preserving existing weights
    try:
        compiled_model = KerasModelBuilder.compile_model(
            model=model,
            optimizer=opt_name,
            learning_rate=lr,
            loss=loss,
            metrics=metrics
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
# IDEC Cluster Probabilities Endpoints
# =====================================================================
@router.get("/{model_id}/idec_probabilities/{dataset_id}")
def get_idec_probabilities(model_id: str, dataset_id: str):
    """
    Computes soft cluster probabilities (Q) from the model's ClusteringLayer on dataset,
    returns assigned clusters, confidence scores, and preview of sample predictions.
    """
    with get_db_connection() as conn:
        m_row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
        if not m_row:
            raise HTTPException(status_code=404, detail="Model not found")
        model_record = dict(m_row)

    weights_path = resolve_storage_path(model_record.get("weights_path"))
    if not weights_path or not weights_path.exists():
        raise HTTPException(status_code=404, detail="Model weights file not found on disk")

    try:
        model = keras.models.load_model(str(weights_path))
    except Exception:
        graph_json = json.loads(model_record.get("graph_json", "{}"))
        model, _ = KerasModelBuilder.build_model(graph_json)

    try:
        X, y, dataset_record = DataManager.load_dataset_arrays(dataset_id)
        X = np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load dataset: {str(e)}")

    model_input_shape = model.input_shape
    if isinstance(model_input_shape, list):
        model_input_shape = model_input_shape[0]

    if len(model_input_shape) == 3 and X.ndim == 2:
        X = np.expand_dims(X, axis=-1)

    outputs = model.outputs if isinstance(model.outputs, list) else [model.output]
    clustering_out_idx = 0
    found_clustering = False

    for idx, out_tensor in enumerate(outputs):
        layer = None
        if hasattr(out_tensor, "_keras_history"):
            hist = out_tensor._keras_history
            layer = hist[0] if isinstance(hist, tuple) else getattr(hist, "operation", None)
        if layer is not None:
            if isinstance(layer, ClusteringLayer) or layer.__class__.__name__ == "ClusteringLayer" or "clustering" in getattr(layer, "name", "").lower():
                clustering_out_idx = idx
                found_clustering = True
                break

    if not found_clustering and len(outputs) == 2:
        output_shapes = model.output_shape if isinstance(model.output_shape, list) else [model.output_shape]
        if len(output_shapes[1]) == 2 and len(output_shapes[0]) > 2:
            clustering_out_idx = 1
            found_clustering = True
        elif len(output_shapes[0]) == 2 and len(output_shapes[1]) > 2:
            clustering_out_idx = 0
            found_clustering = True

    try:
        preds = model.predict(X, verbose=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    if isinstance(preds, list):
        Q = preds[clustering_out_idx]
    else:
        Q = preds

    if Q.ndim != 2:
        raise HTTPException(status_code=400, detail=f"Clustering output must be 2D probabilities tensor, got shape {Q.shape}")

    n_clusters = int(Q.shape[1])
    assigned_clusters = np.argmax(Q, axis=1)
    confidences = np.max(Q, axis=1)
    unique_c, counts_c = np.unique(assigned_clusters, return_counts=True)
    cluster_counts = {int(k): int(v) for k, v in zip(unique_c, counts_c)}

    # Read original IDs from CSV if available
    raw_file_path = resolve_storage_path(dataset_record.get("file_path"))
    input_ids = []
    if raw_file_path and raw_file_path.exists():
        try:
            df_preview = pd.read_csv(raw_file_path)
            for c in ["Input ID", "input_id", "ID", "id", "SampleID", "signal_id"]:
                if c in df_preview.columns:
                    input_ids = df_preview[c].astype(str).tolist()
                    break
        except Exception:
            pass

    if not input_ids or len(input_ids) != len(X):
        input_ids = [f"SIG_{i:04d}" for i in range(len(X))]

    samples = []
    for i in range(min(100, len(X))):
        samples.append({
            "id": input_ids[i],
            "assigned_cluster": int(assigned_clusters[i]),
            "confidence": round(float(confidences[i]), 4),
            "probabilities": [round(float(p), 4) for p in Q[i]]
        })

    return {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "total_samples": int(len(X)),
        "n_clusters": n_clusters,
        "mean_confidence": round(float(np.mean(confidences)), 4),
        "cluster_counts": cluster_counts,
        "samples": samples
    }


@router.get("/{model_id}/idec_probabilities/{dataset_id}/export_csv")
def export_idec_probabilities_csv(model_id: str, dataset_id: str):
    """Exports all IDEC cluster probabilities with Input ID as a downloadable CSV."""
    with get_db_connection() as conn:
        m_row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
        if not m_row:
            raise HTTPException(status_code=404, detail="Model not found")
        model_record = dict(m_row)

    weights_path = resolve_storage_path(model_record.get("weights_path"))
    if not weights_path or not weights_path.exists():
        raise HTTPException(status_code=404, detail="Model weights file not found on disk")

    try:
        model = keras.models.load_model(str(weights_path))
    except Exception:
        graph_json = json.loads(model_record.get("graph_json", "{}"))
        model, _ = KerasModelBuilder.build_model(graph_json)

    try:
        X, y, dataset_record = DataManager.load_dataset_arrays(dataset_id)
        X = np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load dataset: {str(e)}")

    model_input_shape = model.input_shape
    if isinstance(model_input_shape, list):
        model_input_shape = model_input_shape[0]

    if len(model_input_shape) == 3 and X.ndim == 2:
        X = np.expand_dims(X, axis=-1)

    outputs = model.outputs if isinstance(model.outputs, list) else [model.output]
    clustering_out_idx = 0
    found_clustering = False

    for idx, out_tensor in enumerate(outputs):
        layer = None
        if hasattr(out_tensor, "_keras_history"):
            hist = out_tensor._keras_history
            layer = hist[0] if isinstance(hist, tuple) else getattr(hist, "operation", None)
        if layer is not None:
            if isinstance(layer, ClusteringLayer) or layer.__class__.__name__ == "ClusteringLayer" or "clustering" in getattr(layer, "name", "").lower():
                clustering_out_idx = idx
                found_clustering = True
                break

    if not found_clustering and len(outputs) == 2:
        output_shapes = model.output_shape if isinstance(model.output_shape, list) else [model.output_shape]
        if len(output_shapes[1]) == 2 and len(output_shapes[0]) > 2:
            clustering_out_idx = 1
        else:
            clustering_out_idx = 0

    preds = model.predict(X, verbose=0)
    Q = preds[clustering_out_idx] if isinstance(preds, list) else preds

    assigned_clusters = np.argmax(Q, axis=1)
    confidences = np.max(Q, axis=1)

    # Read original IDs
    raw_file_path = resolve_storage_path(dataset_record.get("file_path"))
    input_ids = []
    if raw_file_path and raw_file_path.exists():
        try:
            df_preview = pd.read_csv(raw_file_path)
            for c in ["Input ID", "input_id", "ID", "id", "SampleID", "signal_id"]:
                if c in df_preview.columns:
                    input_ids = df_preview[c].astype(str).tolist()
                    break
        except Exception:
            pass

    if not input_ids or len(input_ids) != len(X):
        input_ids = [f"SIG_{i:04d}" for i in range(len(X))]

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

    filename = f"idec_cluster_probabilities_{model_record['name'].replace(' ', '_')}_{model_id[:8]}.csv"
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

