import json
import uuid
import shutil
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from datetime import datetime, timezone

from app.core.config import DATASETS_DIR, MODELS_DIR, CLUSTERS_DIR, FEATURES_DIR, TEMP_DIR
from app.core.database import get_db_connection, ensure_default_project
from app.schemas.project import ProjectCreate, ProjectUpdate, ProjectResponse

router = APIRouter(prefix="/api/projects", tags=["Projects"])


@router.get("", response_model=List[Dict[str, Any]])
def list_projects():
    """Lists all projects with their creation and active IDs."""
    ensure_default_project()
    with get_db_connection() as conn:
        rows = conn.execute("SELECT * FROM projects ORDER BY updated_at DESC").fetchall()
        return [dict(r) for r in rows]


@router.post("", response_model=Dict[str, Any])
def create_project(payload: ProjectCreate):
    """Creates a new project workspace."""
    proj_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO projects (id, name, description, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (proj_id, payload.name, payload.description or "", now, now)
        )
        conn.commit()
    return {"id": proj_id, "name": payload.name, "description": payload.description, "created_at": now}


@router.get("/{project_id}", response_model=Dict[str, Any])
def get_project(project_id: str):
    """Retrieves full project details including linked datasets and models."""
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Project not found")

        project = dict(row)
        datasets = conn.execute("SELECT id, name, row_count, col_count, file_format, created_at FROM datasets WHERE project_id = ?", (project_id,)).fetchall()
        models = conn.execute("SELECT id, name, task_type, trainable_params, created_at FROM models WHERE project_id = ?", (project_id,)).fetchall()
        workflows = conn.execute("SELECT id, name, created_at, updated_at FROM workflows WHERE project_id = ?", (project_id,)).fetchall()
        runs = conn.execute("SELECT id, model_id, dataset_id, status, created_at, finished_at FROM training_runs WHERE project_id = ? ORDER BY created_at DESC LIMIT 5", (project_id,)).fetchall()

        project["datasets"] = [dict(d) for d in datasets]
        project["models"] = [dict(m) for m in models]
        project["workflows"] = [dict(w) for w in workflows]
        project["recent_training_runs"] = [dict(r) for r in runs]
        return project


@router.put("/{project_id}", response_model=Dict[str, Any])
def update_project(project_id: str, payload: ProjectUpdate):
    """Updates active IDs or metadata for a project."""
    now = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Project not found")

        updates = ["updated_at = ?"]
        params = [now]

        if payload.name is not None:
            updates.append("name = ?")
            params.append(payload.name)
        if payload.description is not None:
            updates.append("description = ?")
            params.append(payload.description)
        if payload.active_dataset_id is not None:
            updates.append("active_dataset_id = ?")
            params.append(payload.active_dataset_id)
        if payload.active_model_id is not None:
            updates.append("active_model_id = ?")
            params.append(payload.active_model_id)
        if payload.active_workflow_id is not None:
            updates.append("active_workflow_id = ?")
            params.append(payload.active_workflow_id)

        params.append(project_id)
        conn.execute(f"UPDATE projects SET {', '.join(updates)} WHERE id = ?", params)
        conn.commit()

        updated_row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return dict(updated_row)


@router.delete("/{project_id}")
def delete_project(project_id: str):
    """Deletes a project, its database records, and associated disk files."""
    with get_db_connection() as conn:
        row = conn.execute("SELECT id, name FROM projects WHERE id = ?", (project_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Project not found")

        # Cascade deletes datasets, models, runs, workflows, etc.
        conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        conn.commit()

    # Clean up project directories on disk
    for dir_base in [DATASETS_DIR, MODELS_DIR, CLUSTERS_DIR, FEATURES_DIR, TEMP_DIR]:
        proj_dir = dir_base / project_id
        if proj_dir.exists():
            shutil.rmtree(proj_dir, ignore_errors=True)

    # Ensure at least one default project remains
    ensure_default_project()

    return {"message": f"Project '{row['name']}' deleted successfully", "id": project_id}


@router.post("/{project_id}/duplicate", response_model=Dict[str, Any])
def duplicate_project(project_id: str):
    """Clones a project workspace including its datasets, models, workflows, features, and clusterings."""
    now = datetime.now(timezone.utc).isoformat()
    new_proj_id = str(uuid.uuid4())

    with get_db_connection() as conn:
        orig = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        if not orig:
            raise HTTPException(status_code=404, detail="Project not found")

        orig_dict = dict(orig)
        new_name = f"{orig_dict['name']} (Copy)"

        conn.execute(
            """
            INSERT INTO projects (id, name, description, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (new_proj_id, new_name, orig_dict.get("description") or "", now, now)
        )

        # Copy physical directories first
        for dir_base in [DATASETS_DIR, MODELS_DIR, CLUSTERS_DIR, FEATURES_DIR]:
            src_dir = dir_base / project_id
            dst_dir = dir_base / new_proj_id
            if src_dir.exists():
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

        # Clone datasets
        dataset_id_map = {}
        datasets = conn.execute("SELECT * FROM datasets WHERE project_id = ?", (project_id,)).fetchall()
        for d in datasets:
            dd = dict(d)
            new_ds_id = str(uuid.uuid4())
            dataset_id_map[dd["id"]] = new_ds_id
            old_fp = dd.get("file_path") or ""
            new_fp = old_fp.replace(project_id, new_proj_id).replace(dd["id"], new_ds_id)
            old_p = DATASETS_DIR / new_proj_id / f"{dd['id']}.npz"
            new_p = DATASETS_DIR / new_proj_id / f"{new_ds_id}.npz"
            if old_p.exists():
                shutil.move(str(old_p), str(new_p))
            conn.execute(
                """
                INSERT INTO datasets (
                    id, project_id, name, file_path, file_format, row_count, col_count,
                    columns_json, label_column, sampling_rate, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_ds_id, new_proj_id, dd["name"], str(new_p) if new_p.exists() else new_fp,
                    dd["file_format"], dd["row_count"], dd["col_count"],
                    dd["columns_json"], dd["label_column"], dd["sampling_rate"],
                    dd["metadata_json"], now
                )
            )

        # Clone models
        model_id_map = {}
        models = conn.execute("SELECT * FROM models WHERE project_id = ?", (project_id,)).fetchall()
        for m in models:
            md = dict(m)
            new_m_id = str(uuid.uuid4())
            model_id_map[md["id"]] = new_m_id
            old_wp = md.get("weights_path") or ""
            new_wp = old_wp.replace(project_id, new_proj_id).replace(md["id"], new_m_id)
            old_mp = MODELS_DIR / new_proj_id / f"{md['id']}.keras"
            new_mp = MODELS_DIR / new_proj_id / f"{new_m_id}.keras"
            if old_mp.exists():
                shutil.move(str(old_mp), str(new_mp))
            conn.execute(
                """
                INSERT INTO models (
                    id, project_id, name, task_type, graph_json, summary_json,
                    weights_path, trainable_params, non_trainable_params, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_m_id, new_proj_id, md["name"], md["task_type"],
                    md["graph_json"], md["summary_json"],
                    str(new_mp) if new_mp.exists() else new_wp,
                    md["trainable_params"], md["non_trainable_params"], now, now
                )
            )

        # Clone workflows
        workflows = conn.execute("SELECT * FROM workflows WHERE project_id = ?", (project_id,)).fetchall()
        for w in workflows:
            wd = dict(w)
            new_w_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO workflows (id, project_id, name, graph_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (new_w_id, new_proj_id, wd["name"], wd["graph_json"], now, now)
            )

        conn.commit()

    return {
        "id": new_proj_id,
        "name": new_name,
        "description": orig_dict.get("description") or "",
        "created_at": now
    }

