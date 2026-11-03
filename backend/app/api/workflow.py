import json
import uuid
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Body
from datetime import datetime, timezone

from app.core.database import get_db_connection
from app.services.node_engine import NodeEngine, NodeValidationError
from app.services.workflow_engine import WorkflowEngine
from app.schemas.node import NodeGraph, GraphValidationResult

router = APIRouter(prefix="/api/workflow", tags=["Visual Workflow Engine"])


@router.get("/catalog")
def get_node_catalog():
    """Returns catalog of all supported workflow node types, typed ports, and parameters."""
    return NodeEngine.get_node_catalog()


@router.post("/validate", response_model=GraphValidationResult)
def validate_workflow(payload: Dict[str, Any] = Body(...)):
    """
    Validates general workflow graph: detects cycles and verifies typed port connections.
    """
    graph_data = payload.get("graph", payload)
    return NodeEngine.validate_graph(graph_data)


@router.post("/execute")
def execute_workflow(payload: Dict[str, Any] = Body(...)):
    """
    Executes a general-purpose workflow DAG step by step.
    Returns status and previews for every node.
    """
    project_id = payload.get("project_id")
    graph_data = payload.get("graph", payload)

    if not project_id:
        raise HTTPException(status_code=400, detail="project_id is required")

    try:
        results = WorkflowEngine.execute_workflow(project_id, graph_data)
        
        # Save workflow run to database if name provided
        workflow_name = payload.get("name", "Workflow Run")
        wf_id = payload.get("workflow_id", str(uuid.uuid4()))
        now = datetime.now(timezone.utc).isoformat()
        
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO workflows (id, project_id, name, graph_json, last_run_results_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    last_run_results_json = excluded.last_run_results_json,
                    graph_json = excluded.graph_json,
                    updated_at = excluded.updated_at
                """,
                (wf_id, project_id, workflow_name, json.dumps(graph_data), json.dumps(results), now, now)
            )
            conn.commit()

        results["workflow_id"] = wf_id
        return results
    except NodeValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")


@router.get("/list/{project_id}")
def list_workflows(project_id: str):
    """Lists saved workflows for a project."""
    with get_db_connection() as conn:
        rows = conn.execute("SELECT id, name, created_at, updated_at FROM workflows WHERE project_id = ? ORDER BY updated_at DESC", (project_id,)).fetchall()
        return [dict(r) for r in rows]


@router.get("/{workflow_id}")
def get_workflow(workflow_id: str):
    """Retrieves full workflow graph JSON and last run output."""
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM workflows WHERE id = ?", (workflow_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Workflow not found")
        d = dict(row)
        d["graph"] = json.loads(d.get("graph_json", "{}"))
        d["last_run_results"] = json.loads(d.get("last_run_results_json", "{}"))
        return d
