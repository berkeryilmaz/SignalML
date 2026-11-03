import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from app.core.database import init_database, ensure_default_project, get_db_connection
from app.services.data_manager import DataManager
from app.services.workflow_engine import WorkflowEngine


@pytest.fixture(autouse=True)
def setup_db():
    init_database()
    ensure_default_project()


def test_full_workflow_dag_execution(tmp_path):
    proj = ensure_default_project()
    proj_id = proj["id"]

    # Create test data (10 signals x 100 points)
    X = np.random.randn(10, 100).astype(np.float32)
    csv_path = tmp_path / "wf_data.csv"
    pd.DataFrame(X).to_csv(csv_path, header=False, index=False)
    ds = DataManager.register_dataset(proj_id, "Workflow Raw Signals", csv_path)

    # Build DAG:
    # Data Source (1) -> CSV Parser (2) -> Normalize (3) -> Fixed Length 128 (4) -> Signal Export (5)
    workflow_graph = {
        "nodes": [
            {"id": "1", "type": "data_source", "properties": {"dataset_id": ds["id"]}},
            {"id": "2", "type": "csv_parser", "properties": {"delimiter": ","}},
            {"id": "3", "type": "normalize", "properties": {"method": "zscore", "per_signal": True}},
            {"id": "4", "type": "fixed_length", "properties": {"target_length": 128, "pad_mode": "constant"}},
            {"id": "5", "type": "signal_export", "properties": {"target_length": 128, "output_format": "csv", "filename": "wf_export"}}
        ],
        "links": [
            ["l1", "1", 0, "2", 0],
            ["l2", "2", 0, "3", 0],
            ["l3", "3", 0, "4", 0],
            ["l4", "4", 0, "5", 0]
        ]
    }

    results = WorkflowEngine.execute_workflow(proj_id, workflow_graph)
    assert results["success"] is True
    assert len(results["execution_order"]) == 5
    assert results["execution_order"] == ["1", "2", "3", "4", "5"]

    # Verify each node succeeded
    for nid, node_res in results["node_results"].items():
        assert node_res["status"] == "success"
        assert node_res["duration_ms"] >= 0


def test_save_and_reload_workflow_graph():
    proj = ensure_default_project()
    proj_id = proj["id"]

    wf_id = "test-wf-persistent"
    sample_graph = {
        "nodes": [{"id": "n1", "type": "data_source", "properties": {"dataset_id": "ds-123"}}],
        "links": []
    }

    import json
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO workflows (id, project_id, name, graph_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))
            """,
            (wf_id, proj_id, "Persistent Workflow", json.dumps(sample_graph))
        )
        conn.commit()

    # Reload
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM workflows WHERE id = ?", (wf_id,)).fetchone()
        assert row is not None
        loaded_graph = json.loads(row["graph_json"])
        assert loaded_graph["nodes"][0]["id"] == "n1"
        assert loaded_graph["nodes"][0]["type"] == "data_source"
