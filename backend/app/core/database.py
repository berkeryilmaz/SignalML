import sqlite3
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.core.config import SQLITE_DB_PATH


def get_db_connection() -> sqlite3.Connection:
    """Returns a SQLite connection with Row factory enabled."""
    conn = sqlite3.connect(str(SQLITE_DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_database() -> None:
    """Initializes the database schema if not present."""
    with get_db_connection() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            active_dataset_id TEXT,
            active_model_id TEXT,
            active_workflow_id TEXT,
            metadata_json TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS datasets (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_format TEXT NOT NULL,
            row_count INTEGER DEFAULT 0,
            col_count INTEGER DEFAULT 0,
            has_header BOOLEAN DEFAULT 0,
            label_col TEXT,
            created_at TEXT NOT NULL,
            metadata_json TEXT DEFAULT '{}',
            FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            name TEXT NOT NULL,
            task_type TEXT DEFAULT 'classification',
            graph_json TEXT NOT NULL,
            summary_json TEXT DEFAULT '{}',
            weights_path TEXT,
            trainable_params INTEGER DEFAULT 0,
            non_trainable_params INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS training_runs (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            dataset_id TEXT NOT NULL,
            status TEXT NOT NULL,
            config_json TEXT NOT NULL,
            history_json TEXT DEFAULT '{}',
            evaluation_json TEXT DEFAULT '{}',
            error_message TEXT,
            created_at TEXT NOT NULL,
            finished_at TEXT,
            FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE,
            FOREIGN KEY(model_id) REFERENCES models(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS workflows (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            name TEXT NOT NULL,
            graph_json TEXT NOT NULL,
            last_run_results_json TEXT DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS features (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            dataset_id TEXT NOT NULL,
            layer_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            shape_json TEXT NOT NULL,
            sample_count INTEGER DEFAULT 0,
            feature_dim INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS clusterings (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_id TEXT NOT NULL,
            algorithm TEXT NOT NULL,
            params_json TEXT NOT NULL,
            result_path TEXT NOT NULL,
            pca_path TEXT,
            metrics_json TEXT DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS exports (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            export_type TEXT NOT NULL,
            name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            size_bytes INTEGER DEFAULT 0,
            metadata_json TEXT DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
        );
        """)
        # Normalize any /workspace/SignalML/storage paths in DB to /app/storage so host Docker
        # container always sees native /app/storage paths while resolve_storage_path() maps locally.
        for tbl, col in [
            ("datasets", "file_path"),
            ("models", "weights_path"),
            ("features", "file_path"),
            ("clusterings", "result_path"),
            ("clusterings", "pca_path"),
            ("exports", "file_path")
        ]:
            try:
                conn.execute(f"""
                    UPDATE {tbl}
                    SET {col} = REPLACE({col}, '/workspace/SignalML/storage', '/app/storage')
                    WHERE {col} LIKE '/workspace/SignalML/storage%'
                """)
            except Exception:
                pass
        conn.commit()


# Helper CRUD methods for projects
def ensure_default_project() -> Dict[str, Any]:
    """Ensures at least one default project exists and returns it."""
    init_database()
    with get_db_connection() as conn:
        cursor = conn.execute("SELECT * FROM projects ORDER BY created_at ASC LIMIT 1")
        row = cursor.fetchone()
        if row:
            return dict(row)
        
        now = datetime.now(timezone.utc).isoformat()
        proj_id = "default-project"
        conn.execute(
            """
            INSERT INTO projects (id, name, description, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (proj_id, "Default Signal Project", "Auto-generated default project workspace", now, now)
        )
        conn.commit()
        return {
            "id": proj_id,
            "name": "Default Signal Project",
            "description": "Auto-generated default project workspace",
            "created_at": now,
            "updated_at": now,
            "active_dataset_id": None,
            "active_model_id": None,
            "active_workflow_id": None,
            "metadata_json": "{}"
        }
