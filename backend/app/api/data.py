import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from app.core.config import DATA_DIR, STORAGE_DIR, MAX_UPLOAD_SIZE, ALLOWED_DATA_EXTENSIONS, resolve_storage_path
from app.core.database import get_db_connection
from app.services.data_manager import DataManager
from app.schemas.data import DatasetCreateConfig, DatasetPreview, DataFolderItem, RawPreviewRequest, DatasetUpdateConfig

router = APIRouter(prefix="/api/data", tags=["Data Manager"])


@router.get("/mount", response_model=List[DataFolderItem])
def scan_data_mount(subpath: str = Query(default="")):
    """Safely lists files and folders mounted at /data."""
    try:
        items = DataManager.scan_data_mount(subpath)
        return items
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/upload")
async def upload_file(
    project_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Uploads a data file or ZIP archive.
    If ZIP, extracts safely with zip-slip and zip-bomb protection.
    """
    temp_dir = STORAGE_DIR / "temp" / project_id
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / file.filename

    # Save uploaded file
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    ext = Path(file.filename).suffix.lower()

    if ext == ".zip":
        try:
            extracted_files = DataManager.handle_zip_upload(temp_path, project_id)
            return {
                "type": "zip",
                "filename": file.filename,
                "extracted_count": len(extracted_files),
                "files": extracted_files
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"ZIP extraction error: {str(e)}")
        finally:
            if temp_path.exists():
                os.remove(temp_path)
    else:
        analysis = {}
        if ext in (".csv", ".txt", ".tsv"):
            analysis = DataManager.analyze_file_format(temp_path)

        return {
            "type": "file",
            "filename": file.filename,
            "temp_path": str(temp_path),
            "analysis": analysis
        }


@router.post("/preview_raw", response_model=Dict[str, Any])
def preview_raw_file(payload: RawPreviewRequest):
    """
    Previews a raw data file (CSV, TXT, TSV, JSON) before registration,
    allowing column toggling, pagination, and signal waveform previews.
    """
    src = resolve_storage_path(payload.file_path) or Path(payload.file_path)
    if not src.exists():
        data_src = DATA_DIR / payload.file_path.lstrip("/")
        if data_src.exists():
            src = data_src
        else:
            raise HTTPException(status_code=404, detail=f"Source file not found: {payload.file_path}")

    try:
        return DataManager.preview_raw_file(
            file_path=src,
            delimiter=payload.delimiter,
            has_header=payload.has_header,
            encoding=payload.encoding,
            offset=payload.offset,
            max_rows=payload.max_rows,
            exclude_cols=payload.exclude_cols,
            label_col=payload.label_col,
            transpose=payload.transpose
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error previewing file: {str(e)}")


@router.post("/register")
def register_dataset(
    project_id: str = Form(...),
    name: str = Form(...),
    file_path: str = Form(...),
    delimiter: str = Form(default=","),
    has_header: bool = Form(default=False),
    encoding: str = Form(default="utf-8"),
    label_col: Optional[str] = Form(default=None),
    exclude_cols: Optional[str] = Form(default=None),
    transpose: bool = Form(default=False),
    schema_config: Optional[str] = Form(default=None)
):
    """
    Registers a raw data file (CSV, TXT, TSV, JSON, NPY, NPZ) into the project as a persistent dataset.
    Supports excluding columns (e.g. ID, timestamp), specifying label columns, and multi-modal schema config.
    """
    src = resolve_storage_path(file_path) or Path(file_path)
    if not src.exists():
        # Check in DATA_DIR
        data_src = DATA_DIR / file_path.lstrip("/")
        if data_src.exists():
            src = data_src
        else:
            raise HTTPException(status_code=404, detail=f"Source file not found: {file_path}")

    try:
        res = DataManager.register_dataset(
            project_id=project_id,
            name=name,
            source_path=src,
            delimiter=delimiter,
            has_header=has_header,
            encoding=encoding,
            label_col=label_col if label_col and label_col.strip() else None,
            exclude_cols=exclude_cols if exclude_cols and exclude_cols.strip() else None,
            transpose=transpose,
            schema_config=schema_config if schema_config and schema_config.strip() else None
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error registering dataset: {str(e)}")


@router.put("/{dataset_id}")
def update_dataset(
    dataset_id: str,
    payload: DatasetUpdateConfig
):
    """Renames dataset or updates its schema_config / metadata_json."""
    try:
        return DataManager.update_dataset(
            dataset_id=dataset_id,
            name=payload.name,
            metadata=payload.metadata,
            schema_config=payload.schema_config
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: str):
    """Deletes dataset from project and removes .npz file from disk."""
    try:
        return DataManager.delete_dataset(dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{dataset_id}/sample_ids", response_model=List[str])
def get_dataset_sample_ids(dataset_id: str):
    """Returns sample IDs list for input traceability."""
    try:
        return DataManager.load_sample_ids(dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



@router.get("/preview/{dataset_id}", response_model=Dict[str, Any])
def preview_dataset(
    dataset_id: str,
    offset: int = Query(default=0, ge=0),
    max_rows: int = Query(default=10, ge=1, le=200)
):
    """Returns sample rows, waveform previews, and statistics for a dataset with pagination."""
    try:
        return DataManager.get_preview(dataset_id, offset=offset, max_rows=max_rows)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/list/{project_id}")
def list_project_datasets(project_id: str):
    """Lists all datasets belonging to a project."""
    with get_db_connection() as conn:
        rows = conn.execute("SELECT * FROM datasets WHERE project_id = ? ORDER BY created_at DESC", (project_id,)).fetchall()
        return [dict(r) for r in rows]
