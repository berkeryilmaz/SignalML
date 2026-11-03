from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class DatasetCreateConfig(BaseModel):
    name: str = Field(..., min_length=1)
    delimiter: str = Field(default=",")
    has_header: bool = Field(default=False)
    encoding: str = Field(default="utf-8")
    label_col: Optional[str] = None  # None, or column name/index
    numeric_cols: Optional[List[str]] = None
    exclude_cols: Optional[List[str]] = None


class DatasetPreview(BaseModel):
    dataset_id: str
    name: str
    row_count: int
    col_count: int
    columns: List[str]
    preview_rows: List[List[Any]]
    summary_stats: Dict[str, Any]
    missing_count: int
    has_header: bool
    label_col: Optional[str] = None
    offset: int = 0
    max_rows: int = 10
    preview_labels: Optional[List[Any]] = None
    excluded_columns: Optional[List[str]] = None
    sample_waveforms: Optional[List[Dict[str, Any]]] = None


class RawPreviewRequest(BaseModel):
    file_path: str
    delimiter: str = ","
    has_header: bool = False
    encoding: str = "utf-8"
    offset: int = 0
    max_rows: int = 10
    exclude_cols: Optional[List[str]] = None
    label_col: Optional[str] = None
    transpose: bool = False


class DataFolderItem(BaseModel):
    name: str
    rel_path: str
    is_dir: bool
    size_bytes: int
    extension: str
