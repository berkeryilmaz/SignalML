import os
import json
import uuid
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from app.core.config import (
    DATA_DIR,
    STORAGE_DIR,
    DATASETS_DIR,
    ALLOWED_DATA_EXTENSIONS,
    MAX_UPLOAD_SIZE,
    resolve_storage_path
)
from app.core.security import extract_zip_safely, resolve_safe_path, SecurityError
from app.core.database import get_db_connection
from datetime import datetime, timezone


class DataManager:
    @staticmethod
    def detect_delimiter(file_path: Path, sample_bytes: int = 16384) -> str:
        """
        Auto-detects the delimiter among ',', ';', '\t', '|' in a text/CSV file.
        Particularly important for Turkish/European files where semicolon is standard.
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                sample = f.read(sample_bytes)
            lines = [l.strip() for l in sample.splitlines() if l.strip() and not l.strip().startswith("#")]
            if lines:
                l0 = lines[0]
                counts = {
                    ",": l0.count(","),
                    ";": l0.count(";"),
                    "\t": l0.count("\t"),
                    "|": l0.count("|"),
                }
                best = max(counts, key=counts.get)
                if counts[best] > 0:
                    return best
        except Exception:
            pass
        return ","

    @staticmethod
    def detect_has_header(file_path: Path, delimiter: str = ",") -> bool:
        """
        Detects whether row 0 contains text headers while subsequent rows contain numbers.
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")][:6]
            if len(lines) >= 2:
                row0 = lines[0].split(delimiter)
                row1 = lines[1].split(delimiter)
                def is_num(val: str) -> bool:
                    try:
                        float(val.strip().replace(",", "."))
                        return True
                    except ValueError:
                        return False
                r0_nums = sum(1 for v in row0 if is_num(v))
                r1_nums = sum(1 for v in row1 if is_num(v))
                if r1_nums > len(row1) * 0.5 and r0_nums < len(row0) * 0.5:
                    return True
        except Exception:
            pass
        return False

    @staticmethod
    def analyze_file_format(file_path: Path) -> Dict[str, Any]:
        """
        Analyzes a CSV/TXT/TSV file to determine delimiter, headers, and signal orientation.
        """
        delim = DataManager.detect_delimiter(file_path)
        has_hdr = DataManager.detect_has_header(file_path, delimiter=delim)
        try:
            header_param = 0 if has_hdr else None
            df = pd.read_csv(file_path, sep=delim, header=header_param, nrows=20, engine="python")
            total_cols = int(df.shape[1])
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                approx_rows = sum(1 for _ in f) - (1 if has_hdr else 0)
            is_1d = (total_cols == 1 and approx_rows > 10) or (total_cols == 2 and approx_rows > 50)
            return {
                "detected_delimiter": delim,
                "detected_has_header": has_hdr,
                "approx_rows": max(0, approx_rows),
                "col_count": total_cols,
                "is_1d_continuous": is_1d
            }
        except Exception:
            return {
                "detected_delimiter": delim,
                "detected_has_header": has_hdr,
                "approx_rows": 0,
                "col_count": 0,
                "is_1d_continuous": False
            }

    @staticmethod
    def scan_data_mount(subpath: str = "") -> List[Dict[str, Any]]:
        """
        Safely scans the mounted /data folder, returning list of valid files and folders.
        Prevents traversal outside of DATA_DIR.
        """
        target_dir = resolve_safe_path(DATA_DIR, subpath)
        if not target_dir.exists() or not target_dir.is_dir():
            return []

        items = []
        for entry in os.scandir(target_dir):
            try:
                stat = entry.stat()
                is_dir = entry.is_dir()
                ext = Path(entry.name).suffix.lower()

                if is_dir or ext in ALLOWED_DATA_EXTENSIONS or ext == ".zip":
                    rel_path = str(Path(entry.path).relative_to(DATA_DIR))
                    items.append({
                        "name": entry.name,
                        "rel_path": rel_path,
                        "is_dir": is_dir,
                        "size_bytes": stat.st_size if not is_dir else 0,
                        "extension": ext if not is_dir else ""
                    })
            except Exception:
                continue

        # Sort: directories first, then alphabetical
        items.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))
        return items

    @staticmethod
    def handle_zip_upload(zip_file_path: Path, project_id: str) -> List[Dict[str, Any]]:
        """
        Extracts ZIP file safely into a project-specific extraction folder.
        """
        extract_dir = DATASETS_DIR / project_id / "extracted" / zip_file_path.stem
        extracted_info = extract_zip_safely(zip_file_path, extract_dir)
        return extracted_info

    @staticmethod
    def load_raw_data(
        file_path: Path,
        delimiter: str = ",",
        has_header: bool = False,
        encoding: str = "utf-8",
        label_col: Optional[str] = None,
        exclude_cols: Optional[Union[List[str], str]] = None,
        transpose: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        """
        Loads data from CSV, TXT, TSV, JSON, JSONL, NPY, or NPZ into numpy arrays.
        Supports auto-detecting delimiters, comma decimals, excluding non-signal columns,
        extracting label columns, and transposing 1D continuous signals.
        Returns:
            X: 2D numpy array of float32 features/signals (num_samples, seq_len)
            y: 1D numpy array of labels (if label_col is specified and present), else None
            columns: list of feature column names
        """
        ext = file_path.suffix.lower()

        if ext in (".npy", ".npz"):
            if ext == ".npy":
                data = np.load(str(file_path), allow_pickle=False)
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                labels = None
                cols = [f"sample_{i}" for i in range(data.shape[1])]
                return data.astype(np.float32), labels, cols
            else:
                with np.load(str(file_path), allow_pickle=False) as npz:
                    keys = list(npz.keys())
                    data_key = next((k for k in keys if k.lower() in ("data", "x", "signals")), keys[0])
                    data = npz[data_key]
                    if data.ndim == 1:
                        data = data.reshape(1, -1)
                    labels = None
                    if "y" in keys or "labels" in keys:
                        l_key = "y" if "y" in keys else "labels"
                        labels = npz[l_key]
                    cols = [f"sample_{i}" for i in range(data.shape[1])]
                    return data.astype(np.float32), labels, cols

        elif ext in (".json", ".jsonl"):
            if ext == ".jsonl":
                df = pd.read_json(str(file_path), lines=True, encoding=encoding)
            else:
                df = pd.read_json(str(file_path), encoding=encoding)
        else:
            # CSV, TXT, TSV
            if delimiter == "auto" or not delimiter:
                delimiter = DataManager.detect_delimiter(file_path)
            header_param = 0 if has_header else None
            sep = delimiter if delimiter != "\\t" else "\t"
            try:
                df = pd.read_csv(
                    str(file_path),
                    sep=sep,
                    header=header_param,
                    encoding=encoding,
                    engine="c"
                )
            except Exception:
                try:
                    df = pd.read_csv(
                        str(file_path),
                        sep=sep,
                        header=header_param,
                        encoding=encoding,
                        engine="python"
                    )
                except Exception:
                    df = pd.read_csv(
                        str(file_path),
                        sep=None,
                        header=header_param,
                        encoding=encoding,
                        engine="python"
                    )

            # Auto-detect if delimiter produced 1 column when another delimiter exists
            if df.shape[1] == 1 and delimiter == ",":
                alt_delim = DataManager.detect_delimiter(file_path)
                if alt_delim != ",":
                    try:
                        df_alt = pd.read_csv(
                            str(file_path),
                            sep=alt_delim,
                            header=header_param,
                            encoding=encoding,
                            engine="python"
                        )
                        if df_alt.shape[1] > 1:
                            df = df_alt
                    except Exception:
                        pass

        labels = None
        # Handle label column extraction if specified
        if label_col is not None and str(label_col).strip() != "":
            target_col = None
            if label_col in df.columns:
                target_col = label_col
            elif str(label_col).isdigit() and int(label_col) < len(df.columns):
                target_col = df.columns[int(label_col)]

            if target_col is not None:
                labels = df[target_col].to_numpy()
                df = df.drop(columns=[target_col])

        # Handle exclusion of unwanted columns (e.g. ID, index, timestamps)
        if exclude_cols is not None:
            if isinstance(exclude_cols, str):
                try:
                    exclude_list = json.loads(exclude_cols)
                except Exception:
                    exclude_list = [c.strip() for c in exclude_cols.split(",") if c.strip()]
            else:
                exclude_list = list(exclude_cols)

            drop_targets = []
            for ec in exclude_list:
                ec_str = str(ec).strip()
                if ec_str in df.columns:
                    drop_targets.append(ec_str)
                elif ec_str.isdigit() and int(ec_str) < len(df.columns):
                    drop_targets.append(df.columns[int(ec_str)])
            if drop_targets:
                df = df.drop(columns=list(set(drop_targets)))

        # Convert comma decimals to dot decimals in text columns
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = df[col].astype(str).str.replace(",", ".").str.strip()
                except Exception:
                    pass

        # Convert remaining data to numeric
        numeric_df = df.apply(pd.to_numeric, errors="coerce")

        cols = [str(c) for c in df.columns]
        # Auto-detect if row 0 was accidentally read as data instead of column headers
        if len(numeric_df) > 1:
            r0_nan_ratio = float(numeric_df.iloc[0].isna().mean())
            r1_nan_ratio = float(numeric_df.iloc[1].isna().mean())
            if r0_nan_ratio > 0.6 and r1_nan_ratio < 0.3:
                cols = [str(c) for c in df.iloc[0]]
                df = df.iloc[1:].reset_index(drop=True)
                numeric_df = df.apply(pd.to_numeric, errors="coerce")

        data = numeric_df.to_numpy(dtype=np.float32)

        # Transpose if requested (e.g., single continuous 1D signal oriented in a column)
        if transpose or (data.ndim == 2 and data.shape[1] == 1 and data.shape[0] > 10):
            data = data.T
            cols = [f"sample_{i}" for i in range(data.shape[1])]
        elif not cols or len(cols) != (data.shape[1] if data.ndim > 1 else len(data)):
            cols = [str(c) for c in df.columns]

        # Guarantee 2D array
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Ensure all data values are finite floats
        data = np.nan_to_num(data, copy=True, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        return data, labels, cols

    @staticmethod
    def register_dataset(
        project_id: str,
        name: str,
        source_path: Path,
        delimiter: str = ",",
        has_header: bool = False,
        encoding: str = "utf-8",
        label_col: Optional[str] = None,
        exclude_cols: Optional[Union[List[str], str]] = None,
        transpose: bool = False
    ) -> Dict[str, Any]:
        """
        Parses source data, saves it persistently in storage/datasets, computes metadata,
        and saves it in the SQLite database.
        """
        data, labels, cols = DataManager.load_raw_data(
            source_path,
            delimiter=delimiter,
            has_header=has_header,
            encoding=encoding,
            label_col=label_col,
            exclude_cols=exclude_cols,
            transpose=transpose
        )

        dataset_id = str(uuid.uuid4())
        dest_dir = DATASETS_DIR / project_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"{dataset_id}.npz"

        # Save numpy binary for high performance and integrity
        save_dict = {"data": data}
        if labels is not None:
            save_dict["labels"] = labels
        np.savez_compressed(str(dest_path), **save_dict)

        # Compute summary statistics
        num_rows, num_cols = data.shape
        missing_count = int(np.isnan(data).sum())

        finite_data = data[np.isfinite(data)]
        if len(finite_data) > 0:
            summary_stats = {
                "min": float(np.min(finite_data)),
                "max": float(np.max(finite_data)),
                "mean": float(np.mean(finite_data)),
                "std": float(np.std(finite_data)),
            }
        else:
            summary_stats = {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}

        # Parse excluded columns list for metadata
        parsed_excluded = []
        if exclude_cols is not None:
            if isinstance(exclude_cols, str):
                try:
                    parsed_excluded = json.loads(exclude_cols)
                except Exception:
                    parsed_excluded = [c.strip() for c in exclude_cols.split(",") if c.strip()]
            else:
                parsed_excluded = list(exclude_cols)

        metadata = {
            "columns": cols,
            "excluded_columns": parsed_excluded,
            "missing_values": missing_count,
            "has_labels": labels is not None,
            "stats": summary_stats,
            "original_filename": source_path.name,
            "delimiter": delimiter,
            "has_header": has_header
        }

        now = datetime.now(timezone.utc).isoformat()
        with get_db_connection() as conn:
            # Ensure project exists
            proj_row = conn.execute("SELECT id FROM projects WHERE id = ?", (project_id,)).fetchone()
            if not proj_row:
                conn.execute(
                    "INSERT INTO projects (id, name, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (project_id, f"Project {project_id}", "Auto-created project", now, now)
                )

            conn.execute(
                """
                INSERT INTO datasets (
                    id, project_id, name, file_path, file_format,
                    row_count, col_count, has_header, label_col,
                    created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_id, project_id, name, str(dest_path), source_path.suffix.lstrip("."),
                    num_rows, num_cols, 1 if has_header else 0, label_col,
                    now, json.dumps(metadata)
                )
            )
            # Set as active dataset for project
            conn.execute(
                "UPDATE projects SET active_dataset_id = ?, updated_at = ? WHERE id = ?",
                (dataset_id, now, project_id)
            )
            conn.commit()

        return {
            "id": dataset_id,
            "name": name,
            "row_count": num_rows,
            "col_count": num_cols,
            "has_header": has_header,
            "label_col": label_col,
            "metadata": metadata
        }

    @staticmethod
    def get_dataset(dataset_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves dataset record by ID."""
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
            if row:
                d = dict(row)
                d["metadata"] = json.loads(d.get("metadata_json", "{}"))
                return d
        return None

    @staticmethod
    def load_dataset_arrays(dataset_id: str) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """
        Loads the NumPy array and optional labels for a registered dataset.
        Resolves storage path dynamically across host and Docker environments.
        """
        record = DataManager.get_dataset(dataset_id)
        if not record:
            raise ValueError(f"Dataset {dataset_id} not found")

        file_path = resolve_storage_path(record["file_path"])
        if not file_path or not file_path.exists():
            # Check fallback in DATASETS_DIR / project_id / dataset_id.npz
            cand = DATASETS_DIR / record.get("project_id", "") / f"{dataset_id}.npz"
            if cand.exists():
                file_path = cand
            else:
                raise FileNotFoundError(f"Dataset file missing on disk: {record['file_path']}")

        with np.load(str(file_path), allow_pickle=False) as npz:
            data = npz["data"]
            labels = npz["labels"] if "labels" in npz else None

        # Ensure finite data
        if not np.all(np.isfinite(data)):
            data = np.nan_to_num(data, copy=True, nan=0.0, posinf=0.0, neginf=0.0)

        return data, labels, record

    @staticmethod
    def get_preview(
        dataset_id: str,
        offset: int = 0,
        max_rows: int = 10,
        exclude_cols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Returns preview rows and summary info with pagination, column exclusion, and sample waveforms."""
        data, labels, record = DataManager.load_dataset_arrays(dataset_id)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim == 0 or data.size == 0:
            data = np.zeros((1, 1), dtype=np.float32)

        metadata = record.get("metadata", {})
        cols = metadata.get("columns", [f"sample_{i}" for i in range(data.shape[1])])

        total_rows = int(data.shape[0])
        total_cols = int(data.shape[1])
        offset = max(0, min(offset, max(0, total_rows - 1)))
        
        raw_slice = data[offset : offset + max_rows]
        preview_data = []
        for row in raw_slice:
            clean_row = [float(v) if np.isfinite(v) else None for v in row]
            preview_data.append(clean_row)

        preview_labels = labels[offset : offset + max_rows].tolist() if labels is not None else None

        # Build sample waveforms for fast inline visualization
        waveforms = []
        for i, row in enumerate(raw_slice):
            row_idx = offset + i
            finite_row = row[np.isfinite(row)]
            seq_len = len(row)
            step = max(1, int(np.ceil(seq_len / 500)))
            waveforms.append({
                "row_index": row_idx,
                "label": preview_labels[i] if preview_labels else None,
                "x": list(range(0, seq_len, step)),
                "y": [float(v) if np.isfinite(v) else 0.0 for v in row[::step]],
                "min": float(np.min(finite_row)) if len(finite_row) > 0 and np.isfinite(np.min(finite_row)) else 0.0,
                "max": float(np.max(finite_row)) if len(finite_row) > 0 and np.isfinite(np.max(finite_row)) else 0.0,
                "mean": float(np.mean(finite_row)) if len(finite_row) > 0 and np.isfinite(np.mean(finite_row)) else 0.0,
                "std": float(np.std(finite_row)) if len(finite_row) > 0 and np.isfinite(np.std(finite_row)) else 0.0
            })

        finite_all = data[np.isfinite(data)]
        stats = {
            "min": float(np.min(finite_all)) if len(finite_all) > 0 and np.isfinite(np.min(finite_all)) else 0.0,
            "max": float(np.max(finite_all)) if len(finite_all) > 0 and np.isfinite(np.max(finite_all)) else 0.0,
            "mean": float(np.mean(finite_all)) if len(finite_all) > 0 and np.isfinite(np.mean(finite_all)) else 0.0,
            "std": float(np.std(finite_all)) if len(finite_all) > 0 and np.isfinite(np.std(finite_all)) else 0.0,
        }

        return {
            "dataset_id": dataset_id,
            "name": record["name"],
            "row_count": total_rows,
            "col_count": total_cols,
            "offset": offset,
            "max_rows": max_rows,
            "columns": cols,
            "excluded_columns": metadata.get("excluded_columns", []),
            "preview_rows": preview_data,
            "preview_labels": preview_labels,
            "summary_stats": stats,
            "missing_count": metadata.get("missing_values", 0),
            "has_header": bool(record.get("has_header", 0)),
            "label_col": record.get("label_col"),
            "sample_waveforms": waveforms
        }

    @staticmethod
    def preview_raw_file(
        file_path: Union[str, Path],
        delimiter: str = ",",
        has_header: bool = False,
        encoding: str = "utf-8",
        offset: int = 0,
        max_rows: int = 10,
        exclude_cols: Optional[Union[List[str], str]] = None,
        label_col: Optional[str] = None,
        transpose: bool = False
    ) -> Dict[str, Any]:
        """
        Previews a raw data file (CSV, TXT, TSV, JSON) before registration,
        allowing column toggling, pagination, signal waveform previews, and transposing.
        """
        p = Path(file_path)
        if not p.exists():
            # Check under DATA_DIR
            alt = DATA_DIR / str(file_path).lstrip("/")
            if alt.exists():
                p = alt
            else:
                raise FileNotFoundError(f"File not found: {file_path}")

        ext = p.suffix.lower()
        if ext in (".npy", ".npz"):
            data, labels, cols = DataManager.load_raw_data(p, delimiter=delimiter, has_header=has_header, encoding=encoding, label_col=label_col, exclude_cols=exclude_cols, transpose=transpose)
            total_rows = data.shape[0]
            total_cols = data.shape[1]
            offset = max(0, min(offset, max(0, total_rows - 1)))
            raw_slice = data[offset : offset + max_rows]
            slice_data = []
            for r in raw_slice:
                slice_data.append([float(v) if np.isfinite(v) else None for v in r])

            finite_all = data[np.isfinite(data)]
            return {
                "file_name": p.name,
                "total_rows": total_rows,
                "total_cols": total_cols,
                "columns": cols,
                "all_columns": cols,
                "active_columns": cols,
                "excluded_columns": [],
                "suggested_id_cols": [],
                "offset": offset,
                "max_rows": max_rows,
                "preview_rows": slice_data,
                "summary_stats": {
                    "min": float(np.min(finite_all)) if len(finite_all) > 0 else 0.0,
                    "max": float(np.max(finite_all)) if len(finite_all) > 0 else 0.0,
                    "mean": float(np.mean(finite_all)) if len(finite_all) > 0 else 0.0,
                    "std": float(np.std(finite_all)) if len(finite_all) > 0 else 0.0
                },
                "missing_count": int(np.isnan(data).sum()),
                "sample_waveforms": [
                    {
                        "row_index": offset + i,
                        "x": list(range(len(r[:500]))),
                        "y": [float(v) if np.isfinite(v) else 0.0 for v in r[:500]],
                        "min": float(np.min(r[np.isfinite(r)])) if len(r[np.isfinite(r)]) > 0 else 0.0,
                        "max": float(np.max(r[np.isfinite(r)])) if len(r[np.isfinite(r)]) > 0 else 0.0,
                        "mean": float(np.mean(r[np.isfinite(r)])) if len(r[np.isfinite(r)]) > 0 else 0.0,
                        "std": float(np.std(r[np.isfinite(r)])) if len(r[np.isfinite(r)]) > 0 else 0.0
                    } for i, r in enumerate(raw_slice)
                ]
            }

        # CSV / TSV / TXT / JSON
        if delimiter == "auto" or not delimiter:
            delimiter = DataManager.detect_delimiter(p)
        header_param = 0 if has_header else None
        sep = delimiter if delimiter != "\\t" else "\t"
        try:
            df = pd.read_csv(str(p), sep=sep, header=header_param, encoding=encoding, engine="c")
        except Exception:
            try:
                df = pd.read_csv(str(p), sep=sep, header=header_param, encoding=encoding, engine="python")
            except Exception:
                df = pd.read_csv(str(p), sep=None, header=header_param, encoding=encoding, engine="python")

        # Auto-detect if delimiter produced 1 column when another delimiter exists
        if df.shape[1] == 1 and delimiter == ",":
            alt_delim = DataManager.detect_delimiter(p)
            if alt_delim != ",":
                try:
                    df_alt = pd.read_csv(str(p), sep=alt_delim, header=header_param, encoding=encoding, engine="python")
                    if df_alt.shape[1] > 1:
                        df = df_alt
                except Exception:
                    pass

        # Handle comma decimals in text columns
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = df[col].astype(str).str.replace(",", ".").str.strip()
                except Exception:
                    pass

        all_columns = [str(c) for c in df.columns]

        # Auto-detect ID or index columns
        suggested_id_cols = []
        for c in all_columns:
            c_low = str(c).lower().strip()
            if c_low in ("id", "_id", "index", "idx", "uid", "sample_id", "patient_id", "record_id", "signal_id"):
                suggested_id_cols.append(c)

        # Parse excluded columns
        parsed_exclude = []
        if exclude_cols:
            if isinstance(exclude_cols, str):
                try:
                    parsed_exclude = json.loads(exclude_cols)
                except Exception:
                    parsed_exclude = [c.strip() for c in exclude_cols.split(",") if c.strip()]
            else:
                parsed_exclude = [str(c).strip() for c in exclude_cols]

        # Extract label column if specified
        label_values = None
        if label_col is not None and str(label_col).strip() != "":
            target_col = None
            if label_col in df.columns:
                target_col = label_col
            elif str(label_col).isdigit() and int(label_col) < len(df.columns):
                target_col = df.columns[int(label_col)]
            if target_col is not None:
                label_values = df[target_col].tolist()
                df = df.drop(columns=[target_col])

        # Exclude selected columns
        cols_to_drop = [c for c in parsed_exclude if c in df.columns]
        if cols_to_drop:
            active_df = df.drop(columns=cols_to_drop)
        else:
            active_df = df

        # Transpose if requested or if 1D continuous column
        if transpose or (active_df.shape[1] == 1 and len(active_df) > 10):
            active_df = active_df.T
            active_columns = [f"sample_{i}" for i in range(active_df.shape[1])]
            all_columns = active_columns
        else:
            active_columns = [str(c) for c in active_df.columns]

        total_rows = len(active_df)
        offset = max(0, min(offset, max(0, total_rows - 1)))
        
        # Paginated rows
        df_slice = active_df.iloc[offset : offset + max_rows]
        numeric_slice = df_slice.apply(pd.to_numeric, errors="coerce")
        slice_matrix = numeric_slice.to_numpy(dtype=np.float32)

        # Stats on numeric data
        finite_vals = slice_matrix[np.isfinite(slice_matrix)]
        if len(finite_vals) > 0:
            stats = {
                "min": float(np.min(finite_vals)),
                "max": float(np.max(finite_vals)),
                "mean": float(np.mean(finite_vals)),
                "std": float(np.std(finite_vals))
            }
        else:
            stats = {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}

        # Build sample waveforms
        waveforms = []
        for i in range(len(df_slice)):
            row_idx = offset + i
            row_arr = slice_matrix[i]
            finite_row = row_arr[np.isfinite(row_arr)]
            seq_len = len(row_arr)
            step = max(1, int(np.ceil(seq_len / 500)))
            waveforms.append({
                "row_index": row_idx,
                "label": label_values[row_idx] if label_values and row_idx < len(label_values) else None,
                "x": list(range(0, seq_len, step)),
                "y": [float(v) if np.isfinite(v) else 0.0 for v in row_arr[::step]],
                "min": float(np.min(finite_row)) if len(finite_row) > 0 else 0.0,
                "max": float(np.max(finite_row)) if len(finite_row) > 0 else 0.0,
                "mean": float(np.mean(finite_row)) if len(finite_row) > 0 else 0.0,
                "std": float(np.std(finite_row)) if len(finite_row) > 0 else 0.0,
            })

        # Format preview rows values strictly JSON-compliant
        preview_rows = []
        for _, row in df_slice.iterrows():
            formatted_row = []
            for v in row.values:
                if pd.isna(v) or v is None:
                    formatted_row.append(None)
                elif isinstance(v, (np.floating, float)):
                    formatted_row.append(float(v) if np.isfinite(v) else None)
                elif isinstance(v, (np.integer, int)):
                    formatted_row.append(int(v))
                else:
                    formatted_row.append(str(v))
            preview_rows.append(formatted_row)

        return {
            "file_name": p.name,
            "total_rows": total_rows,
            "total_cols": len(active_columns),
            "all_columns": all_columns,
            "active_columns": active_columns,
            "excluded_columns": parsed_exclude,
            "suggested_id_cols": suggested_id_cols,
            "offset": offset,
            "max_rows": max_rows,
            "preview_rows": preview_rows,
            "summary_stats": stats,
            "missing_count": int(numeric_slice.isna().sum().sum()),
            "sample_waveforms": waveforms
        }
