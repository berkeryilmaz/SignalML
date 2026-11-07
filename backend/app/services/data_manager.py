import os
import json
import uuid
import shutil
import re
import fnmatch
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
    def resolve_sequence_columns(all_columns: List[str], pattern: str) -> List[str]:
        """
        Resolves column names from pattern.
        Supports:
        - Comma-separated lists: e.g. "s1, s2, s3"
        - Wildcard patterns: e.g. "s*", "sig_*"
        - Range patterns: e.g. "s1-s100", "sample_0-sample_511", "sig1-sig10"
        - Index ranges: e.g. "0-511" (matches columns named '0'..'511' or 0-based indices in all_columns)
        - Regexes: e.g. "^s\\d+$"
        - Direct column names
        Maintains sequential order and returns only columns that match or exist.
        """
        if not pattern or not isinstance(pattern, str):
            return []
        p = pattern.strip()
        if not p:
            return []

        # 1. Comma-separated list
        if "," in p:
            parts = [part.strip() for part in p.split(",") if part.strip()]
            results: List[str] = []
            for part in parts:
                matched = DataManager.resolve_sequence_columns(all_columns, part)
                for m in matched:
                    if m not in results:
                        results.append(m)
            return results

        # 2. Range patterns like s1-s100, sample_0-sample_511, or index range 0-511
        range_match = re.match(r"^([a-zA-Z_]*?)(\d+)\s*-\s*([a-zA-Z_]*?)(\d+)$", p)
        if range_match:
            pfx1, start_str, pfx2, end_str = range_match.groups()
            start_num = int(start_str)
            end_num = int(end_str)
            if start_num > end_num:
                start_num, end_num = end_num, start_num

            # Case: no prefix (e.g. "0-511")
            if not pfx1 and (not pfx2 or pfx2 == pfx1):
                name_set = set(str(c) for c in all_columns)
                num_named = [str(i) for i in range(start_num, end_num + 1) if str(i) in name_set]
                if len(num_named) == (end_num - start_num + 1):
                    return num_named
                if 0 <= start_num < len(all_columns):
                    idx_end = min(len(all_columns), end_num + 1)
                    return [str(c) for c in all_columns[start_num:idx_end]]
                if num_named:
                    return num_named

            # Case: prefix specified (e.g. "s1-s100" or "sample_0-sample_511")
            prefix = pfx1 if (not pfx2 or pfx2 == pfx1) else pfx1
            col_rx = re.compile(rf"^{re.escape(prefix)}(\d+)$")
            matched_items = []
            for col in all_columns:
                m = col_rx.match(str(col))
                if m:
                    num = int(m.group(1))
                    if start_num <= num <= end_num:
                        matched_items.append((num, str(col)))
            if matched_items:
                matched_items.sort(key=lambda x: x[0])
                return [c for _, c in matched_items]

            # Fallback for exact generated names
            generated = [f"{prefix}{i}" for i in range(start_num, end_num + 1)]
            found = [c for c in generated if c in all_columns]
            if found:
                return found

        # 3. Regex pattern (starts with ^, ends with $, contains \\d or regex operators)
        if p.startswith("^") or p.endswith("$") or "\\d" in p or (any(c in p for c in ("+", "[", "]", "(", ")")) and not any(c in p for c in ("*", "?"))):
            try:
                rx = re.compile(p)
                return [str(col) for col in all_columns if rx.search(str(col))]
            except re.error:
                pass

        # 4. Wildcard pattern (e.g. "s*", "sig_*")
        if "*" in p or "?" in p:
            matched = [str(col) for col in all_columns if fnmatch.fnmatchcase(str(col), p)]
            if matched:
                return matched
            return [str(col) for col in all_columns if fnmatch.fnmatch(str(col).lower(), p.lower())]

        # 5. Exact match in all_columns
        if p in all_columns:
            return [p]
        for col in all_columns:
            if str(col).lower() == p.lower():
                return [str(col)]

        return []

    @staticmethod
    def detect_sequence_groups(all_columns: List[str]) -> List[Dict[str, Any]]:
        """
        Auto-detects contiguous sequential columns in all_columns
        (e.g. s1-s100, sample_0-sample_511, 0-511).
        Returns a list of suggested sequence group dicts.
        """
        if not all_columns or len(all_columns) < 3:
            return []

        groups: List[Dict[str, Any]] = []
        rx = re.compile(r"^([a-zA-Z_]*)(\d+)$")

        i = 0
        n = len(all_columns)
        while i < n:
            m = rx.match(str(all_columns[i]))
            if not m:
                i += 1
                continue

            prefix = m.group(1)
            first_num = int(m.group(2))
            prev_num = first_num
            col_run = [str(all_columns[i])]
            j = i + 1

            while j < n:
                m_next = rx.match(str(all_columns[j]))
                if not m_next:
                    break
                pfx_next = m_next.group(1)
                num_next = int(m_next.group(2))
                if pfx_next == prefix and num_next == prev_num + 1:
                    col_run.append(str(all_columns[j]))
                    prev_num = num_next
                    j += 1
                else:
                    break

            if len(col_run) >= 3:
                last_num = prev_num
                if prefix:
                    pattern = f"{prefix}{first_num}-{prefix}{last_num}"
                    name = f"sequence_{prefix.strip('_') or 'group'}"
                else:
                    pattern = f"{first_num}-{last_num}"
                    name = "sequence_indexed"

                groups.append({
                    "name": name,
                    "prefix": prefix,
                    "pattern": pattern,
                    "columns": col_run,
                    "count": len(col_run)
                })
                i = j
            else:
                i += 1

        return groups


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
        transpose: bool = False,
        schema_config: Optional[Union[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Parses source data, saves it persistently in storage/datasets, computes metadata,
        performs automated preprocessing and reshaping if schema_config is provided,
        tracks Input ID traceability, and saves it in the SQLite database.
        """
        parsed_schema_config = None
        if schema_config is not None:
            if isinstance(schema_config, str) and schema_config.strip():
                try:
                    parsed_schema_config = json.loads(schema_config)
                except Exception:
                    parsed_schema_config = None
            elif isinstance(schema_config, dict):
                parsed_schema_config = schema_config

        dataset_id = str(uuid.uuid4())
        dest_dir = DATASETS_DIR / project_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"{dataset_id}.npz"
        now = datetime.now(timezone.utc).isoformat()

        if parsed_schema_config is None:
            # Backward-compatible loading
            data, labels, cols = DataManager.load_raw_data(
                source_path,
                delimiter=delimiter,
                has_header=has_header,
                encoding=encoding,
                label_col=label_col,
                exclude_cols=exclude_cols,
                transpose=transpose
            )

            num_rows, num_cols = data.shape
            # Input ID Traceability: default sample_ids
            sample_ids = np.array([str(i) for i in range(num_rows)], dtype=str)

            save_dict = {
                "data": data,
                "sample_ids": sample_ids
            }
            if labels is not None:
                save_dict["labels"] = labels
            np.savez_compressed(str(dest_path), **save_dict)

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
                "has_header": has_header,
                "sample_ids_count": len(sample_ids)
            }

        else:
            # Multi-Modal Automated Preprocessing pipeline
            ext = source_path.suffix.lower()
            if ext in (".npy", ".npz"):
                data, labels, cols = DataManager.load_raw_data(
                    source_path,
                    delimiter=delimiter,
                    has_header=has_header,
                    encoding=encoding,
                    label_col=label_col,
                    exclude_cols=exclude_cols,
                    transpose=transpose
                )
                num_rows, num_cols = data.shape
                sample_ids = np.array([str(i) for i in range(num_rows)], dtype=str)
                X_seq = np.expand_dims(data, axis=-1).astype(np.float32)
                X_tabular = None
                y = labels
                task_type = "classification" if labels is not None else "unsupervised"
                recommended_activation = "softmax" if labels is not None else "linear"
                recommended_loss = "sparse_categorical_crossentropy" if labels is not None else "mse"
                recommended_metrics = ["accuracy"] if labels is not None else []
                pipeline_state = {
                    "sequence_normalization": "none",
                    "seq_len": int(num_cols),
                    "task_type": task_type,
                    "recommended_activation": recommended_activation,
                    "recommended_loss": recommended_loss,
                    "recommended_metrics": recommended_metrics
                }
                save_dict = {
                    "data": data,
                    "X_seq": X_seq,
                    "sample_ids": sample_ids
                }
                if labels is not None:
                    save_dict["labels"] = labels
                    save_dict["y"] = labels
                np.savez_compressed(str(dest_path), **save_dict)
            else:
                # Tabular files (CSV, TXT, TSV, JSON)
                if delimiter == "auto" or not delimiter:
                    delimiter = DataManager.detect_delimiter(source_path)
                sep = delimiter if delimiter != "\\t" else "\t"
                header_param = 0 if has_header else None
                try:
                    df = pd.read_csv(str(source_path), sep=sep, header=header_param, encoding=encoding, engine="c")
                except Exception:
                    try:
                        df = pd.read_csv(str(source_path), sep=sep, header=header_param, encoding=encoding, engine="python")
                    except Exception:
                        df = pd.read_csv(str(source_path), sep=None, header=header_param, encoding=encoding, engine="python")

                if df.shape[1] == 1 and delimiter == ",":
                    alt_delim = DataManager.detect_delimiter(source_path)
                    if alt_delim != ",":
                        try:
                            df_alt = pd.read_csv(str(source_path), sep=alt_delim, header=header_param, encoding=encoding, engine="python")
                            if df_alt.shape[1] > 1:
                                df = df_alt
                        except Exception:
                            pass

                if not has_header:
                    df.columns = [f"sample_{i}" for i in range(df.shape[1])]
                else:
                    df.columns = [str(c) for c in df.columns]

                if transpose:
                    df = df.T
                    df.columns = [f"sample_{i}" for i in range(df.shape[1])]
                    df = df.reset_index(drop=True)

                num_rows = len(df)
                raw_columns = list(df.columns)

                col_configs = parsed_schema_config.get("columns", {})
                if isinstance(col_configs, list):
                    col_configs = {c["name"]: c for c in col_configs if "name" in c}

                seq_groups = parsed_schema_config.get("sequence_groups", [])
                default_scaler = parsed_schema_config.get("default_numeric_scaler", "standard")
                default_encoder = parsed_schema_config.get("default_categorical_encoder", "onehot")
                seq_norm_method = parsed_schema_config.get("sequence_normalization", "row_zscore")

                pipeline_state: Dict[str, Any] = {
                    "numeric_stats": {},
                    "categorical_vocabs": {},
                    "sequence_normalization": seq_norm_method
                }

                # 1. ID Column Extraction (Never feed ID into training features!)
                id_col_name = None
                for col_name, c_cfg in col_configs.items():
                    if c_cfg.get("role") == "id" and col_name in df.columns:
                        id_col_name = col_name
                        break

                if id_col_name is None:
                    for col in df.columns:
                        c_low = str(col).lower().strip()
                        if c_low in ("id", "_id", "uid", "sample_id", "patient_id", "record_id", "signal_id"):
                            id_col_name = col
                            break

                if id_col_name is not None and id_col_name in df.columns:
                    sample_ids = np.array([str(v) for v in df[id_col_name].tolist()], dtype=str)
                    df = df.drop(columns=[id_col_name])
                else:
                    sample_ids = np.array([str(i) for i in range(num_rows)], dtype=str)

                # 2. Output Target Extraction
                out_col_name = None
                for col_name, c_cfg in col_configs.items():
                    if c_cfg.get("role") == "output" and col_name in df.columns:
                        out_col_name = col_name
                        break

                if out_col_name is None and label_col:
                    if str(label_col) in df.columns:
                        out_col_name = str(label_col)
                    elif str(label_col).isdigit() and int(label_col) < len(df.columns):
                        out_col_name = df.columns[int(label_col)]

                out_series = None
                out_cfg = {}
                if out_col_name is not None and out_col_name in df.columns:
                    out_cfg = col_configs.get(out_col_name, {})
                    out_series = df[out_col_name]
                    df = df.drop(columns=[out_col_name])

                # 3. Excluded / Ignored Columns
                parsed_exclude = []
                if exclude_cols:
                    if isinstance(exclude_cols, str):
                        try:
                            parsed_exclude = json.loads(exclude_cols)
                        except Exception:
                            parsed_exclude = [c.strip() for c in exclude_cols.split(",") if c.strip()]
                    else:
                        parsed_exclude = list(exclude_cols)

                ignore_cols = set(str(c) for c in parsed_exclude)
                for col_name, c_cfg in col_configs.items():
                    if c_cfg.get("role") == "ignore":
                        ignore_cols.add(str(col_name))

                cols_to_drop = [c for c in ignore_cols if c in df.columns]
                if cols_to_drop:
                    df = df.drop(columns=cols_to_drop)

                # 4. Sequential / 1D-Signal Inputs
                seq_cols_ordered: List[str] = []
                for sg in seq_groups:
                    sg_cols = sg.get("columns", [])
                    if not sg_cols and sg.get("pattern"):
                        sg_cols = DataManager.resolve_sequence_columns(list(df.columns), sg["pattern"])
                    for sc in sg_cols:
                        if sc in df.columns and sc not in seq_cols_ordered:
                            seq_cols_ordered.append(sc)

                for col_name, c_cfg in col_configs.items():
                    if c_cfg.get("type") == "sequence" and c_cfg.get("role", "input") == "input":
                        if col_name in df.columns and col_name not in seq_cols_ordered:
                            seq_cols_ordered.append(col_name)

                X_seq = None
                if seq_cols_ordered:
                    seq_data_list = []
                    for sc in seq_cols_ordered:
                        col_s = df[sc]
                        if not pd.api.types.is_numeric_dtype(col_s):
                            col_s = col_s.astype(str).str.replace(",", ".").str.strip()
                        num_s = pd.to_numeric(col_s, errors="coerce").fillna(0.0)
                        seq_data_list.append(num_s.to_numpy(dtype=np.float32))
                    seq_matrix = np.column_stack(seq_data_list).astype(np.float32)
                    df = df.drop(columns=seq_cols_ordered)

                    # Determine sequence normalization
                    norm_to_apply = (seq_groups[0].get("normalization") if seq_groups else None) or seq_norm_method or "row_zscore"
                    norm_to_apply = norm_to_apply.lower()

                    eps = 1e-7
                    if norm_to_apply == "row_zscore":
                        row_means = np.mean(seq_matrix, axis=1, keepdims=True)
                        row_stds = np.std(seq_matrix, axis=1, keepdims=True)
                        seq_matrix = (seq_matrix - row_means) / np.where(row_stds == 0, eps, row_stds)
                    elif norm_to_apply == "row_minmax":
                        row_mins = np.min(seq_matrix, axis=1, keepdims=True)
                        row_maxs = np.max(seq_matrix, axis=1, keepdims=True)
                        diff = row_maxs - row_mins
                        seq_matrix = (seq_matrix - row_mins) / np.where(diff == 0, eps, diff)
                    elif norm_to_apply in ("row_median", "subtract_median"):
                        row_medians = np.median(seq_matrix, axis=1, keepdims=True)
                        seq_matrix = seq_matrix - row_medians
                    elif norm_to_apply == "global_standard":
                        g_mean = float(np.mean(seq_matrix))
                        g_std = float(np.std(seq_matrix))
                        pipeline_state["sequence_global_mean"] = g_mean
                        pipeline_state["sequence_global_std"] = g_std
                        seq_matrix = (seq_matrix - g_mean) / (g_std if g_std != 0 else eps)
                    elif norm_to_apply == "global_minmax":
                        g_min = float(np.min(seq_matrix))
                        g_max = float(np.max(seq_matrix))
                        diff = g_max - g_min
                        pipeline_state["sequence_global_min"] = g_min
                        pipeline_state["sequence_global_max"] = g_max
                        seq_matrix = (seq_matrix - g_min) / (diff if diff != 0 else eps)

                    seq_matrix = np.nan_to_num(seq_matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                    X_seq = np.expand_dims(seq_matrix, axis=-1)
                    pipeline_state["sequence_normalization"] = norm_to_apply
                    pipeline_state["seq_len"] = int(seq_matrix.shape[1])
                    pipeline_state["sequence_columns"] = seq_cols_ordered

                # 5. Tabular Inputs (Numeric + Categorical)
                numeric_parts = []
                cat_parts = []

                for col in df.columns:
                    c_cfg = col_configs.get(col, {})
                    col_type = c_cfg.get("type")
                    if not col_type:
                        s_test = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")
                        col_type = "numeric" if float(s_test.notna().mean()) >= 0.8 else "categorical"

                    if col_type == "numeric":
                        s = df[col]
                        if not pd.api.types.is_numeric_dtype(s):
                            s = s.astype(str).str.replace(",", ".").str.strip()
                        vals = pd.to_numeric(s, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
                        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)

                        scaler = (c_cfg.get("scaler") or default_scaler or "standard").lower()
                        c_mean = float(np.mean(vals))
                        c_std = float(np.std(vals))
                        c_min = float(np.min(vals))
                        c_max = float(np.max(vals))
                        eps = 1e-7

                        if scaler == "standard":
                            scaled = (vals - c_mean) / (c_std if c_std != 0 else eps)
                        elif scaler == "minmax":
                            diff = c_max - c_min
                            scaled = (vals - c_min) / (diff if diff != 0 else eps)
                        else:
                            scaled = vals

                        pipeline_state["numeric_stats"][col] = {
                            "mean": c_mean, "std": c_std, "min": c_min, "max": c_max, "scaler": scaler
                        }
                        numeric_parts.append(scaled.reshape(-1, 1).astype(np.float32))

                    elif col_type == "categorical":
                        s = df[col].fillna("missing").astype(str)
                        encoder = (c_cfg.get("encoder") or default_encoder or "onehot").lower()
                        vocab = sorted(list(s.unique()))
                        v_map = {val: idx for idx, val in enumerate(vocab)}
                        pipeline_state["categorical_vocabs"][col] = {
                            "vocab": vocab, "encoder": encoder
                        }
                        indices = s.map(v_map).fillna(0).to_numpy(dtype=int)
                        if encoder == "onehot":
                            onehot = np.zeros((len(s), len(vocab)), dtype=np.float32)
                            onehot[np.arange(len(s)), indices] = 1.0
                            cat_parts.append(onehot)
                        else:
                            cat_parts.append(indices.reshape(-1, 1).astype(np.float32))

                tabular_blocks = []
                if numeric_parts:
                    tabular_blocks.append(np.hstack(numeric_parts))
                if cat_parts:
                    tabular_blocks.append(np.hstack(cat_parts))

                X_tabular = None
                if tabular_blocks:
                    X_tabular = np.hstack(tabular_blocks).astype(np.float32)
                    X_tabular = np.nan_to_num(X_tabular, nan=0.0, posinf=0.0, neginf=0.0)

                # 6. Output Target
                y = None
                labels = None
                if out_series is not None:
                    out_type = out_cfg.get("type")
                    if not out_type:
                        s_num = pd.to_numeric(out_series.astype(str).str.replace(",", "."), errors="coerce")
                        if float(s_num.notna().mean()) >= 0.8 and (out_series.nunique() > 20 or not pd.api.types.is_integer_dtype(s_num.dropna())):
                            out_type = "numeric"
                        else:
                            out_type = "categorical"

                    if out_type == "numeric":
                        task_type = "regression"
                        recommended_activation = "linear"
                        recommended_loss = "mse"
                        recommended_metrics = ["mae"]
                        y = pd.to_numeric(out_series.astype(str).str.replace(",", "."), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
                        labels = None
                    else:
                        task_type = "classification"
                        recommended_activation = "softmax"
                        recommended_loss = "sparse_categorical_crossentropy"
                        recommended_metrics = ["accuracy"]
                        classes = sorted(list(out_series.dropna().astype(str).unique()))
                        c_map = {c: i for i, c in enumerate(classes)}
                        encoded_y = out_series.astype(str).map(c_map).fillna(0).to_numpy(dtype=np.int32)
                        y = encoded_y
                        labels = encoded_y
                        pipeline_state["classes"] = classes
                else:
                    task_type = "unsupervised"
                    recommended_activation = "linear"
                    recommended_loss = "mse"
                    recommended_metrics = []

                # 7. Backward-compatible 2D data representation
                if X_seq is not None:
                    data = X_seq.squeeze(-1)
                elif X_tabular is not None:
                    data = X_tabular
                else:
                    data = np.zeros((len(sample_ids), 1), dtype=np.float32)

                cols = raw_columns

            pipeline_state["task_type"] = task_type
            pipeline_state["recommended_activation"] = recommended_activation
            pipeline_state["recommended_loss"] = recommended_loss
            pipeline_state["recommended_metrics"] = recommended_metrics

            save_dict = {
                "data": data,
                "sample_ids": sample_ids.astype(str)
            }
            if X_seq is not None:
                save_dict["X_seq"] = X_seq
            if X_tabular is not None:
                save_dict["X_tabular"] = X_tabular
            if y is not None:
                save_dict["y"] = y
            if labels is not None:
                save_dict["labels"] = labels
            np.savez_compressed(str(dest_path), **save_dict)

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

            metadata = {
                "columns": cols,
                "excluded_columns": cols_to_drop if 'cols_to_drop' in locals() else [],
                "missing_values": missing_count,
                "has_labels": labels is not None,
                "stats": summary_stats,
                "original_filename": source_path.name,
                "delimiter": delimiter,
                "has_header": has_header,
                "pipeline_state": pipeline_state,
                "task_type": task_type,
                "recommended_activation": recommended_activation,
                "recommended_loss": recommended_loss,
                "recommended_metrics": recommended_metrics,
                "schema_config": parsed_schema_config,
                "sample_ids_count": len(sample_ids),
                "has_sequence": X_seq is not None,
                "has_tabular": X_tabular is not None,
            }
            if X_seq is not None:
                metadata["seq_shape"] = list(X_seq.shape)
            if X_tabular is not None:
                metadata["tabular_shape"] = list(X_tabular.shape)

        num_rows, num_cols = data.shape
        with get_db_connection() as conn:
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
    def load_sample_ids(dataset_id: str) -> List[str]:
        """
        Retrieves the exact Input ID list for any registered dataset.
        If sample_ids was saved in .npz, returns it. Otherwise defaults to row index strings.
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
            if "sample_ids" in npz:
                return [str(x) for x in npz["sample_ids"]]
            elif "data" in npz:
                return [str(i) for i in range(len(npz["data"]))]
            return []

    @staticmethod
    def load_dataset_tensors(dataset_id: str) -> Dict[str, Any]:
        """
        Loads all multi-modal tensors and arrays from the registered dataset .npz file:
        data, X_seq, X_tabular, y, labels, sample_ids, and dataset record.
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

        result: Dict[str, Any] = {"record": record}
        with np.load(str(file_path), allow_pickle=False) as npz:
            for k in ("data", "X_seq", "X_tabular", "y", "labels", "sample_ids"):
                if k in npz:
                    result[k] = npz[k]
                else:
                    result[k] = None

        if result["sample_ids"] is None and result["data"] is not None:
            result["sample_ids"] = np.array([str(i) for i in range(len(result["data"]))])

        return result

    @staticmethod
    def update_dataset(
        dataset_id: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        schema_config: Optional[Union[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Renames a dataset or updates schema_config and metadata."""
        record = DataManager.get_dataset(dataset_id)
        if not record:
            raise ValueError(f"Dataset {dataset_id} not found")

        meta = record.get("metadata", {})
        if metadata:
            meta.update(metadata)
        if schema_config:
            parsed_cfg = json.loads(schema_config) if isinstance(schema_config, str) else schema_config
            meta["schema_config"] = parsed_cfg

        new_name = name.strip() if name and name.strip() else record["name"]
        now = datetime.now(timezone.utc).isoformat()

        with get_db_connection() as conn:
            conn.execute(
                "UPDATE datasets SET name = ?, metadata_json = ? WHERE id = ?",
                (new_name, json.dumps(meta), dataset_id)
            )
            conn.execute(
                "UPDATE projects SET updated_at = ? WHERE id = ?",
                (now, record["project_id"])
            )
            conn.commit()

        updated = DataManager.get_dataset(dataset_id)
        return updated or {}

    @staticmethod
    def delete_dataset(dataset_id: str) -> Dict[str, Any]:
        """Deletes dataset from SQLite and removes .npz file from disk."""
        record = DataManager.get_dataset(dataset_id)
        if not record:
            raise ValueError(f"Dataset {dataset_id} not found")

        file_path = resolve_storage_path(record["file_path"])
        if file_path and file_path.exists():
            try:
                file_path.unlink()
            except Exception:
                pass

        cand = DATASETS_DIR / record.get("project_id", "") / f"{dataset_id}.npz"
        if cand.exists():
            try:
                cand.unlink()
            except Exception:
                pass

        with get_db_connection() as conn:
            conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
            conn.execute("UPDATE projects SET active_dataset_id = NULL WHERE active_dataset_id = ?", (dataset_id,))
            conn.commit()

        return {"status": "deleted", "id": dataset_id}


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
    def _build_column_schema(
        all_columns: List[str],
        df: Optional[pd.DataFrame] = None,
        data_arr: Optional[np.ndarray] = None,
        suggested_id_cols: Optional[List[str]] = None,
        label_col: Optional[str] = None,
        excluded_cols: Optional[List[str]] = None,
        suggested_groups: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        suggested_id_set = set(str(c) for c in (suggested_id_cols or []))
        ex_set = set(str(c) for c in (excluded_cols or []))
        seq_cols_set = set()
        for grp in (suggested_groups or []):
            for c in grp.get("columns", []):
                seq_cols_set.add(str(c))

        schema = []
        for c in all_columns:
            c_str = str(c)
            c_low = c_str.lower().strip()

            # Role
            if c_str in suggested_id_set or c_low in ("id", "_id", "index", "idx", "uid", "sample_id", "patient_id", "record_id", "signal_id"):
                role = "id"
            elif label_col is not None and (c_str == str(label_col) or (str(label_col).isdigit() and int(label_col) < len(all_columns) and all_columns[int(label_col)] == c_str)):
                role = "output"
            elif label_col is None and c_low in ("label", "target", "y", "class", "diagnosis", "status", "output"):
                role = "output"
            elif c_str in ex_set:
                role = "ignore"
            else:
                role = "input"

            # Type & Unique Count & Sample Values
            if c_str in seq_cols_set:
                inferred_type = "sequence"
            else:
                inferred_type = "numeric"

            unique_count = 0
            sample_values = []

            if df is not None and c_str in df.columns:
                series = df[c_str]
                num_series = pd.to_numeric(series.astype(str).str.replace(",", "."), errors="coerce")
                valid_num_ratio = float(num_series.notna().mean()) if len(series) > 0 else 0.0
                if valid_num_ratio >= 0.8:
                    if role == "output" and series.nunique() <= 20 and pd.api.types.is_integer_dtype(num_series.dropna()):
                        inferred_type = "categorical"
                    elif c_str not in seq_cols_set:
                        inferred_type = "numeric"
                else:
                    if c_str not in seq_cols_set:
                        inferred_type = "categorical"

                unique_count = int(series.nunique(dropna=True))
                raw_samples = series.dropna().unique()[:5].tolist()
                for sv in raw_samples:
                    if isinstance(sv, (np.floating, float)):
                        sample_values.append(float(sv) if np.isfinite(sv) else None)
                    elif isinstance(sv, (np.integer, int)):
                        sample_values.append(int(sv))
                    else:
                        sample_values.append(str(sv))
            elif data_arr is not None:
                inferred_type = "sequence" if c_str in seq_cols_set else "numeric"
                unique_count = int(min(len(data_arr), 100))
                idx = all_columns.index(c_str) if c_str in all_columns else 0
                sample_values = [float(v) for v in data_arr[:5, idx] if np.isfinite(v)]

            schema.append({
                "name": c_str,
                "inferred_type": inferred_type,
                "inferred_role": role,
                "unique_count": unique_count,
                "sample_values": sample_values
            })

        return schema

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
            suggested_groups = DataManager.detect_sequence_groups(cols)
            if not suggested_groups and len(cols) > 1:
                suggested_groups = [{
                    "name": "sequence_signals",
                    "prefix": "sample_",
                    "pattern": f"sample_0-sample_{total_cols - 1}",
                    "columns": cols,
                    "count": total_cols
                }]
            column_schema = DataManager._build_column_schema(cols, data_arr=data, suggested_groups=suggested_groups)

            return {
                "file_name": p.name,
                "total_rows": total_rows,
                "total_cols": total_cols,
                "columns": cols,
                "all_columns": cols,
                "active_columns": cols,
                "excluded_columns": [],
                "suggested_id_cols": [],
                "column_schema": column_schema,
                "suggested_sequence_groups": suggested_groups,
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
        df_original = df.copy()

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

        suggested_groups = DataManager.detect_sequence_groups(all_columns)
        column_schema = DataManager._build_column_schema(
            all_columns,
            df=df_original,
            suggested_id_cols=suggested_id_cols,
            label_col=label_col,
            excluded_cols=parsed_exclude,
            suggested_groups=suggested_groups
        )

        return {
            "file_name": p.name,
            "total_rows": total_rows,
            "total_cols": len(active_columns),
            "all_columns": all_columns,
            "active_columns": active_columns,
            "excluded_columns": parsed_exclude,
            "suggested_id_cols": suggested_id_cols,
            "column_schema": column_schema,
            "suggested_sequence_groups": suggested_groups,
            "offset": offset,
            "max_rows": max_rows,
            "preview_rows": preview_rows,
            "summary_stats": stats,
            "missing_count": int(numeric_slice.isna().sum().sum()),
            "sample_waveforms": waveforms
        }
