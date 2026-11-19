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
        Auto-detects sequential prefix families in all_columns
        (e.g. s0..s511, a_0..a_511, dvdt_0..dvdt_511, ch1_0..ch1_100, 0..511).
        Supports interleaved columns and contiguous or monotonically increasing suffixes.
        For each detected group with count >= 3, returns dict with name, channel_name, prefix, pattern, columns, count.
        """
        if not all_columns or len(all_columns) < 3:
            return []

        rx_prefix = re.compile(r"^(([a-zA-Z_][a-zA-Z0-9_]*?)_?)(\d+)$")
        rx_numeric = re.compile(r"^(\d+)$")

        prefix_groups: Dict[Tuple[str, str], List[Tuple[str, int]]] = {}

        for col in all_columns:
            col_str = str(col)
            m = rx_prefix.match(col_str)
            if m:
                raw_prefix = m.group(1)
                clean_prefix = m.group(2).rstrip("_") or "seq"
                num_suffix = int(m.group(3))
                key = (raw_prefix, clean_prefix)
                if key not in prefix_groups:
                    prefix_groups[key] = []
                prefix_groups[key].append((col_str, num_suffix))
            else:
                m_num = rx_numeric.match(col_str)
                if m_num:
                    raw_prefix = ""
                    clean_prefix = "indexed"
                    num_suffix = int(m_num.group(1))
                    key = (raw_prefix, clean_prefix)
                    if key not in prefix_groups:
                        prefix_groups[key] = []
                    prefix_groups[key].append((col_str, num_suffix))

        groups: List[Dict[str, Any]] = []
        for (raw_prefix, clean_prefix), items in prefix_groups.items():
            items_sorted = sorted(items, key=lambda x: x[1])
            seen = set()
            col_run: List[str] = []
            for col_name, _ in items_sorted:
                if col_name not in seen:
                    seen.add(col_name)
                    col_run.append(col_name)

            if len(col_run) >= 3:
                pattern = f"{col_run[0]}-{col_run[-1]}"
                groups.append({
                    "name": f"channel_{clean_prefix}",
                    "channel_name": clean_prefix,
                    "prefix": raw_prefix,
                    "pattern": pattern,
                    "columns": col_run,
                    "count": len(col_run)
                })

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
        transpose: bool = False,
        missing_strategy: Optional[str] = "zero"
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

        # Handle missing values according to strategy
        if missing_strategy == "drop":
            valid_mask = ~numeric_df.isna().any(axis=1)
            if labels is not None:
                labels = labels[valid_mask.to_numpy()]
            numeric_df = numeric_df.dropna().reset_index(drop=True)
            df = df.iloc[valid_mask.to_numpy()].reset_index(drop=True)
        elif missing_strategy == "mean":
            numeric_df = numeric_df.fillna(numeric_df.mean())
        elif missing_strategy == "ffill":
            numeric_df = numeric_df.ffill().bfill()

        cols = [str(c) for c in df.columns]
        # Auto-detect if row 0 was accidentally read as data instead of column headers
        if len(numeric_df) > 1:
            r0_nan_ratio = float(numeric_df.iloc[0].isna().mean())
            r1_nan_ratio = float(numeric_df.iloc[1].isna().mean())
            if r0_nan_ratio > 0.6 and r1_nan_ratio < 0.3:
                cols = [str(c) for c in df.iloc[0]]
                df = df.iloc[1:].reset_index(drop=True)
                numeric_df = df.apply(pd.to_numeric, errors="coerce")
                if missing_strategy == "drop":
                    numeric_df = numeric_df.dropna().reset_index(drop=True)

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
        schema_config: Optional[Union[str, Dict[str, Any]]] = None,
        missing_strategy: Optional[str] = "zero",
        split_config: Optional[Union[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Parses source data, saves it persistently in storage/datasets, computes metadata,
        handles missing value strategy and train/val/test split configuration,
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

        parsed_split_config = None
        if split_config:
            if isinstance(split_config, str):
                try:
                    parsed_split_config = json.loads(split_config)
                except Exception:
                    if split_config == "70_15_15":
                        parsed_split_config = {"train": 0.70, "val": 0.15, "test": 0.15}
                    elif split_config == "80_10_10":
                        parsed_split_config = {"train": 0.80, "val": 0.10, "test": 0.10}
                    elif split_config == "80_20_0":
                        parsed_split_config = {"train": 0.80, "val": 0.20, "test": 0.0}
                    elif split_config == "100_0_0":
                        parsed_split_config = {"train": 1.0, "val": 0.0, "test": 0.0}
                    else:
                        parsed_split_config = {"preset": split_config}
            elif isinstance(split_config, dict):
                parsed_split_config = split_config

        dataset_id = str(uuid.uuid4())
        dest_dir = DATASETS_DIR / project_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"{dataset_id}.npz"
        now = datetime.now(timezone.utc).isoformat()

        if parsed_schema_config is None:
            # Backward-compatible loading: first inspect tabular file for ID and extra columns if has_header
            extracted_sample_ids = None
            extra_columns_dict: Dict[str, List[str]] = {}
            auto_exclude_for_raw = []
            if exclude_cols is not None:
                if isinstance(exclude_cols, str):
                    try:
                        auto_exclude_for_raw = json.loads(exclude_cols)
                    except Exception:
                        auto_exclude_for_raw = [c.strip() for c in exclude_cols.split(",") if c.strip()]
                else:
                    auto_exclude_for_raw = [str(c) for c in exclude_cols]

            ext_check = source_path.suffix.lower()
            if has_header and not transpose and ext_check not in (".npy", ".npz"):
                try:
                    delim_use = DataManager.detect_delimiter(source_path) if (delimiter == "auto" or not delimiter) else delimiter
                    sep_use = delim_use if delim_use != "\\t" else "\t"
                    df_head = pd.read_csv(str(source_path), sep=sep_use, header=0, encoding=encoding)
                    df_head.columns = [str(c) for c in df_head.columns]
                    seq_groups_detected = DataManager.detect_sequence_groups(list(df_head.columns))
                    seq_col_set = {c for g in seq_groups_detected for c in g.get("columns", [])}
                    id_col_cand = None
                    for col in df_head.columns:
                        if col.lower().strip() in ("id", "_id", "input_id", "input id", "uid", "sample_id", "sampleid", "patient_id", "record_id", "signal_id"):
                            id_col_cand = col
                            break
                    if id_col_cand and id_col_cand in df_head.columns and len(df_head.columns) > 1:
                        extracted_sample_ids = np.array([str(v) for v in df_head[id_col_cand].tolist()], dtype=str)
                        if id_col_cand not in auto_exclude_for_raw and id_col_cand != label_col:
                            auto_exclude_for_raw.append(id_col_cand)
                    for col in df_head.columns:
                        if col == id_col_cand:
                            continue
                        if (seq_col_set and col not in seq_col_set) or (col in auto_exclude_for_raw):
                            extra_columns_dict[col] = ["" if pd.isna(v) else str(v) for v in df_head[col].tolist()]
                except Exception:
                    pass

            data, labels, cols = DataManager.load_raw_data(
                source_path,
                delimiter=delimiter,
                has_header=has_header,
                encoding=encoding,
                label_col=label_col,
                exclude_cols=auto_exclude_for_raw if auto_exclude_for_raw else exclude_cols,
                transpose=transpose,
                missing_strategy=missing_strategy
            )

            # Missing value handling
            if missing_strategy == "drop":
                valid_mask = np.all(np.isfinite(data), axis=1)
                data = data[valid_mask]
                if labels is not None:
                    labels = labels[valid_mask]
                if extracted_sample_ids is not None and len(extracted_sample_ids) == len(valid_mask):
                    extracted_sample_ids = extracted_sample_ids[valid_mask]
                for k_col in list(extra_columns_dict.keys()):
                    if len(extra_columns_dict[k_col]) == len(valid_mask):
                        extra_columns_dict[k_col] = [v for v, m in zip(extra_columns_dict[k_col], valid_mask) if m]
            elif missing_strategy == "mean":
                col_means = np.nanmean(data, axis=0)
                col_means = np.nan_to_num(col_means, nan=0.0)
                inds = np.where(~np.isfinite(data))
                data[inds] = np.take(col_means, inds[1])
            data = np.nan_to_num(data, copy=True, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            if data.ndim == 1:
                data = data.reshape(1, -1)
            elif data.ndim == 0:
                data = data.reshape(1, 1)

            num_rows, num_cols = int(data.shape[0]), int(data.shape[1])
            # Input ID Traceability: use extracted_sample_ids if matching row count, else default sample_ids
            if extracted_sample_ids is not None and len(extracted_sample_ids) == num_rows:
                sample_ids = extracted_sample_ids.astype(str)
            else:
                sample_ids = np.array([str(i) for i in range(num_rows)], dtype=str)

            save_dict = {
                "data": data,
                "sample_ids": sample_ids
            }
            if extra_columns_dict:
                save_dict["extra_columns_json"] = np.array(json.dumps(extra_columns_dict))
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
                "extra_columns": list(extra_columns_dict.keys()),
                "missing_values": missing_count,
                "missing_strategy": missing_strategy or "zero",
                "has_labels": labels is not None,
                "stats": summary_stats,
                "original_filename": source_path.name,
                "delimiter": delimiter,
                "has_header": has_header,
                "sample_ids_count": len(sample_ids)
            }
            if parsed_split_config:
                metadata["split_config"] = parsed_split_config

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
                df_full_for_extra = df.copy()
                extra_columns_dict: Dict[str, List[str]] = {}

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
                        if c_low in ("id", "_id", "input_id", "input id", "uid", "sample_id", "sampleid", "patient_id", "record_id", "signal_id"):
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
                stack_mode = (parsed_schema_config.get("sequence_stack_mode") or "multichannel").lower()
                is_multichannel = len(seq_groups) > 1 and stack_mode in ("multichannel", "multi_channel", "stack", "channels")

                def _extract_cols_matrix(df_source: pd.DataFrame, col_names: List[str]) -> np.ndarray:
                    parts = []
                    for c in col_names:
                        col_s = df_source[c]
                        if not pd.api.types.is_numeric_dtype(col_s):
                            col_s = col_s.astype(str).str.replace(",", ".").str.strip()
                        num_s = pd.to_numeric(col_s, errors="coerce").fillna(0.0)
                        parts.append(num_s.to_numpy(dtype=np.float32))
                    return np.column_stack(parts).astype(np.float32)

                def _apply_sequence_norm(matrix: np.ndarray, norm_method: Optional[str]) -> np.ndarray:
                    norm = (norm_method or "row_zscore").lower()
                    eps = 1e-7
                    mat = matrix.copy()
                    if norm == "row_zscore":
                        row_means = np.mean(mat, axis=1, keepdims=True)
                        row_stds = np.std(mat, axis=1, keepdims=True)
                        mat = (mat - row_means) / np.where(row_stds == 0, eps, row_stds)
                    elif norm == "row_minmax":
                        row_mins = np.min(mat, axis=1, keepdims=True)
                        row_maxs = np.max(mat, axis=1, keepdims=True)
                        diff = row_maxs - row_mins
                        mat = (mat - row_mins) / np.where(diff == 0, eps, diff)
                    elif norm in ("row_median", "subtract_median"):
                        row_medians = np.median(mat, axis=1, keepdims=True)
                        mat = mat - row_medians
                    elif norm in ("row_mode", "subtract_mode"):
                        from app.services.signal_processing import subtract_mode
                        mat = subtract_mode(mat, per_signal=True)
                    elif norm in ("row_gaussian_peak", "subtract_gaussian_peak"):
                        from app.services.signal_processing import subtract_gaussian_peak
                        mat = subtract_gaussian_peak(mat, per_signal=True)
                    elif norm in ("row_mean", "subtract_mean"):
                        from app.services.signal_processing import subtract_mean
                        mat = subtract_mean(mat, per_signal=True)
                    elif norm in ("row_first_n_median", "subtract_first_n_median", "first_n_median"):
                        from app.services.signal_processing import subtract_first_n_median
                        mat = subtract_first_n_median(mat, n=50, per_signal=True)
                    elif norm in ("row_first_n_mean", "subtract_first_n_mean", "first_n_mean"):
                        from app.services.signal_processing import subtract_first_n_mean
                        mat = subtract_first_n_mean(mat, n=50, per_signal=True)
                    elif norm in ("row_last_n_median", "subtract_last_n_median", "last_n_median"):
                        from app.services.signal_processing import subtract_last_n_median
                        mat = subtract_last_n_median(mat, n=50, per_signal=True)
                    elif norm in ("row_linear_baseline", "subtract_linear_baseline"):
                        from app.services.signal_processing import subtract_linear_baseline
                        mat = subtract_linear_baseline(mat)
                    elif norm in ("row_polynomial_baseline", "subtract_polynomial_baseline"):
                        from app.services.signal_processing import subtract_polynomial_baseline
                        mat = subtract_polynomial_baseline(mat)
                    elif norm in ("row_rolling_baseline", "subtract_rolling_baseline"):
                        from app.services.signal_processing import subtract_rolling_baseline
                        mat = subtract_rolling_baseline(mat)
                    elif norm in ("row_scale_min", "scale_min"):
                        from app.services.signal_processing import scale_min
                        mat = scale_min(mat, per_signal=True)
                    elif norm in ("row_scale_max", "scale_max"):
                        from app.services.signal_processing import scale_max
                        mat = scale_max(mat, per_signal=True)
                    elif norm == "global_standard":
                        g_mean = float(np.mean(mat))
                        g_std = float(np.std(mat))
                        pipeline_state["sequence_global_mean"] = g_mean
                        pipeline_state["sequence_global_std"] = g_std
                        mat = (mat - g_mean) / (g_std if g_std != 0 else eps)
                    elif norm == "global_minmax":
                        g_min = float(np.min(mat))
                        g_max = float(np.max(mat))
                        diff = g_max - g_min
                        pipeline_state["sequence_global_min"] = g_min
                        pipeline_state["sequence_global_max"] = g_max
                        mat = (mat - g_min) / (diff if diff != 0 else eps)
                    elif norm in ("none", "raw", "identity"):
                        pass
                    return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

                X_seq = None
                if is_multichannel:
                    channel_matrices = []
                    all_seq_cols = []
                    for sg in seq_groups:
                        sg_cols = sg.get("columns", [])
                        if not sg_cols and sg.get("pattern"):
                            sg_cols = DataManager.resolve_sequence_columns(list(df.columns), sg["pattern"])
                        valid_cols = [c for c in sg_cols if c in df.columns]
                        if not valid_cols:
                            continue
                        all_seq_cols.extend(valid_cols)
                        raw_mat = _extract_cols_matrix(df, valid_cols)
                        norm_to_apply = sg.get("normalization") or seq_norm_method or "row_zscore"
                        norm_mat = _apply_sequence_norm(raw_mat, norm_to_apply)
                        channel_matrices.append(norm_mat)

                    if channel_matrices:
                        max_seq_len = max(m.shape[1] for m in channel_matrices)
                        padded_matrices = []
                        for m in channel_matrices:
                            if m.shape[1] < max_seq_len:
                                m = np.pad(m, ((0, 0), (0, max_seq_len - m.shape[1])), mode="constant")
                            padded_matrices.append(m)

                        X_seq = np.stack(padded_matrices, axis=-1).astype(np.float32)
                        df = df.drop(columns=[c for c in set(all_seq_cols) if c in df.columns])

                        pipeline_state["seq_len"] = int(X_seq.shape[1])
                        pipeline_state["seq_channels"] = int(X_seq.shape[2])
                        pipeline_state["sequence_group_names"] = [
                            sg.get("channel_name") or sg.get("name", f"ch_{i}")
                            for i, sg in enumerate(seq_groups)
                        ]
                        pipeline_state["sequence_groups"] = seq_groups
                        pipeline_state["sequence_columns"] = all_seq_cols
                        pipeline_state["sequence_normalization"] = seq_norm_method or "per_group"

                else:
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

                    if seq_cols_ordered:
                        seq_matrix = _extract_cols_matrix(df, seq_cols_ordered)
                        df = df.drop(columns=seq_cols_ordered)

                        norm_to_apply = (seq_groups[0].get("normalization") if seq_groups else None) or seq_norm_method or "row_zscore"
                        seq_matrix = _apply_sequence_norm(seq_matrix, norm_to_apply)

                        X_seq = np.expand_dims(seq_matrix, axis=-1)
                        pipeline_state["sequence_normalization"] = norm_to_apply
                        pipeline_state["seq_len"] = int(seq_matrix.shape[1])
                        pipeline_state["seq_channels"] = 1
                        pipeline_state["sequence_columns"] = seq_cols_ordered
                        if seq_groups:
                            pipeline_state["sequence_group_names"] = [
                                seq_groups[0].get("channel_name") or seq_groups[0].get("name", "ch_0")
                            ]
                            pipeline_state["sequence_groups"] = seq_groups

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
                    if X_seq.shape[2] == 1:
                        data = X_seq.squeeze(-1)
                    else:
                        data = X_seq[:, :, 0]
                elif X_tabular is not None:
                    data = X_tabular
                else:
                    data = np.zeros((len(sample_ids), 1), dtype=np.float32)

                cols = raw_columns

                # Collect non-sequence extra columns from the main table (e.g., Source, excluded/ignored columns, metadata, target)
                seq_used_set = set(pipeline_state.get("sequence_columns", []))
                if not seq_used_set:
                    detected_grps = DataManager.detect_sequence_groups(raw_columns)
                    seq_used_set = {c for g in detected_grps for c in g.get("columns", [])}
                for col_c in raw_columns:
                    if col_c == id_col_name:
                        continue
                    if (seq_used_set and col_c not in seq_used_set) or (col_c in ignore_cols) or (col_c == out_col_name):
                        if col_c in df_full_for_extra.columns:
                            extra_columns_dict[col_c] = [
                                "" if pd.isna(v) else str(v)
                                for v in df_full_for_extra[col_c].tolist()
                            ]

            pipeline_state["task_type"] = task_type
            pipeline_state["recommended_activation"] = recommended_activation
            pipeline_state["recommended_loss"] = recommended_loss
            pipeline_state["recommended_metrics"] = recommended_metrics

            save_dict = {
                "data": data,
                "sample_ids": sample_ids.astype(str)
            }
            if 'extra_columns_dict' in locals() and extra_columns_dict:
                save_dict["extra_columns_json"] = np.array(json.dumps(extra_columns_dict))
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
                "extra_columns": list(extra_columns_dict.keys()) if ('extra_columns_dict' in locals() and extra_columns_dict) else [],
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
            metadata["missing_strategy"] = missing_strategy or "zero"
            if parsed_split_config:
                metadata["split_config"] = parsed_split_config
            if X_seq is not None:
                metadata["seq_shape"] = list(X_seq.shape)
            if X_tabular is not None:
                metadata["tabular_shape"] = list(X_tabular.shape)

        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim == 0:
            data = data.reshape(1, 1)

        num_rows = int(data.shape[0])
        num_cols = int(data.shape[1])
        metadata["num_rows"] = num_rows
        metadata["row_count"] = num_rows
        metadata["col_count"] = num_cols
        metadata["num_cols"] = num_cols

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
            "num_rows": num_rows,
            "total_rows": num_rows,
            "col_count": num_cols,
            "num_cols": num_cols,
            "total_cols": num_cols,
            "has_header": has_header,
            "label_col": label_col,
            "metadata": metadata
        }

    @staticmethod
    def _is_default_zero_indexed(sids: List[str]) -> bool:
        """Returns True if sids is strictly the default ['0', '1', ..., str(N-1)] fallback."""
        if not sids:
            return True
        if sids[0] != "0":
            return False
        return sids == [str(i) for i in range(len(sids))]

    @staticmethod
    def resolve_dataset_chain(dataset_id: str) -> List[str]:
        """
        Traces dataset_id back through original_dataset_id and source_feature_id
        to return the lineage chain [dataset_id, parent_id, ..., root_dataset_id].
        """
        chain: List[str] = []
        visited = set()
        curr = str(dataset_id).strip() if dataset_id else ""
        while curr and curr not in visited:
            visited.add(curr)
            chain.append(curr)
            rec = DataManager.get_dataset(curr)
            if not rec:
                break
            meta = rec.get("metadata") or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            next_ds = meta.get("original_dataset_id") or meta.get("parent_dataset_id")
            if not next_ds and meta.get("source_feature_id"):
                try:
                    with get_db_connection() as conn:
                        f_row = conn.execute(
                            "SELECT dataset_id FROM features WHERE id = ?",
                            (meta["source_feature_id"],)
                        ).fetchone()
                        if f_row and f_row["dataset_id"]:
                            next_ds = f_row["dataset_id"]
                except Exception:
                    pass
            curr = str(next_ds).strip() if next_ds else ""
        return chain

    @staticmethod
    def load_sample_ids(dataset_id: str, expected_count: Optional[int] = None) -> List[str]:
        """
        Retrieves the exact Input ID list for any registered dataset (or derived/processed dataset).
        Walks the dataset lineage chain (original_dataset_id / source_feature_id) so that
        processed datasets and extracted feature datasets always preserve real sample IDs
        (e.g. starting at 1 instead of falling back to 0..N-1).
        """
        record = DataManager.get_dataset(dataset_id)
        if not record:
            raise ValueError(f"Dataset {dataset_id} not found")

        chain = DataManager.resolve_dataset_chain(dataset_id)
        fallback_sids: Optional[List[str]] = None
        target_len = expected_count

        for ds_id in chain:
            rec = DataManager.get_dataset(ds_id) if ds_id != dataset_id else record
            if not rec:
                continue
            file_path = resolve_storage_path(rec["file_path"])
            if not file_path or not file_path.exists():
                cand = DATASETS_DIR / rec.get("project_id", "") / f"{ds_id}.npz"
                if cand.exists():
                    file_path = cand
                else:
                    continue
            try:
                with np.load(str(file_path), allow_pickle=False) as npz:
                    if target_len is None and "data" in npz:
                        target_len = len(npz["data"])
                    if "sample_ids" in npz:
                        sids = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in npz["sample_ids"]]
                        if target_len is not None and len(sids) >= target_len:
                            sids = sids[:target_len]
                        if fallback_sids is None and sids:
                            fallback_sids = sids
                        if sids and not DataManager._is_default_zero_indexed(sids):
                            return sids
            except Exception:
                continue

        if fallback_sids is not None:
            return fallback_sids
        if target_len is not None and target_len > 0:
            return [str(i) for i in range(target_len)]
        raise FileNotFoundError(f"Dataset file missing on disk: {record['file_path']}")

    @staticmethod
    def _extract_extra_columns_from_source_csv(rec: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Fallback for older datasets registered before extra_columns_json was stored in .npz:
        locates original_filename in storage/temp or DATA_DIR and extracts non-sequence columns.
        """
        meta = rec.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        orig_name = meta.get("original_filename")
        if not orig_name or str(orig_name).lower().endswith((".npy", ".npz")):
            return {}

        candidates: List[Path] = []
        proj_id = rec.get("project_id", "")
        if proj_id:
            temp_proj = STORAGE_DIR / "temp" / proj_id / orig_name
            if temp_proj.exists():
                candidates.append(temp_proj)
        temp_root = STORAGE_DIR / "temp"
        if temp_root.exists():
            for p in temp_root.rglob(orig_name):
                if p.is_file():
                    candidates.append(p)
        data_cand = DATA_DIR / orig_name
        if data_cand.exists():
            candidates.append(data_cand)

        if not candidates:
            return {}

        csv_path = candidates[0]
        try:
            delim = meta.get("delimiter") or ","
            if delim == "auto" or not delim:
                delim = DataManager.detect_delimiter(csv_path)
            sep = delim if delim != "\\t" else "\t"
            has_hdr = meta.get("has_header", bool(rec.get("has_header", 1)))
            if not has_hdr:
                return {}
            df = pd.read_csv(str(csv_path), sep=sep, header=0)
            df.columns = [str(c) for c in df.columns]
            all_cols = list(df.columns)
            seq_groups = DataManager.detect_sequence_groups(all_cols)
            seq_col_set = {c for g in seq_groups for c in g.get("columns", [])}
            excluded_set = set(str(c) for c in (meta.get("excluded_columns") or []))
            sc_cols = (meta.get("schema_config") or {}).get("columns") or {}

            id_col = None
            for c_name, c_cfg in sc_cols.items():
                if isinstance(c_cfg, dict) and c_cfg.get("role") == "id":
                    id_col = c_name
                    break
            if not id_col:
                for col in all_cols:
                    if col.lower().strip() in ("id", "_id", "input_id", "input id", "uid", "sample_id", "sampleid", "patient_id", "record_id", "signal_id"):
                        id_col = col
                        break

            extra_dict: Dict[str, List[str]] = {}
            for col in all_cols:
                if col == id_col:
                    continue
                if (seq_col_set and col not in seq_col_set) or (col in excluded_set):
                    extra_dict[col] = ["" if pd.isna(v) else str(v) for v in df[col].tolist()]
            return extra_dict
        except Exception:
            return {}

    @staticmethod
    def load_extra_columns(
        dataset_id: str,
        selected_columns: Optional[List[str]] = None,
        expected_count: Optional[int] = None
    ) -> Dict[str, List[str]]:
        """
        Loads non-signal / metadata columns from the main table (e.g. 'Source', category, batch, etc.)
        by walking the dataset lineage chain (original_dataset_id / source_feature_id).
        """
        chain = DataManager.resolve_dataset_chain(dataset_id)
        extra_dict: Dict[str, List[str]] = {}

        for ds_id in chain:
            rec = DataManager.get_dataset(ds_id)
            if not rec:
                continue
            file_path = resolve_storage_path(rec["file_path"])
            if not file_path or not file_path.exists():
                cand = DATASETS_DIR / rec.get("project_id", "") / f"{ds_id}.npz"
                if cand.exists():
                    file_path = cand
            if file_path and file_path.exists():
                try:
                    with np.load(str(file_path), allow_pickle=False) as npz:
                        if "extra_columns_json" in npz:
                            raw_json = str(npz["extra_columns_json"].item())
                            loaded = json.loads(raw_json)
                            if isinstance(loaded, dict) and loaded:
                                extra_dict = {str(k): [str(v) for v in vals] for k, vals in loaded.items()}
                                break
                except Exception:
                    pass
            # Fallback to source CSV if available
            csv_extra = DataManager._extract_extra_columns_from_source_csv(rec)
            if csv_extra:
                extra_dict = csv_extra
                break

        if not extra_dict:
            return {}

        if selected_columns is not None:
            sel_set = [str(c).strip() for c in selected_columns if str(c).strip()]
            extra_dict = {k: extra_dict[k] for k in sel_set if k in extra_dict}

        if expected_count is not None and expected_count > 0:
            aligned: Dict[str, List[str]] = {}
            for k, vals in extra_dict.items():
                if len(vals) >= expected_count:
                    aligned[k] = vals[:expected_count]
                else:
                    aligned[k] = vals + [""] * (expected_count - len(vals))
            return aligned

        return extra_dict

    @staticmethod
    def get_available_extra_columns(dataset_id: str) -> List[str]:
        """Returns the list of available extra column names from the main table for a dataset or its root ancestor."""
        extra = DataManager.load_extra_columns(dataset_id)
        return list(extra.keys())

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
        schema_config: Optional[Union[str, Dict[str, Any]]] = None,
        row_count: Optional[int] = None,
        col_count: Optional[int] = None
    ) -> Dict[str, Any]:
        """Renames a dataset or updates schema_config, metadata, row_count, col_count."""
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
        new_row_count = int(row_count) if row_count is not None else record.get("row_count", 0)
        new_col_count = int(col_count) if col_count is not None else record.get("col_count", 0)
        meta["row_count"] = new_row_count
        meta["num_rows"] = new_row_count
        meta["col_count"] = new_col_count
        meta["num_cols"] = new_col_count
        now = datetime.now(timezone.utc).isoformat()

        with get_db_connection() as conn:
            conn.execute(
                "UPDATE datasets SET name = ?, metadata_json = ?, row_count = ?, col_count = ? WHERE id = ?",
                (new_name, json.dumps(meta), new_row_count, new_col_count, dataset_id)
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
                d["num_rows"] = d.get("row_count", 0)
                d["total_rows"] = d.get("row_count", 0)
                d["num_cols"] = d.get("col_count", 0)
                d["total_cols"] = d.get("col_count", 0)
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
            "num_rows": total_rows,
            "total_rows": total_rows,
            "col_count": total_cols,
            "num_cols": total_cols,
            "total_cols": total_cols,
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
                "row_count": total_rows,
                "num_rows": total_rows,
                "total_cols": total_cols,
                "col_count": total_cols,
                "num_cols": total_cols,
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
            "row_count": total_rows,
            "num_rows": total_rows,
            "total_cols": len(active_columns),
            "col_count": len(active_columns),
            "num_cols": len(active_columns),
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

    @staticmethod
    def get_dataset_io_schema(dataset_id: str) -> Dict[str, Any]:
        """
        Inspects dataset record and .npz file to report IO schema:
        signal sequence length, channels, tabular dimensions, classification classes,
        available derived signal channels, and target sources.
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
                raise FileNotFoundError(f"Dataset file missing on disk: {record.get('file_path')}")

        with np.load(str(file_path), allow_pickle=False) as npz:
            keys = list(npz.keys())
            data = npz["data"] if "data" in npz else None
            X_seq = npz["X_seq"] if "X_seq" in npz else None
            X_tabular = npz["X_tabular"] if "X_tabular" in npz else None
            labels = npz["labels"] if "labels" in npz else (npz["y"] if "y" in npz else None)
            has_region_labels = "region_labels" in keys

        n_samples = 0
        for cand in (X_seq, data, X_tabular):
            if cand is not None:
                n_samples = int(cand.shape[0])
                break

        seq_len = 0
        if X_seq is not None and X_seq.ndim >= 2:
            seq_len = int(X_seq.shape[1])
        elif data is not None and data.ndim >= 2:
            seq_len = int(data.shape[1])

        raw_channels = 1
        if X_seq is not None and X_seq.ndim == 3:
            raw_channels = int(X_seq.shape[2])

        tabular_dim = 0
        if X_tabular is not None and X_tabular.ndim >= 2:
            tabular_dim = int(X_tabular.shape[1])

        has_labels = labels is not None and len(labels) > 0
        n_classes = 0
        label_values = []
        if has_labels:
            unique_vals = np.unique(labels)
            n_classes = int(len(unique_vals))
            for v in unique_vals[:20]:
                if isinstance(v, (np.integer, int)):
                    label_values.append(int(v))
                elif isinstance(v, (np.floating, float)):
                    label_values.append(float(v))
                else:
                    label_values.append(str(v))

        metadata = record.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = {}

        pipeline_state = metadata.get("pipeline_state", {})
        if isinstance(pipeline_state, str):
            try:
                pipeline_state = json.loads(pipeline_state)
            except Exception:
                pipeline_state = {}

        seq_groups_raw = pipeline_state.get("sequence_groups") or metadata.get("sequence_groups")
        if not seq_groups_raw and metadata.get("schema_config"):
            seq_groups_raw = metadata["schema_config"].get("sequence_groups", [])

        sequence_groups: List[Dict[str, Any]] = []
        if seq_groups_raw and isinstance(seq_groups_raw, list):
            for sg in seq_groups_raw:
                c_name = sg.get("channel_name") or sg.get("name", "").replace("channel_", "").replace("sequence_", "")
                raw_pfx = sg.get("prefix", "")
                sequence_groups.append({
                    "name": sg.get("name") or f"channel_{c_name}",
                    "channel_name": c_name,
                    "prefix": raw_pfx,
                    "pattern": sg.get("pattern", ""),
                    "count": sg.get("count", len(sg.get("columns", [])))
                })

        columns = metadata.get("columns", [f"col_{i}" for i in range(seq_len or tabular_dim or 1)])
        tabular_columns = metadata.get("tabular_columns", [f"tab_{i}" for i in range(tabular_dim)]) if tabular_dim > 0 else []

        available_signal_channels = [
            {"id": "raw", "label": "Raw Signal x(t)", "description": "Original 1D waveform"},
            {"id": "derivative_1", "label": "1st Derivative (dx/dt)", "description": "Central finite difference first derivative"},
            {"id": "derivative_2", "label": "2nd Derivative (d²x/dt²)", "description": "Second derivative (curvature / inflection)"},
            {"id": "region_mask", "label": "Regional Labels (Bölgesel Etiket)", "description": "Pointwise regional event / peak mask"},
            {"id": "envelope_rms", "label": "RMS Envelope", "description": "Moving root-mean-square amplitude envelope"},
            {"id": "rolling_mean", "label": "Rolling Mean", "description": "Smoothed baseline signal"},
            {"id": "fft_magnitude", "label": "FFT Spectrum", "description": "Frequency domain magnitude profile"}
        ]

        # Add multi-channel group channels (e.g. group:s, group:a, group:dvdt)
        for sg in sequence_groups:
            c_name = sg.get("channel_name")
            if c_name:
                ch_id = f"group:{c_name}"
                if not any(c.get("id") == ch_id for c in available_signal_channels):
                    available_signal_channels.append({
                        "id": ch_id,
                        "label": f"Channel Group: {c_name} ({sg.get('pattern', '')})",
                        "description": f"Pre-grouped channel '{c_name}' from dataset"
                    })

        available_target_sources = [
            {"id": "reconstruction", "label": "Signal Reconstruction (Autoencoder)", "default_loss": "mse", "default_activation": "linear"},
            {"id": "region_labels", "label": "Regional Labels / 1D Segmentation (Bölgesel Etiketler)", "default_loss": "binary_crossentropy", "default_activation": "sigmoid"},
            {"id": "label_class", "label": "Class Label (Classification y)", "default_loss": "sparse_categorical_crossentropy", "default_activation": "softmax"},
            {"id": "regression", "label": "Continuous Target (Regression)", "default_loss": "mse", "default_activation": "linear"},
            {"id": "clustering", "label": "Cluster Probabilities (IDEC)", "default_loss": "kld", "default_activation": "softmax"}
        ]

        return {
            "dataset_id": dataset_id,
            "name": record.get("name", "Dataset"),
            "row_count": n_samples,
            "num_rows": n_samples,
            "total_rows": n_samples,
            "seq_len": seq_len,
            "raw_channels": raw_channels,
            "sequence_groups": sequence_groups,
            "tabular_dim": tabular_dim,
            "has_labels": has_labels,
            "n_classes": n_classes,
            "label_values": label_values,
            "has_region_labels": has_region_labels,
            "columns": columns,
            "tabular_columns": tabular_columns,
            "label_col": record.get("label_col"),
            "available_signal_channels": available_signal_channels,
            "available_target_sources": available_target_sources
        }
