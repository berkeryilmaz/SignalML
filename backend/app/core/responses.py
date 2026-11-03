"""
backend/app/core/responses.py
Hardened JSON response classes and recursive sanitizers for SignalML.
Guarantees RFC 8259 JSON compliance by preventing NaN and Inf serialization crashes.
"""

import math
import json
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional, Set
from starlette.responses import JSONResponse
import numpy as np


def sanitize_for_json(obj: Any, nan_value: Any = None, _seen: Optional[Set[int]] = None) -> Any:
    """
    Recursively transforms data structures to ensure standard JSON compliance:
    1. Replaces NaN, +Inf, -Inf float values with `nan_value` (default: None -> JSON null).
    2. Converts NumPy numeric scalars, booleans, and arrays into native Python types.
    3. Converts Path, datetime, date, UUID objects into JSON-compatible strings.
    4. Serializes Pydantic models if passed raw.
    5. Converts dictionary keys to strings.
    6. Prevents infinite loops on circular references.
    """
    if obj is None:
        return None

    # Native Python primitives
    if isinstance(obj, str):
        return obj

    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)

    # Floats and NumPy floating scalars
    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return nan_value
        return val

    # NumPy integer scalars and int
    if isinstance(obj, (int, np.integer)):
        return int(obj)

    # Complex numbers (e.g. from FFT, Hilbert transforms)
    if isinstance(obj, (complex, np.complexfloating)):
        real_val = float(obj.real)
        imag_val = float(obj.imag)
        return {
            "real": real_val if math.isfinite(real_val) else nan_value,
            "imag": imag_val if math.isfinite(imag_val) else nan_value
        }

    # NumPy ndarrays (convert to nested list and recursively sanitize)
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist(), nan_value, _seen)

    # Other NumPy generic types
    if isinstance(obj, np.generic):
        return sanitize_for_json(obj.item(), nan_value, _seen)

    # Pydantic v2 / v1 models
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        return sanitize_for_json(obj.model_dump(), nan_value, _seen)
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        return sanitize_for_json(obj.dict(), nan_value, _seen)

    # Standard serialization helpers
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    if isinstance(obj, uuid.UUID):
        return str(obj)

    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            return str(obj)

    # Circular reference guard for complex nested structures
    if _seen is None:
        _seen = set()
    obj_id = id(obj)
    if obj_id in _seen:
        return None
    _seen.add(obj_id)

    try:
        if isinstance(obj, dict):
            return {
                (int(k) if isinstance(k, np.integer) else (str(k) if not isinstance(k, (str, int)) else k)): sanitize_for_json(v, nan_value, _seen)
                for k, v in obj.items()
            }

        if isinstance(obj, (list, tuple, set, frozenset)):
            return [sanitize_for_json(item, nan_value, _seen) for item in obj]
    finally:
        _seen.remove(obj_id)

    return obj


class SafeJSONResponse(JSONResponse):
    """
    Hardened JSONResponse subclass that prevents Starlette's:
    ValueError: Out of range float values are not JSON compliant: nan / inf.

    Before serialization, payload elements are recursively sanitized so that
    NaN and Inf values become `null` (None), and NumPy scalars/arrays are converted
    to native Python equivalents.
    """
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        clean_content = sanitize_for_json(content, nan_value=None)
        return json.dumps(
            clean_content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")


def patch_starlette_json_response() -> None:
    """
    Applies defense-in-depth monkey patch to Starlette's JSONResponse.render.
    Ensures that any direct imports of JSONResponse from starlette/fastapi
    throughout third-party or legacy code are also protected from NaN crashes.
    """
    def _safe_render(self: JSONResponse, content: Any) -> bytes:
        clean_content = sanitize_for_json(content, nan_value=None)
        return json.dumps(
            clean_content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")

    JSONResponse.render = _safe_render
