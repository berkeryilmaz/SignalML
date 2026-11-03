import math
import json
import numpy as np
from app.core.responses import SafeJSONResponse, sanitize_for_json


def test_sanitize_primitives_and_nans():
    payload = {
        "nan_val": float("nan"),
        "posinf": float("inf"),
        "neginf": float("-inf"),
        "valid_float": 42.5,
        "valid_int": 10,
        "valid_str": "signal",
        "nested": {
            "np_nan": np.nan,
            "np_float": np.float32(1.23),
            "np_int": np.int64(99),
            "arr": np.array([1.0, np.nan, 3.0])
        }
    }
    cleaned = sanitize_for_json(payload)
    assert cleaned["nan_val"] is None
    assert cleaned["posinf"] is None
    assert cleaned["neginf"] is None
    assert cleaned["valid_float"] == 42.5
    assert cleaned["nested"]["np_nan"] is None
    assert cleaned["nested"]["arr"] == [1.0, None, 3.0]


def test_safe_json_response_render():
    response = SafeJSONResponse(content={"silhouette": float("nan"), "coords": [float("inf"), 1.0]})
    rendered_bytes = response.render({"silhouette": float("nan"), "coords": [float("inf"), 1.0]})
    decoded = json.loads(rendered_bytes.decode("utf-8"))
    assert decoded["silhouette"] is None
    assert decoded["coords"] == [None, 1.0]


def test_sanitize_complex_numbers():
    payload = {
        "c_scalar": complex(3.0, 4.0),
        "c_nan": complex(float("nan"), 2.0),
        "np_c": np.complex128(1.5 - 2.5j)
    }
    cleaned = sanitize_for_json(payload)
    assert cleaned["c_scalar"] == {"real": 3.0, "imag": 4.0}
    assert cleaned["c_nan"] == {"real": None, "imag": 2.0}
    assert cleaned["np_c"] == {"real": 1.5, "imag": -2.5}
    # Test strict json dumps with allow_nan=False
    dumped = json.dumps(cleaned, allow_nan=False)
    assert dumped is not None
