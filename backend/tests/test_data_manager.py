import pytest
import json
import io
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path

from app.core.security import extract_zip_safely, SecurityError, resolve_safe_path
from app.services.data_manager import DataManager


def test_csv_and_txt_loading(tmp_path):
    # 1. Create a dummy CSV file (row-based signals)
    csv_file = tmp_path / "test_signals.csv"
    raw_data = np.random.randn(10, 100).astype(np.float32)
    pd.DataFrame(raw_data).to_csv(csv_file, header=False, index=False)

    data, labels, cols = DataManager.load_raw_data(csv_file, delimiter=",", has_header=False)
    assert data.shape == (10, 100)
    assert labels is None
    assert len(cols) == 100
    assert np.allclose(data, raw_data, atol=1e-5)

    # 2. Create a tab-separated TXT file with header and label column
    txt_file = tmp_path / "test_signals.txt"
    df = pd.DataFrame(raw_data)
    df["label"] = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    df.to_csv(txt_file, sep="\t", header=True, index=False)

    data_txt, labels_txt, cols_txt = DataManager.load_raw_data(
        txt_file, delimiter="\t", has_header=True, label_col="label"
    )
    assert data_txt.shape == (10, 100)
    assert labels_txt is not None
    assert len(labels_txt) == 10
    assert labels_txt[1] == 1


def test_npy_npz_loading(tmp_path):
    # NPY
    npy_file = tmp_path / "test.npy"
    arr = np.ones((5, 50), dtype=np.float32) * 3.14
    np.save(npy_file, arr)

    loaded_npy, _, _ = DataManager.load_raw_data(npy_file)
    assert loaded_npy.shape == (5, 50)
    assert np.isclose(loaded_npy[0, 0], 3.14)

    # NPZ with labels
    npz_file = tmp_path / "test.npz"
    np.savez(npz_file, data=arr, labels=np.array([1, 2, 3, 4, 5]))
    loaded_npz, loaded_labels, _ = DataManager.load_raw_data(npz_file)
    assert loaded_npz.shape == (5, 50)
    assert len(loaded_labels) == 5


def test_zip_safety_and_extraction(tmp_path):
    # 1. Valid ZIP
    zip_path = tmp_path / "valid.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("signals/data1.csv", "1,2,3\n4,5,6")
        zf.writestr("signals/data2.csv", "7,8,9\n10,11,12")

    extract_dir = tmp_path / "extracted_valid"
    extracted = extract_zip_safely(zip_path, extract_dir)
    assert len(extracted) == 2
    assert (extract_dir / "signals/data1.csv").exists()

    # 2. Path Traversal ZIP (Zip Slip)
    bad_zip_path = tmp_path / "malicious.zip"
    with zipfile.ZipFile(bad_zip_path, "w") as zf:
        zf.writestr("../../etc/passwd", "root:x:0:0")

    with pytest.raises(SecurityError):
        extract_zip_safely(bad_zip_path, tmp_path / "extracted_bad")


def test_register_and_preview_dataset(tmp_path):
    csv_file = tmp_path / "register_test.csv"
    raw = np.arange(40, dtype=np.float32).reshape(4, 10)
    pd.DataFrame(raw).to_csv(csv_file, header=False, index=False)

    proj_id = "test-proj-dm"
    res = DataManager.register_dataset(
        project_id=proj_id,
        name="Registered Signals",
        source_path=csv_file,
        has_header=False
    )
    assert res["row_count"] == 4
    assert res["col_count"] == 10

    # Preview
    preview = DataManager.get_preview(res["id"], max_rows=2)
    assert len(preview["preview_rows"]) == 2
    assert preview["col_count"] == 10
    assert preview["summary_stats"]["max"] == 39.0


def test_semicolon_and_comma_decimal_loading(tmp_path):
    # Test European/Turkish CSV with semicolon delimiter and comma decimal points
    csv_file = tmp_path / "turkish_format.csv"
    content = "sig1;sig2;sig3\n1,25;2,50;3,75\n4,10;5,20;6,30\n"
    csv_file.write_text(content, encoding="utf-8")

    # Delimiter detection
    delim = DataManager.detect_delimiter(csv_file)
    assert delim == ";"

    # Loading with auto-detected/semicolon delimiter and comma decimal
    data, labels, cols = DataManager.load_raw_data(csv_file, delimiter=";", has_header=True)
    assert data.shape == (2, 3)
    assert cols == ["sig1", "sig2", "sig3"]
    assert np.isclose(data[0, 0], 1.25)
    assert np.isclose(data[0, 1], 2.50)
    assert np.isclose(data[1, 2], 6.30)


def test_transpose_single_column_signal(tmp_path):
    # Test 1D column signal transposed to row
    csv_file = tmp_path / "single_column.csv"
    values = [float(i) for i in range(100)]
    content = "\n".join(str(v) for v in values)
    csv_file.write_text(content, encoding="utf-8")

    data, labels, cols = DataManager.load_raw_data(csv_file, has_header=False, transpose=True)
    assert data.shape == (1, 100)
    assert np.isclose(data[0, 0], 0.0)
    assert np.isclose(data[0, 99], 99.0)


def test_nan_and_inf_safety_in_data_loading(tmp_path):
    csv_file = tmp_path / "nan_inf.csv"
    content = "1.0,NaN,3.0\n4.0,inf,-inf\n"
    csv_file.write_text(content, encoding="utf-8")

    data, labels, cols = DataManager.load_raw_data(csv_file, delimiter=",", has_header=False)
    assert data.shape == (2, 3)
    assert np.all(np.isfinite(data))
    assert np.isclose(data[0, 1], 0.0)  # NaN converted to 0.0
    assert np.isclose(data[1, 1], 0.0)  # inf converted to 0.0
    assert np.isclose(data[1, 2], 0.0)  # -inf converted to 0.0


def test_resolve_sequence_columns():
    cols = ["id", "age", "gender"] + [f"s{i}" for i in range(1, 101)] + ["target"]
    
    # 1. Wildcard pattern
    assert DataManager.resolve_sequence_columns(cols, "s*") == [f"s{i}" for i in range(1, 101)]
    assert DataManager.resolve_sequence_columns(["sig_0", "sig_1", "other"], "sig_*") == ["sig_0", "sig_1"]

    # 2. Range pattern s1-s100
    res_range = DataManager.resolve_sequence_columns(cols, "s1-s10")
    assert res_range == [f"s{i}" for i in range(1, 11)]

    # Range pattern sample_0-sample_511
    sample_cols = [f"sample_{i}" for i in range(20)]
    assert DataManager.resolve_sequence_columns(sample_cols, "sample_5-sample_10") == [f"sample_{i}" for i in range(5, 11)]

    # 3. Index range 0-511
    idx_cols = [f"col_{c}" for c in ("A", "B", "C", "D", "E")]
    assert DataManager.resolve_sequence_columns(idx_cols, "1-3") == ["col_B", "col_C", "col_D"]

    # 4. Regex pattern ^s\d+$
    regex_res = DataManager.resolve_sequence_columns(["s1", "s2", "s3", "status", "s999"], r"^s\d+$")
    assert regex_res == ["s1", "s2", "s3", "s999"]

    # 5. Comma-separated list
    assert DataManager.resolve_sequence_columns(cols, "s1, s5, s10") == ["s1", "s5", "s10"]

    # 6. Invalid / empty pattern
    assert DataManager.resolve_sequence_columns(cols, "") == []
    assert DataManager.resolve_sequence_columns(cols, "nonexistent*") == []


def test_preview_raw_file_schema_and_suggested_groups(tmp_path):
    csv_file = tmp_path / "multimodal_preview.csv"
    df = pd.DataFrame({
        "patient_id": [f"PID_{i:03d}" for i in range(20)],
        "age": np.random.randint(20, 80, size=20),
        "gender": ["M", "F"] * 10,
        "s1": np.random.randn(20),
        "s2": np.random.randn(20),
        "s3": np.random.randn(20),
        "s4": np.random.randn(20),
        "diagnosis": ["Healthy", "Arrhythmia"] * 10
    })
    df.to_csv(csv_file, index=False)

    preview = DataManager.preview_raw_file(csv_file, has_header=True)
    assert preview["total_rows"] == 20
    assert "suggested_sequence_groups" in preview
    assert len(preview["suggested_sequence_groups"]) >= 1

    group = preview["suggested_sequence_groups"][0]
    assert group["prefix"] == "s"
    assert group["pattern"] == "s1-s4"
    assert group["count"] == 4

    assert "column_schema" in preview
    schema_map = {item["name"]: item for item in preview["column_schema"]}
    assert schema_map["patient_id"]["inferred_role"] == "id"
    assert schema_map["patient_id"]["unique_count"] == 20
    assert len(schema_map["patient_id"]["sample_values"]) > 0

    assert schema_map["age"]["inferred_type"] == "numeric"
    assert schema_map["age"]["inferred_role"] == "input"

    assert schema_map["gender"]["inferred_type"] == "categorical"
    assert schema_map["gender"]["inferred_role"] == "input"

    assert schema_map["s1"]["inferred_type"] == "sequence"
    assert schema_map["s1"]["inferred_role"] == "input"

    assert schema_map["diagnosis"]["inferred_role"] == "output"
    assert schema_map["diagnosis"]["inferred_type"] == "categorical"


def test_register_dataset_with_schema_config_and_input_ids(tmp_path):
    csv_file = tmp_path / "sensor_data.csv"
    num_samples = 30
    df = pd.DataFrame({
        "sensor_id": [f"DEVICE_{i:04d}" for i in range(num_samples)],
        "temp": np.linspace(20.0, 35.0, num_samples, dtype=np.float32),
        "location": ["Zone_A", "Zone_B", "Zone_C"] * 10,
        "sig_1": np.sin(np.linspace(0, np.pi, num_samples)),
        "sig_2": np.cos(np.linspace(0, np.pi, num_samples)),
        "sig_3": np.sin(np.linspace(0, 2*np.pi, num_samples)),
        "anomaly": [0, 1] * 15
    })
    df.to_csv(csv_file, index=False)

    schema_config = {
        "columns": {
            "sensor_id": {"role": "id", "type": "categorical"},
            "temp": {"role": "input", "type": "numeric", "scaler": "standard"},
            "location": {"role": "input", "type": "categorical", "encoder": "onehot"},
            "sig_1": {"role": "input", "type": "sequence"},
            "sig_2": {"role": "input", "type": "sequence"},
            "sig_3": {"role": "input", "type": "sequence"},
            "anomaly": {"role": "output", "type": "categorical"}
        },
        "sequence_groups": [
            {
                "name": "vibration_signal",
                "pattern": "sig_1-sig_3",
                "normalization": "row_zscore"
            }
        ]
    }

    proj_id = "test-proj-schema"
    res = DataManager.register_dataset(
        project_id=proj_id,
        name="MultiModal Sensor Dataset",
        source_path=csv_file,
        has_header=True,
        schema_config=schema_config
    )

    dataset_id = res["id"]

    # 1. Input ID Traceability verification
    loaded_ids = DataManager.load_sample_ids(dataset_id)
    assert len(loaded_ids) == num_samples
    assert loaded_ids[0] == "DEVICE_0000"
    assert loaded_ids[5] == "DEVICE_0005"

    # 2. Multi-Modal Tensors verification
    tensors = DataManager.load_dataset_tensors(dataset_id)
    assert tensors["sample_ids"] is not None
    assert len(tensors["sample_ids"]) == num_samples

    # X_seq 3D Tensor: (batch_size, seq_len, 1)
    X_seq = tensors["X_seq"]
    assert X_seq is not None
    assert X_seq.shape == (num_samples, 3, 1)
    # Check row-wise z-score normalization on X_seq
    row0 = X_seq[0, :, 0]
    assert np.isclose(np.mean(row0), 0.0, atol=1e-4)
    assert np.isclose(np.std(row0), 1.0, atol=1e-3)

    # X_tabular 2D Tensor: temp (1 col) + location onehot (3 cols) = 4 cols
    X_tabular = tensors["X_tabular"]
    assert X_tabular is not None
    assert X_tabular.shape == (num_samples, 4)

    # y and labels: classification with 2 classes
    assert tensors["y"] is not None
    assert len(tensors["y"]) == num_samples
    assert tensors["labels"] is not None

    # Metadata & Pipeline State
    metadata = res["metadata"]
    assert metadata["has_sequence"] is True
    assert metadata["has_tabular"] is True
    assert metadata["task_type"] == "classification"
    assert metadata["recommended_activation"] == "softmax"
    assert metadata["recommended_loss"] == "sparse_categorical_crossentropy"
    assert metadata["recommended_metrics"] == ["accuracy"]

    pipeline_state = metadata["pipeline_state"]
    assert "numeric_stats" in pipeline_state
    assert "temp" in pipeline_state["numeric_stats"]
    assert "categorical_vocabs" in pipeline_state
    assert "location" in pipeline_state["categorical_vocabs"]
    assert pipeline_state["categorical_vocabs"]["location"]["vocab"] == ["Zone_A", "Zone_B", "Zone_C"]


def test_regression_output_and_crud(tmp_path):
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    csv_file = tmp_path / "regression_data.csv"
    num_samples = 15
    df = pd.DataFrame({
        "uid": [f"ID_{i}" for i in range(num_samples)],
        "s1": np.linspace(1.0, 10.0, num_samples),
        "s2": np.linspace(2.0, 20.0, num_samples),
        "target_amp": np.linspace(100.0, 500.0, num_samples)
    })
    df.to_csv(csv_file, index=False)

    schema_config = {
        "columns": {
            "uid": {"role": "id", "type": "categorical"},
            "s1": {"role": "input", "type": "sequence"},
            "s2": {"role": "input", "type": "sequence"},
            "target_amp": {"role": "output", "type": "numeric"}
        },
        "sequence_normalization": "global_minmax"
    }

    proj_id = "test-proj-crud"
    reg_res = client.post("/api/data/register", data={
        "project_id": proj_id,
        "name": "Regression Test",
        "file_path": str(csv_file),
        "has_header": "true",
        "schema_config": json.dumps(schema_config)
    })
    assert reg_res.status_code == 200
    ds = reg_res.json()
    ds_id = ds["id"]
    assert ds["metadata"]["task_type"] == "regression"
    assert ds["metadata"]["recommended_activation"] == "linear"
    assert ds["metadata"]["recommended_loss"] == "mse"

    # Test GET /api/data/{dataset_id}/sample_ids
    ids_res = client.get(f"/api/data/{ds_id}/sample_ids")
    assert ids_res.status_code == 200
    assert ids_res.json() == [f"ID_{i}" for i in range(num_samples)]

    # Test PUT /api/data/{dataset_id} (Rename)
    put_res = client.put(f"/api/data/{ds_id}", json={
        "name": "Renamed Regression Dataset"
    })
    assert put_res.status_code == 200
    assert put_res.json()["name"] == "Renamed Regression Dataset"

    # Test DELETE /api/data/{dataset_id}
    del_res = client.delete(f"/api/data/{ds_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"

    # Confirm 404 after delete
    preview_res = client.get(f"/api/data/preview/{ds_id}")
    assert preview_res.status_code == 404


