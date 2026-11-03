import pytest
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

