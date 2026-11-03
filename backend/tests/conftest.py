import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Add backend directory to sys.path so 'app' is cleanly importable
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Use an isolated temporary storage directory during tests so pytest never
# pollutes or overwrites the production /workspace/SignalML/storage/signalml.db
_TEST_ROOT = Path(tempfile.mkdtemp(prefix="signalml_test_"))
_TEST_STORAGE = _TEST_ROOT / "storage"
_TEST_DATA = _TEST_ROOT / "data"
_TEST_STORAGE.mkdir(parents=True, exist_ok=True)
_TEST_DATA.mkdir(parents=True, exist_ok=True)

os.environ["SIGNALML_STORAGE_DIR"] = str(_TEST_STORAGE)
os.environ["SIGNALML_DATA_DIR"] = str(_TEST_DATA)


@atexit.register
def _cleanup_test_root():
    shutil.rmtree(_TEST_ROOT, ignore_errors=True)
