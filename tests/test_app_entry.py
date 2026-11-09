"""`streamlit run app/main.py` puts only app/ on the import path, unlike pytest.

The AppTest smoke tests run inside pytest (repo root on the path), so they missed a
`ModuleNotFoundError: No module named 'app'` at real start-up. This runs the entry script in a
fresh interpreter with the repo root removed from the path, the way Streamlit does.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCRIPT = """
import runpy, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
sys.path = [p for p in sys.path if p and Path(p).resolve() != root]
sys.path.insert(0, str(root / "app"))
runpy.run_path(str(root / "app" / "main.py"), run_name="__main__")
print("ENTRY_OK")
"""


def test_entry_script_imports_with_only_app_dir_on_the_path():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(ROOT)],
        cwd=ROOT / "app",
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert "ENTRY_OK" in result.stdout, result.stderr[-2000:]
