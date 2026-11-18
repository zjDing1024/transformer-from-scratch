import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, PYTHONPATH=str(ROOT / "src"))


def test_launcher_import_is_standard_library_only():
    code = "import sys; import transformer_diagnostics.cli; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], env=ENV, check=True)


def test_cli_offline_parity_then_refuses_overwrite(tmp_path):
    path = tmp_path / "portable.json"
    args = [
        sys.executable,
        "-m",
        "transformer_diagnostics.cli",
        "--parity-only",
        "--output",
        str(path),
    ]
    result = subprocess.run(args, env=ENV, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(path.read_text())
    assert report["parity"]["status"] == "passed"
    assert report["memory_status"] == "not_requested"
    assert report["memory_workers"] == []
    assert report["lightweight_launcher_no_torch"]
    assert not report["training_performed"] and not report["timing_performed"]
    original = path.read_bytes()
    repeated = subprocess.run(args, env=ENV, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert "output exists" in repeated.stderr
    assert path.read_bytes() == original
