"""The built wheel carries every file the installed package reads at run time."""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "orbit_ssm"
#: non-Python files under the package that nothing loads (documentation inside the data directory)
_NOT_LOADED = {"README.md"}


def _build_wheel(tmp_path: Path) -> zipfile.ZipFile:
    """Build with the project's own backend from a scratch copy, so the source tree gains no build/ or egg-info."""
    tree = tmp_path / "tree"
    tree.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        if (ROOT / name).exists():
            shutil.copy2(ROOT / name, tree / name)
    shutil.copytree(ROOT / "src", tree / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    out = tmp_path / "dist"
    out.mkdir()
    code = "from setuptools import build_meta; print(build_meta.build_wheel(%r))" % str(out)
    done = subprocess.run([sys.executable, "-c", code], cwd=tree, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    wheels = list(out.glob("*.whl"))
    assert len(wheels) == 1, wheels
    return zipfile.ZipFile(wheels[0])


@pytest.mark.slow
def test_wheel_contains_every_runtime_data_file(tmp_path):
    """pins: the wheel built from the tree contains every module of `orbit_ssm` and every non-Python file the code
    loads at run time (the per-device calibration JSON under kernels/calibration_data/<family>/, read through
    __file__-relative paths by kernels/calibration.py and the calibration_* modules), one for one with the source tree;
    a data file the package-data glob misses would be absent from every installed copy while passing from the source
    tree."""
    names = set(zipfile.ZipFile.namelist(_build_wheel(tmp_path)))
    source_files = {p.relative_to(PACKAGE.parent).as_posix() for p in PACKAGE.rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    data = {n for n in source_files if not n.endswith(".py") and Path(n).name not in _NOT_LOADED}
    assert len([n for n in data if n.endswith(".json")]) >= 15, sorted(data)
    families = {Path(n).parts[3] for n in data if "calibration_data" in n}
    assert {"mh", "ssd_mimo", "ssd_mimo_fused", "ssd_mimo_chunkpar", "ssd_mimo_levers", "ssd_mimo_route"} <= families
    missing = sorted(n for n in source_files - names if n.endswith(".py") or n in data)
    assert missing == [], "files of the source tree absent from the wheel:\n" + "\n".join(missing)
