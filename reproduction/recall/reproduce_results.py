
from pathlib import Path
import runpy
import sys
root = Path(__file__).resolve().parent / 'views'
files = sorted((root / 'sections').glob('generated_mq_*.tex'))
assert len(files) == 3
expected = {p: p.read_bytes() for p in files}
sys.path.insert(0, str(root / 'figures'))
runpy.run_path(str(root / 'figures/mqmkvar_results.py'), run_name='__main__')
for p, content in expected.items():
    assert p.read_bytes() == content, 'Regeneration differs: ' + p.name
print('PASS: frozen per-seed statistics and all 3 corrected recall passages reproduce exactly')
