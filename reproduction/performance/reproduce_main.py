
from pathlib import Path
import runpy
import sys
root = Path(__file__).resolve().parent / 'views'
tables = root / 'tables/bench'
sys.path.insert(0, str(root / 'figures'))
names = ['table3_parity_shapes.tex', 'table4_cost_readable.tex', 'compact_inference.tex', 'compact_training.tex', 'compact_scope_memory.tex', 'compact_mixer_memory.tex', 'compact_recovery_extras.tex', 'compact_small_summary.tex', 'compact_table_notes.tex', 'table1a_m1_current_detail.tex', 'hm_workspace_tradeoff.tex']
expected = {n: (tables / n).read_bytes() for n in names}
runpy.run_path(str(root / 'figures/bench_tables.py'), run_name='__main__')
for n, data in expected.items():
    assert (tables / n).read_bytes() == data, 'Regenerated view differs: ' + n
print('PASS: all ' + str(len(names)) + ' main/compact performance views reproduce exactly')
