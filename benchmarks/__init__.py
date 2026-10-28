"""Measurement package — the third peer of ``tests/`` and ``src/orbit_ssm/``.

`tests/` proves the kernels are RIGHT; `benchmarks/` proves how fast they
are.  Same three-module shape as the collected test suite:

    bench_kernels.py   kernel entry points vs the reference kernels
    bench_ssms.py      the production mixers vs the original Mamba layers

plus the shared spine — `measure.py` (THE measurement core: the protocol,
the budgets, the precision columns, the round-robin driver and every
metric), `harness.py` (the `Cell` contract, the record schema and the
grid CLI, all timing delegated to `measure`), `shapes.py` (the named grids,
each cross-referenced to its source of truth) and `baselines.py`
(reference-kernel adapters, each with an `available()` probe).

Nothing here is imported by `src/orbit_ssm/`, and pytest never collects it
(`pyproject.toml` pins `testpaths = ["tests"]`).  The package lives under `src/` (src layout):
install it once with `pip install -e .` (or export `PYTHONPATH=$PWD/src`) before running
`python -m benchmarks...` or `python -m experiments...` from the repository root.  Run a module directly:

    python -m benchmarks.bench_kernels --grid smoke --out /tmp/x.jsonl

See `benchmarks/README.md` for the protocol and the GH200 recipe.
"""

__all__ = ["measure", "harness", "shapes", "baselines"]
