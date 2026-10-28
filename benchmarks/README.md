# `benchmarks/`: timing the kernels and mixers

`tests/` checks that the kernels are right; `benchmarks/` measures how fast they are.  Nothing here is imported by
`src/orbit_ssm/`, and pytest does not collect it.  Every number a timing table quotes comes from these tools.

```
benchmarks/
  measure.py     the measurement core: protocol, budgets, precision columns, round-robin driver, metrics
  harness.py     the Cell contract, the record schema, the grid command line (all timing is delegated to measure.py)
  shapes.py      the named grids and where each shape comes from
  baselines.py   adapters for the reference kernels and layers (and a probe of which ones are installed)
  cases.py       the case families the mixer grids are built from
  bench_kernels.py  the scan kernels against the reference kernels
  bench_ssms.py     the full mixers against the original Mamba layers
  route_record.py   records which kernel route a call took
```

## Quick start

A CUDA GPU is needed for timing; `--list-grids` and `--dry-run` run on a CPU.

```bash
python -m benchmarks.baselines                         # which reference implementations are installed
python -m benchmarks.bench_kernels --list-grids
python -m benchmarks.bench_kernels --grid smoke --out /tmp/smoke.jsonl     # plumbing check, never quote it
python -m benchmarks.bench_kernels --grid ssd_parity26 --dry-run           # the cells by comparison group, no timing
python -m benchmarks.bench_kernels --grid paper:ssd_parity26 --out /tmp/p.jsonl
python -m benchmarks.bench_ssms --grid ssm_lm_spine --precisions bf16 --out /tmp/m.jsonl
```

Results go to `$HYBRID_SSM_RESULTS` (default `~/hybrid-ssm-results`) unless `--out` is given.  Both commands share
their options: `--grid`, `--budget {smoke,full,deep,paper}`, `--precisions ieee,tf32,bf16`, `--passes fwd,fwd_bwd`,
`--arms`, `--group-mode {auto,resident,rebuild}`, `--dry-run`, `--warm-only` (compile every cell once, write no
timings), `--cells {full,paper}`, `--append`.  `bench_ssms` adds `--profile` (self-CUDA time split per cell) and
`--graphed` (rows that replay one captured CUDA graph of the whole step).

## Grids

`bench_kernels --list-grids`: `smoke`, `mh_siso`, `mh_features`, `paper_c3`, `paper_c3_nscale`, `paper_c3_lscale`,
`ssd_parity26`, `ssd_parity_m3`, `ssd_parity_m3_eq`, `ssd_long_l`, `ssd_mimo_gates`, `ssd_features`,
`kernel_shared_set`.  `bench_ssms --list-grids`: `smoke`, `ssm_mqmkvar`, `ssm_lm_spine`, `ssm_shared_set` and its
size variants.  `paper_c3*` are the kernel-timing tables and figure of the paper; `ssd_parity*` compare the SSD
kernel with the Mamba-2 and Mamba-3 kernels on the same 26 shapes; `mh_*` time the Hybrid-Mamba kernel against the
Mamba-1 selective scan; `ssd_features` prices each optional input of the SSD kernel against its own absence.

`--cells paper` (or `--grid paper:<grid>`) keeps only the cells a paper table quotes, and `--budget paper` times
them with a shorter window.  A grid without a paper subset refuses `--cells paper` instead of running in full.

## Protocol

1. **Round-robin in one process, with a canary.**  The variants of a comparison group are measured in turns, with a
   fixed 4096^3 matmul at the head of each round.  The quoted number is the median over rounds and every record
   carries `canary_spread`; above 1.15 the group was contended and must not be quoted.  A group is the comparison
   unit (the arms one ratio compares), not the shape.  `--dry-run` prints the group census.
2. **Real random upstream gradients**, never `y.sum()`: a sum loss hands strided kernels an expanded-ones `dY` that
   reads as a single value per row and flatters them.
3. **Forward rows run under `no_grad()`**, the inference path; a forward row that builds a graph times something
   nobody runs.
4. **Call the family entry point** (`ssd_mimo_scan`, `mh_scan`), which refuses a configuration outside the fused
   envelope instead of silently taking a slower path.
5. **Three precision columns**, `ieee`, `tf32`, `bf16`.  At the kernel level they are operand casts plus the
   kernel's `precision=` argument; at module level bf16 is `torch.autocast` over fp32 parameters.
6. **Memory is the per-cell delta**, `mem_delta_bytes = peak - baseline`; the process-wide peak stays as provenance.

Timing tiers: forward rows use `triton.testing.do_bench` (its `rep` is a time budget in ms); forward+backward rows
use CUDA events around N iterations with one sync at the end, which avoids the L2 flush of `do_bench`.  Budgets:
`smoke` (3 rounds, small), `full` (3 rounds, 500 ms / 30 iterations; the default), `deep` (5 rounds, 50 iterations),
`paper` (2 rounds, 150 ms / 10 iterations; warmup is the same as `full`).

## Records

One JSON line per cell: run and host provenance (`gpu`, `torch`, `triton`, `cuda`, `git_sha`, ...), the cell
(`family`, `arm`, `shape`, `group`, `precision`, `pass`, `tier`), the timing (`ms_median`, `ms_min`, `ms_max`,
`samples`, `rounds`, `iters`), the canary fields, the memory fields and `error`.  A cell that fails to build or run
is still written, with `error` set and the timings null; a missing baseline is a recorded skip, not a crash.
`ms_median` is wall time per call.  The parity grids also carry `ms_cuda_self`, the summed self-CUDA time of the same
call, so a reader can tell per-call wall time from pure GPU work.

## Calibration tools

The kernels read per-GPU calibration tables (`src/orbit_ssm/kernels/calibration_data/`, described in its `README.md`).
To refit them on a new GPU, use the tools in `tools/` (`calibrate_kernels.py`, `calibrate_routes.py`,
`calibrate_levers.py`, `calibrate_mh_p1.py`, `cp_calibrate.py`) and check the result with
`tools/check_calibration.py`; they time through the cells of this package.
