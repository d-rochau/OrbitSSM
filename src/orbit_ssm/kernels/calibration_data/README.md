# Shipped kernel calibration files

One JSON file per (family, GPU name): `<family>/<gpu slug>.json`, families `ssd_mimo` (SISO scan forward /
backward / rotation adjoint), `ssd_mimo_fused` (general-route backward kernels), `ssd_mimo_chunkpar`
(chunk-parallel crossover thresholds), `ssd_mimo_levers` (calibrated speed levers per phase), `ssd_mimo_route` (the structural route of a rotating
R=O=1 bf16 call: SISO vs ROTSPLIT vs in-kernel general, by occupancy and length, eligibility-gated; `tools/calibrate_routes.py`). A file is used
only when its provenance matches the GPU name, the Triton version and the family's kernel code hash; a stale
file is ignored by name (never silently).

* **Shape set.** Every file is measured on the one shared shape set, `src/orbit_ssm/kernels/shape_sets.py`
  (3 size classes x balanced/long/wide/batch at equal work, plus the LM cells); cells that do not fit are
  listed in `provenance.dropped_shapes`, deviations in `provenance.notes` (R4O4 on the LM cells only; the six
  fp32 real R1O1 fallback cells feed only `fused_kb1`).
* **Modes** (`SSD_MIMO_CALIBRATION`, default `on`):
  * `on` (default, production): the first use of a tuning key searches ONLY the calibrated **light set**
    (2-4 configs per kernel x group, worst case <= 1.02 over the group's shapes) and persists the winner
    only if it beats the calibrated fixed config by more than the measured noise.
  * `pinned`: the single calibrated **fixed config** of the call's group, scoped by the recorded chunk T, no
    search anywhere. **For reproducible tests and gates, NOT for production speed**: the fixed config is the
    minimax over the shapes and a near-tie can flip between calibration passes (measured: a few kernel x
    group entries differ by 1.05-1.09 between two passes while the light sets agree within 1.014). The
    fixed config is therefore chosen by minimax over the POOLED timings of pass 1 + pass 2 where both measured
    (`provenance.fixed_basis`), but the light set is what production serves.
  * `off`: the old grids, uncalibrated.
* **Fail fast.** Benchmark and gate jobs run `tools/check_calibration.py` first: a missing / stale file, or a
  kernel without an entry that the file does not list in `unlaunched_kernels`, aborts the job (an
  uncalibrated measurement would pay light-grid autotune compiles and time a configuration production does
  not run).
* **Regenerate** after any change of a calibrated kernel's source: `tools/calibrate_kernels.py`,
  `tools/cp_calibrate.py`, `tools/calibrate_levers.py` (see their docstrings; `--part i/n` splits a family over
  GPUs of one model, `--from-raw a,b --gpu NAME --sms N` derives offline, `--pool` pools a second pass).
* **fp32 axis.** fp32 IO (tf32 operands: torch's default dtype) is calibrated separately -- every MMA operand is
  twice as wide, and 16-bit-tuned configs at fp32 hit a ptxas register collapse (32 registers + kilobytes of stack).  Groups
  `<group>_fp32_t<T>` (one per chunk T in {32, 64}; `complex_R1O1_fp32_t*` in `ssd_mimo`, `real_R1O1_fp32_t*` /
  `complex_R1O1_fp32_t*` in `ssd_mimo_fused`) are measured on the fp32 cells of the shared set and MERGED into the existing
  files (`tools/calibrate_kernels.py --from-raw ... --merge-groups ...`, `calibration.merge_groups`; the 16-bit groups and
  defaults are untouched).  The structural fp32 choices are keys with an fp32 suffix in the same files: the real route's
  `real_r1o1|T|<kind>|fp32` (+ `|Trule|...|fp32`, the real-decay calibration at `--dtype fp32`), the rotating route's
  `route|<grad|nograd>_fp32|<group>|...` incl. the SISO chunk `siso_T` (`tools/calibrate_routes.py --dtype fp32`), and the
  lever group `real_R1O1_fp32`.  A GPU file without fp32 entries keeps the 16-bit behaviour for fp32 calls.  A kernel-source
  change that is bit-identical for every stored group (a constexpr branch only fp32 compiles) is re-stamped, not recalibrated,
  with its evidence (`tools/restamp_calibration.py`, `provenance.restamped`).
* **Real R = O = 1 (Mamba-2 class) group `real_R1O1`** (the real-decay calibration, stages `elig` / `screen` / `choose` /
  `ablate` / `verify` / `decide`): a per-GPU STRUCTURAL choice of the chunk length T in {32, 64} (thresholds
  `real_r1o1|T|train_fwd`, `|train_bwd`, `|fwd_only` of `ssd_mimo_chunkpar`; read by `_scan_chunk(real=True)` /
  `_scan_chunk_bwd(real=True)`; `T_fwd != T_bwd` is the `chunk_size_bwd` pair) and five lever units in
  `ssd_mimo_levers` (`real_gluon_kbc`, `real_gate_dy_io`, `real_kc_shared_cb`, `real_dg_lo` backward; `real_cp_g_lo` forward;
  the former opt-in env levers `SSD_MIMO_GLUON_KBC`, `_GATE_DY_IO`, `_BWD_KC_SHARED_CB`, `_BWD_DG_LO`, `_CP_G_LO`, which still
  force on/off when set).  Eligibility is the fp64 oracle (`ssd_check.py` m2exact-like + C11 arms, `ssd_check_saturated.py`
  plain / reset / large_phase, 5 draws, signed-bias gates), per unit alone and as a combination.  The fused / ssd kernel
  configs of the group are recalibrated at the chosen T (`calibration_fused._real_r1o1_chunks`).
  The chunk-parallel forward crossover is measured against the route production takes otherwise (the GENERAL fused forward) and
  separately for a training call (`fwd`, at T_fwd) and a forward-only call (`fwd_nograd`, at the forward-only T; falls back to `fwd`
  when absent) by `tools/cp_calibrate.py` (`CP_CAL_CHUNK` / `CP_CAL_CHUNK_FWD`).  A `(T_fwd, T_bwd)` pair is screened too but
  `chunk_size_bwd` is refused by name on the real SISO / chunk-parallel forward (`SALL=io`), so every shipped structure is T_fwd = T_bwd
  for training.  Order of a recalibration: `elig` -> `screen` -> `choose` -> `ablate` -> `verify` -> `decide`, then recalibrate the
  fused / ssd kernels and the crossover at the chosen T with the chosen levers ON (`run_recal.sh` in the run dir), then `final`.


* **MH scan at P = 1 (`mh/`, family `mh`, key `p1_cells`; `tools/calibrate_mh_p1.py`).** The exact Mamba-1 path (HM `m1exact`)
  has a PER-CELL table, not a fixed config: its optimum moves with every one of (B, H, N, L, IO dtype, feature class), so the
  file holds, per cell of the shared shape set at P = 1 (H = d_inner) x N in {16, 128} x {bf16, fp32}, the best measured
  config of the inference forward (`fwd`), the checkpoint-writing forward (`fwd_trn`) and the backward (`bwd`), as MH pin
  strings.  Feature class = the Mamba-1 mixer's call (`m1`: dt softplus + dt bias + D skip in the kernel).  Served in
  `mh_forward` / `mh_backward` as a pin; any other cell, an explicit `HYBRID_MH_AUTOTUNE=full`, a `fwd_config` /
  `bwd_config` pin, `HYBRID_MH_CALIBRATION=off`, or a stale file (GPU / Triton / kernel code hash) runs the runtime tuner as
  before.  A cell is written only if the production path on the table is <= 1.02 x the production path on the tuner
  (inference forward and forward+backward, same process, interleaved; recorded as `ratio`), the chosen config agrees with the
  tuner's output at the tuner's own tolerance, is finite at the extreme decay rate and peaks at <= 1.10 x the tuner's memory.
  Stats: `calibration_mh.P1_STATS` (hit / miss).  No `check_calibration.py` requirement: a GPU without a table is not an error.
