SSD-MIMO and HybridMamba: supplementary implementation
====================================================

Start here
----------
This compact supplement includes both models, their GPU kernels and references,
shared layers, a small correctness test, and selected reproduction tools/results.
The Python package name is orbit_ssm.
Datasets, checkpoints, campaign logs and tuning-attempt archives are not bundled.

Install on Linux with Python 3.12. The measured stack was PyTorch 2.14.0+cu130,
Triton 3.8.0, causal-conv1d 1.6.0 and mamba-ssm 2.3.2.post1. First install compatible
PyTorch and Triton packages; then, from this extracted directory:

  python -m pip install --no-deps .
  python -m pip install pytest
  python -m pytest -q tests/test_supplement_smoke.py

The test checks outputs, gradients and state continuation. CUDA checks skip when
no GPU is available. For official-baseline comparisons, install causal-conv1d
and mamba-ssm against your PyTorch/CUDA stack. Other optional experiment
requirements are in requirements-experiments.txt. Third-party packages and data
must be acquired separately under their own terms.

Contents
--------
src/           Model, kernel, reference and calibration code.
tests/         One small smoke-test file.
benchmarks/    Shared timing and memory measurement tools for the two families.
experiments/   Recall task/configurations and language-model training/evaluation.
reproduction/ Selected table inputs, generators and paper measurement harness.

Rebuild selected reported tables (CPU, no training)
--------------------------------------------------
  python reproduction/performance/reproduce_main.py
  python reproduction/recall/reproduce_results.py
  python reproduction/lm/tools/lm_eval_task_tables.py

These regenerate/check the main and compact performance tables, three recall
summaries, and downstream LM task summaries from included results. They are not
new experimental runs. The Markdown files under reproduction/lm/records are
source result tables used by the LM generator, not internal project notes.
The full historical campaign and every appendix diagnostic are not included.

Measure performance (CUDA)
--------------------------
Run from this directory. Enumerate available benchmark grids first:

  python -m benchmarks.bench_kernels --list-grids
  python -m benchmarks.bench_ssms --list-grids

The paper harness supports SSM-core and full-mixer measurements. For example,
enumerate the five-shape SSM-core cost workload without running GPU measurements:

  PYTHONPATH=. HMB_TAGS=scope python reproduction/performance/harness/fin_run.py bench_kernels --grid kernel_scope_n128 --precisions bf16 --passes fwd_bwd --dry-run

For measurement, replace --dry-run with --budget paper --rounds 3 --out results.jsonl.
Use --arms to select a smaller subset. Inspect --help and the grid listing before
launching larger sweeps. Reference checks against the official Mamba variants
are also available via reproduction/numerics/tools/p1_numerics.py --help.

Scope and caveats
----------------
Paper performance uses H100 PCIe, bfloat16 and warmed measurements. Kernel/core
and full-mixer scopes differ. Total retained nonparameter memory for backward
and incremental peak allocation are distinct metrics; neither is whole-model
training memory. Official Mamba-3 fp32-input comparisons use bf16 operands
internally; its measured provider uses softplus activation.

The HM implementation contains the measured backward temporary-buffer reduction.
Five-shape results use this implementation; retained-for-backward memory is
unchanged. Some rotating P32 mixers become slower. Additional LM shapes and
small/medium/fp32 HM views retain the earlier schedule, as labelled in the paper.
SSD/Mamba-2/Mamba-3 results are unchanged. The SSM-core fp64 checker includes the
correction that avoids repeating B/C convolution on post-convolution inputs.

Calibration is hardware-specific. A legacy HM calibration remains rejected by
its validity checks. A Blackwell N=96 tiled backward configuration can exceed
shared-memory capacity. See the paper's numerical appendix for precision and
gradient limitations. The smoke test does not certify every configuration.

Selected recall training configurations can be checked without launching training:

  PYTHONPATH=. python reproduction/recall/check_training_configs.py

This checks the included final SSD-MIMO and HybridMamba campaign configurations.
Other historical campaign manifests are not included. For a direct Lightning run,
pass a built spec to experiments.lightning.run_lightning_trial with
report_to_ray=False. Training additionally needs Zoology, Lightning and
torchmetrics; Ray is optional and only needed with report_to_ray=True.
See experiments/tasks/mqmkvar/builder.py and experiments/mqmkvar/. experiments/lm/ contains tokenization, training and evaluation
entry points. Obtain FineWeb-Edu and evaluation datasets separately. No full
training rerun is claimed by this compact packaging check.

License: Apache-2.0; see LICENSE. No Git history or private release receipts are
included. Model and kernel Python source is unchanged from the validated release.
Revision references use anonymous labels. Benchmark metadata omits hostnames,
checkout paths and Git identities; local data defaults use ./data/.
