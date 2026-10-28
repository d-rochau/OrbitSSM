# Environment and installation

## Supported stack

Linux x86_64, Python 3.12 or newer, PyTorch 2.14.x, Triton 3.8.x, and a GPU of
compute capability sm_80 or higher. The kernels were validated on A100 (sm_80),
H100 PCIe (sm_90) and RTX PRO 5000 Blackwell (sm_120). This is an evidence
record, not a guarantee for every shape or GPU; other architectures (including
ARM) need separate validation. The package metadata pins `torch>=2.14,<2.15`
and `triton>=3.8,<3.9`. CPU execution works without a GPU but still imports
Triton.

## Install with uv

```bash
git clone https://github.com/d-rochau/OrbitSSM.git && cd OrbitSSM
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch==2.14.* --index-url https://download.pytorch.org/whl/cu130
uv pip install triton==3.8.*
uv pip install --no-deps .
uv pip install "pytest>=9" PyYAML numpy pytest-timeout   # the test extras
```

`cu130` wheels need a CUDA 13 driver; pick the index matching your driver
otherwise. `--no-deps` keeps the installer from replacing the torch build you
chose.

The same with conda and pip:

```bash
git clone https://github.com/d-rochau/OrbitSSM.git && cd OrbitSSM
conda create -n orbit python=3.12 && conda activate orbit
python -m pip install torch==2.14.* --index-url https://download.pytorch.org/whl/cu130
python -m pip install triton==3.8.*
python -m pip install --no-deps .
python -m pip install "pytest>=9" PyYAML numpy pytest-timeout
```

The torch and Triton wheels are several GB. If your home directory has a small
quota, keep the environment and the uv cache on a large filesystem:

```bash
export UV_CACHE_DIR=/big/fs/uv-cache
export UV_PYTHON_INSTALL_DIR=/big/fs/uv-python
export UV_LINK_MODE=hardlink      # cache and venv on the same filesystem
```

## C++20 compiler

Needed only for:

- `torch.compile` on CPU (Inductor generates C++20 code); gcc 10 or newer.
- Building `causal-conv1d` or `mamba-ssm` from source (levels 2 and 3 below).

Not needed for installing torch or this package, for the GPU kernels (Triton
compiles them), or for eager CPU execution. With an older system compiler,
set `CXX` to a newer g++ or load one (for example a gcc toolset) before
running. The CPU tests that use `torch.compile` skip with a message when no
C++20 compiler is found; GPU compile tests are unaffected.

## What to install: three levels

**1. Core (all users).** torch, Triton and this package, as above. No compiler
and no CUDA toolkit are needed: the GPU kernels are Triton, compiled at run time.
Every default route of SSD-MIMO and HybridMamba runs on this level, including
the causal convolution and the gated RMSNorm (our own Triton kernels). Most of
the test suite runs here; tests that need a package below skip by name.

**2. The `causal-conv1d` package (optional, not needed for speed).** Used only
when a layer is built with `kernel_options={"conv_kernel": "causal_conv1d"}`
(refused by name when the package is missing). Our Triton convolution ties it:
in a full layer training step the time ratio ours / package is 0.99 to 1.004 on
A100 and H100, bf16 and fp32 (table in the README). There is no wheel for this
torch, so it builds from source and needs a C++20 compiler plus a CUDA toolkit
(`nvcc`) whose major version matches the CUDA of your torch build:

```bash
python -m pip install --no-build-isolation causal-conv1d   # 1.6.0 measured
```

**3. Official baselines (benchmarks and parity tests only).** `mamba-ssm`
(2.3.2.post1 measured) together with `causal-conv1d` and the Python packages of
the `baselines` extra. Our layers never import it by default; it provides the
official Mamba-1/2/3 arms of the benchmarks (`benchmarks/baselines.py`,
`orbit_ssm.presets`), the tests that copy official weights into our mixers, and
two explicit opt-ins (`kernel_options` `norm_kernel="mamba_ssm"`,
`m3_kernel="upstream"`). Build it from source against the installed torch
(C++20 compiler and matching `nvcc`, as in level 2):

```bash
python -m pip install --no-build-isolation causal-conv1d mamba-ssm
python -c "import causal_conv1d, mamba_ssm; print(mamba_ssm.__version__)"   # check
```

The CUDA build is required for all three Mamba versions, not only for Mamba-1.
Mamba-2 and Mamba-3 are Triton kernels, but `import mamba_ssm` (and therefore
any `mamba_ssm.modules.*` import) loads Mamba-1's compiled `selective_scan_cuda`
extension unconditionally; an install that skips the CUDA build
(`MAMBA_SKIP_CUDA_BUILD=TRUE`) leaves the package unimportable. What each
baseline then uses:

| baseline | needs beyond an importable `mamba_ssm` |
|---|---|
| Mamba-1 | `causal-conv1d` on its fused path |
| Mamba-2 | `causal-conv1d` on its default fused path |
| Mamba-3 SISO | nothing |
| Mamba-3 MIMO | TileLang (not used by our benchmarks) |

The Mamba-3 provider must be state-spaces/mamba revision
`e9594ce1c732d97440f0332fdc43170a2294dbfa` or later (the first one carrying
`heavy_tail_activation` in `mamba_ssm/modules/mamba3.py`). Mamba-3 providers
differ in their A activation; verify copied-weight outputs before interpreting
comparison timings.

Do not transfer compiled extensions or Triton/Inductor caches across GPU
architectures or torch builds.

## Running the tests

From the source root (pytest puts `src` on the path, so this tests the source
tree):

```bash
python -m pytest -q -m "not gpu" tests                   # CPU part (about 3 minutes)
python -m pytest -q tests                                # GPU and CPU (about 11 minutes on one GPU, cold cache)
python -m pytest -q --require-gpu tests                  # the same; fails if no GPU is usable
python -m pytest -q -m "not gpu" --profile smoke tests   # CPU smoke: one cell per test
```

The default profile is `user`: one minimal cell per test function and the
contract suite over its base option spaces (`smoke`, `compact` and `full` add
cells). Without a GPU the CUDA cases skip; a skipped case is not a validation
pass. The first GPU run compiles Triton kernels; set `TRITON_CACHE_DIR` to a
persistent directory to keep them.

## A GPU that is not one of the validated three

Launch configurations, chunk lengths, route choices and speed switches are
calibrated per GPU model (`src/orbit_ssm/kernels/calibration_data/`). On another
GPU nothing is refused and results stay within the same tolerances: missing
tables fall back to generic launch configurations, the fixed chunk rule, the
SISO route and switches off, which is slower and pays autotuning compiles on
first use. Run this once on such a GPU:

```bash
python tools/calibrate_kernels.py --all --list   # the stages, their commands and expected times; runs nothing
python tools/calibrate_kernels.py --all          # every table of this GPU, in dependency order
```

It measures the chunk structure first (the chunk-parallel crossover and, where
the tools are present, the chunk lengths), then the launch configurations at
those chunk lengths, the rotating-route table, the speed switches and the MH
P = 1 table, and finishes with `tools/check_calibration.py`. The results go to
the local calibration files (`$SSD_CALIBRATION_DIR`, else
`~/.cache/orbit_ssm/calibration`), which take precedence over the shipped ones
and become inert when the GPU, Triton version or kernel code changes. Expect
most of a day on one GPU; `--list` gives the per-stage times measured on the
validated GPUs. The run is resumable (each finished stage leaves a marker in
`--work`, default `~/.cache/orbit_ssm/calibrate_run`); `--only` / `--skip`
select stages or groups, `--keep-going` continues past a failed stage.

Smaller checks: `python examples/quickstart.py` (CPU reference example) and
`python -m benchmarks.bench_kernels --list-grids`.

## Check which copy is used

To confirm the installed package, not the source tree, is imported, run from
outside the checkout with `PYTHONPATH` unset:

```bash
cd /tmp && env -u PYTHONPATH python -c "import orbit_ssm; print(orbit_ssm.__file__)"
```

The path must point into the environment's `site-packages`. Pytest from the
source root uses the source tree instead (the `pythonpath` setting in
`pyproject.toml`); to test the installed copy, run pytest on a copy of `tests`
outside the checkout.
