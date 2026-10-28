# OrbitSSM

PyTorch implementations of **SSD-MIMO** and **HybridMamba** selective state space layers, with fused
Triton/Gluon GPU kernels, accompanying the paper *Diagonal Equivalence in State Space Models:
Characterization, Retrieval Rank and Selective Extensions*.

* **SSD-MIMO** is the Mamba-2/3 lineage (scalar decay per head, trapezoid or Euler write) with input rank `R`
  and output rank `O`; `R = O = 1` is the SISO layer.
* **HybridMamba** is the hybrid decay-law family: per-coordinate decay laws and optional rotation, chosen per
  block of state rows (`row_blocks`).
* Both are complete layers (input projection, causal convolution, coefficient maps, recurrence, gate, output
  projection) behind one interface; `ResidualBlock`, `BlockStack` and `CausalLM` compose them into models.
* Every layer has a PyTorch route that runs on CPU (it is also the fp64 oracle of the kernel tests) and CUDA
  kernels for GPUs.

The Python package is `orbit_ssm`; the pip distribution is `orbit-ssm`.

## Installation

**Supported range.** Linux, Python >= 3.12, PyTorch 2.14.x and Triton 3.8.x: the versions the code was tested
with, and the only ones `pyproject.toml` admits. GPU kernels need compute capability 8.0 or newer; they were
validated on sm_80 (A100), sm_90 (H100) and sm_120 (RTX PRO 5000 Blackwell). Other GPUs and shapes are
untested. Without a GPU every layer runs on its PyTorch route (`backend="torch"`), which is meant for tests
and small experiments, not for speed.

Install PyTorch and Triton for your CUDA driver first, then this package without letting pip replace them
(in a fresh Python 3.12 environment; `cu130` needs a CUDA 13 driver, pick the index matching yours otherwise):

```bash
git clone https://github.com/d-rochau/OrbitSSM.git && cd OrbitSSM
python -m pip install torch==2.14.* --index-url https://download.pytorch.org/whl/cu130
python -m pip install triton==3.8.*
python -m pip install --no-deps .        # with uv: uv pip install ... for each line
```

`--no-deps` keeps pip from installing or replacing torch and Triton: the dependency pins would otherwise pull
a default torch build that may not match your CUDA driver.

Optional packages are declared as extras in `pyproject.toml`; with `--no-deps` pip installs none of them, so
install the ones you need yourself:

| extra | packages | purpose |
|---|---|---|
| `test` | pytest, PyYAML, numpy, pytest-timeout | the test suite |
| `conv-package` | `causal-conv1d` | the optional compiled causal-convolution package (explicit opt-in, below; not needed for speed) |
| `baselines` | `mamba-ssm`, `causal-conv1d`, lightning, torchmetrics | the official Mamba implementations that the benchmarks and parity tests compare against; not used by our layers |

Both compiled packages build from source against your torch (C++20 compiler and a CUDA toolkit matching torch's
CUDA). `mamba-ssm` needs its CUDA build even for the Mamba-2/3 baselines, since `import mamba_ssm` loads Mamba-1's
compiled extension; see [docs/environment.md](docs/environment.md) for the three install levels.

**causal-conv1d is an optional extra, never used unless asked for.** The layers' causal convolution is our own
plain-Triton kernel (`kernel_options={"conv_kernel": "auto"}`, the default, same as `"triton"`; about 3 to 10x faster
than torch's convolution), whether or not the package is installed; SSD-MIMO's fused one-node route runs on it too.
`"causal_conv1d"` opts in to the compiled package (refused by name when it is not importable),
`"torch"` selects torch's convolution, and a layer's `conv_kernel_in_use` reports what runs. Nothing else outside PyTorch and Triton is required: the gated RMSNorm is implemented in this repository
(for HybridMamba `kernel_options={"norm_kernel": "triton"}`, the default, or `"torch"`).

Speed of our conv against the package, in a training step: one layer forward + backward with `conv_kernel="triton"` vs
`"causal_conv1d"`, same weights and input, everything else identical (time ratio ours / package, below 1 is faster;
20 steps back to back divided by 20, median of 5 interleaved rounds; `python -m benchmarks.bench_conv_step`; `benchmarks.bench_conv` times the kernels alone).

| layer | B x L | A100 bf16 | A100 fp32 | H100 bf16 | H100 fp32 |
|---|---|---|---|---|---|
| SSD-MIMO d_model 1024 | 8 x 2048 | 1.001 | 1.000 | 0.992 | 0.991 |
| SSD-MIMO d_model 640 | 6 x 3000 | 0.998 | 1.001 | 0.997 | 1.001 |
| SSD-MIMO d_model 2048 | 4 x 4096 | 1.002 | 1.001 | 1.004 | 1.002 |
| HybridMamba d_model 1024 | 8 x 2048 | 1.000 | 1.000 | 1.000 | 0.991 |
| HybridMamba d_model 640 | 6 x 3000 | 1.001 | 1.001 | 0.998 | 1.002 |

The conv kernels alone, on the layer's own call (channel-major [B, C, L], width 4, SiLU, bf16, back-to-back calls), in µs ours / package:

| GPU | B x C x L | forward | backward |
|---|---|---|---|
| A100 | 4 x 4096 x 4096 | 166 / 164 | 271 / 259 |
| A100 | 8 x 2048 x 2048 | 86 / 87 | 141 / 143 |
| H100 | 4 x 4096 x 4096 | 154 / 153 | 233 / 230 |
| H100 | 8 x 2048 x 2048 | 79 / 80 | 120 / 125 |

A single isolated call also pays Triton's Python launch cost (about 0.1 ms per call), so very small shapes called one
at a time favour the compiled package; in a training step that cost overlaps with the GPU work.

More on the environment and its limits: [docs/environment.md](docs/environment.md).

## Quickstart

A layer takes two arguments: a **config** (the architecture) and an **execution** (how it runs).

```python
import torch
from orbit_ssm import SSDMimoConfig, SSDMimoSSM

config = SSDMimoConfig(d_model=64, d_state=16, expand=1, d_conv=0, d_ssm_in=1, d_ssm_out=1, n_groups=16)
layer = SSDMimoSSM(config, execution={"backend": "torch"})   # the default "auto" uses CUDA for GPU inputs

x = torch.randn(2, 8, 64)                                      # [batch, length, d_model]
y = layer(x)                                                   # same shape
y.square().mean().backward()
```

`HybridMambaConfig` / `HybridMambaSSM` work the same way. HybridMamba has no `precision` or `chunk_size`
option; its token-by-token CPU recurrence is `backend="sequential"`:

```python
from orbit_ssm import HybridMambaConfig, HybridMambaSSM

hm_config = HybridMambaConfig(d_model=64, d_state=16, expand=1, n_heads=2)
hm = HybridMambaSSM(hm_config, execution={"backend": "sequential"})
y = hm(x)
```

### Config objects and Execution

* The **config** (`SSDMimoConfig`, `HybridMambaConfig`) is a keyword-only dataclass; `help(SSDMimoConfig)` lists
  every field. A layer and `CausalLM` also accept its strict dict (`SSDMimoConfig.from_dict`): unknown or
  wrongly typed keys raise. `config.build(execution=...)` returns the layer and `layer.to_recipe()` the
  recipe it was built from. [docs/CONFIG_GLOSSARY.md](docs/CONFIG_GLOSSARY.md) maps every config field to the
  symbol of the paper (`d_ssm_in` = R, `d_ssm_out` = O, `d_state` = N, `skip` = D, and the decay laws to the
  paper's names).
* **Execution** (`orbit_ssm.Execution`, or the equivalent dict) holds the choices that do not change what is
  computed: `backend` (`"auto"`, `"cuda"`, `"torch"`, `"sequential"`), `precision`, `chunk_size`, `checkpoint`
  and `kernel_options`. A field the family cannot honour raises instead of being ignored. Change it on a
  live model with `model.set_execution(...)` or `with model.execution(...):`.

### Recurrent state and decoding

`layer(x, state=..., return_state=True)` carries the recurrent `State` across calls, and `layer.step` is one
decoding step. Both agree with the full-sequence forward:

```python
y1, state = layer(x[:, :4], return_state=True)
y2, state = layer(x[:, 4:], state=state, return_state=True)   # continues where y1 stopped

state = layer.init_state(2)                                    # batch size 2
y_t, state = layer.step(x[:, 0], state)                        # one token
```

### Language model

```python
from orbit_ssm import CausalLM

lm = CausalLM(vocab_size=256, layers=config, n_layers=2, d_mlp=128, execution={"backend": "torch"})     # vocabulary of 256
ids = torch.randint(0, 256, (2, 16))
logits = lm(ids)                                               # [2, 16, 256]
out = lm.generate(ids[:, :4], max_new_tokens=5, temperature=0)  # greedy: prefill, then token-by-token steps

mixed = CausalLM(vocab_size=256, layers=[config, hm_config], d_mlp=128, execution={"backend": "torch"})  # one config per layer
```

`examples/lm_roundtrip.py` is a longer CPU example: a training step, streaming, step decoding, saving and
reloading the recipe and weights, and `generate`. `orbit_ssm.presets` provides `mamba1_config`,
`mamba2_config` and `mamba3_config`, the family configs that reproduce the official Mamba-1/2/3 layers.

## Tests

```bash
python -m pip install pytest PyYAML numpy pytest-timeout
python -m pytest -q tests                                 # the smoke profile (default): CPU tests in a few minutes,
                                                          # plus the GPU smoke when a CUDA device is present
python -m pytest -q -m "not gpu" tests                    # CPU only
python -m pytest -q --profile compact tests               # more parameter cells per test (the merge gate)
```

The CPU run covers the config and `Execution` API (strict inputs, recipes), the PyTorch routes against fp64
oracles (values and gradients), `step` against the full-sequence forward, recurrent-state continuation,
carry dtypes, bf16 decoding, the kernel-option switches and the refusal to fall back to a non-kernel route
silently. The `gpu`-marked tests compare the CUDA kernels of SSD-MIMO and HybridMamba with the same oracles
and skip without a CUDA device (a skipped test is not a validation pass). `--profile smoke|compact|full`
chooses how many parameter cells each test runs (`smoke`, the default: one per test function on the CPU and, on a
GPU, the fp64-oracle check of one representative configuration per kernel route and precision of SSD-MIMO and
HybridMamba; `compact`: a pair cover of the parameter cells; `full`: all). `--tuning calibrated|light|full` chooses
how kernel launches pick their config: by default each launch runs ONE config, the calibrated optimum of the GPU
when a calibration ships for it, else the fixed fallback (the test header says which); `light` and `full`
exercise the candidate searches. The first CUDA run compiles the kernels, which takes a while (about 10-15
minutes for the smoke on a cold cache).

## Kernel benchmarks

```bash
python -m benchmarks.bench_kernels --list-grids
python -m benchmarks.bench_ssms --list-grids
python -m benchmarks.bench_kernels --grid smoke --out smoke.jsonl      # needs a GPU
```

Compare like with like: same weights, inputs, precision and operation scope; keep compilation and autotuning
out of warmed timings; record the device, geometry, routes and calibration source. Peak allocation and
saved-for-backward storage are different metrics. Kernel configurations are calibrated per device, and a
missing entry makes the first call autotune. The training campaigns of the paper are not part of this
repository.

**Paper timings.** The timing and memory tables of the paper were measured on the code tagged
`paper1-submission`, the first commit of this repository (the paper's supplementary code, with references to
unreleased work removed; kernels and launch configurations unchanged). The library has changed since, for
example the causal convolution, which then ran the `causal-conv1d` package (installed for the paper's
measurements; torch's convolution without it) and now runs our own Triton kernel.
To reproduce the paper's numbers, check out that tag; timings of the current version differ and should be
re-measured on your system.

## Citation

```bibtex
@article{orbitssm,
  title = {Diagonal Equivalence in State Space Models: Characterization, Retrieval Rank and Selective Extensions},
  note  = {TODO: arXiv id}
}
```

## Code availability

This repository contains the model code, the CUDA/Triton/Gluon kernels, the tests and the kernel benchmarks
for the SSD-MIMO and HybridMamba results of the paper. The training pipelines, datasets, checkpoints and the
model-level experiment campaigns are not included. Measured speeds and memory depend on the GPU, driver and
library versions; re-measure them on your own system.

## License

Apache-2.0; see [LICENSE](LICENSE). Optional dependencies keep their own licenses.
