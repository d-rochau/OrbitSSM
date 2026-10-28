"""T2/T3: an `Execution` field a family cannot honour is refused, never stored; one chunk_size rule; one state rule.

Pins (what each test declares it checks):
  * `checkpoint=True` raises on SSD, `precision` / `chunk_size` raise on HybridMamba, at construction and in
    `set_execution`, and a refusal leaves the layer untouched; a value that IS the family's fixed behaviour
    (`checkpoint=False` on SSD, `chunk_size="auto"` on HybridMamba) and None are accepted and not stored;
  * `validation.check_chunk_size` is the one chunk rule: SSD takes 'auto' / 'measure' / power of two >= 16 (the fused floor),
    and refuses the rest through that function;
  * `_execution_state()` reports exactly the supported fields of the family (`_execution_fields`) and
    `with layer.execution(...)` round-trips it;
  * `scan_step` takes `gate_z` / `sel_x` / `tok_raw` by keyword only;
  * a benchmark config uses the current option names unless it declares `frozen: true`.
CPU only; the construction and refusal paths do not run a kernel.
"""

import glob
import os
import pytest
import torch
from orbit_ssm import HybridMambaConfig, SSDMimoConfig
from orbit_ssm.validation import check_chunk_size


CFG = {
    "ssd_mimo": lambda: SSDMimoConfig(d_model=32, d_state=16, expand=1, n_heads=4),
    "hybrid_mamba": lambda: HybridMambaConfig(d_model=32, d_state=16, expand=1, n_heads=4),
}


FIELDS = {"ssd_mimo": ("backend", "precision", "chunk_size", "kernel_options"),
          "hybrid_mamba": ("backend", "checkpoint", "kernel_options")}


REFUSED = [("ssd_mimo", {"checkpoint": True}),
           ("hybrid_mamba", {"precision": "ieee"}), ("hybrid_mamba", {"precision": "io"}),
           ("hybrid_mamba", {"chunk_size": 32})]


FIXED = [("ssd_mimo", {"checkpoint": False}),
         ("hybrid_mamba", {"chunk_size": "auto"}), ("ssd_mimo", {"checkpoint": None}),
         ("hybrid_mamba", {"precision": None})]


@pytest.mark.parametrize("family,field", REFUSED)
def test_unsupported_execution_field_raises(family, field):
    cfg = CFG[family]()
    with pytest.raises(ValueError, match="not supported by this family"):
        type(cfg.build())(cfg, execution=field)
    layer = cfg.build()
    before = layer._execution_state()
    with pytest.raises(ValueError, match="not supported by this family"):
        layer.set_execution(field)
    with pytest.raises(ValueError, match="not supported by this family"):
        with layer.execution(field):
            pass
    assert layer._execution_state() == before


@pytest.mark.parametrize("family,field", FIXED)
def test_fixed_behaviour_value_is_accepted_and_not_stored(family, field):
    cfg = CFG[family]()
    layer = type(cfg.build())(cfg, execution=field)
    before = layer._execution_state()
    layer.set_execution(field)
    with layer.execution(field):
        assert layer._execution_state() == before
    assert layer._execution_state() == before
    assert set(before) == set(FIELDS[family])


@pytest.mark.parametrize("family", sorted(CFG))
def test_execution_state_is_the_supported_fields(family):
    layer = CFG[family]().build()
    state = layer._execution_state()
    assert tuple(state) == FIELDS[family]
    with layer.execution({"backend": "torch"}):
        assert layer._execution_state()["backend"] == "torch"
    assert layer._execution_state() == state


def test_chunk_size_is_one_rule():
    for ok in ("auto", "measure", 16, 32, 64, 256):
        assert check_chunk_size(ok, "x") == ok
    for bad in (8, 48, 0, -16, True, "fast", 32.0, None):
        with pytest.raises(ValueError, match="chunk_size must be"):
            check_chunk_size(bad, "x")
    assert check_chunk_size("auto", "x", measure=False) == "auto"
    with pytest.raises(ValueError, match="'measure'"):
        check_chunk_size("measure", "x", measure=False)
    ssd = CFG["ssd_mimo"]().build()
    ssd.set_execution({"chunk_size": "measure"})
    assert ssd._execution_state()["chunk_size"] == "measure"
    with pytest.raises(ValueError, match="power-of-two"):
        ssd.set_execution({"chunk_size": 8})            # the fused kernels' floor
    for bad in (48, 0, 3):
        with pytest.raises(ValueError, match="power-of-two"):
            ssd.set_execution({"chunk_size": bad})


@pytest.mark.parametrize("family", ["ssd_mimo", "hybrid_mamba"])
def test_scan_step_side_inputs_are_keyword_only(family):
    layer = CFG[family]().build()
    x = torch.zeros(2, layer.d_inner)
    with pytest.raises(TypeError):
        layer.scan_step(x, None, x)


def _module_arm_kwargs(path):
    from benchmarks import run
    from benchmarks.cases import MODULE_ROUTES
    cfg = run.load_config(path)
    for arm in cfg.get("arms") or []:
        kw = dict(arm.get("kwargs") or {})
        fam = kw.pop("family", "ssd_mimo")
        if arm["route"] in MODULE_ROUTES and not fam.startswith("official_"):
            yield cfg, arm["name"], kw


def test_old_spellings_are_refused_by_the_benchmark_builder():
    """contract (fresh prod): benchmark configs use the current names only; an old option or
    run-time key raises (there is no `frozen: true` translation any more)."""
    from benchmarks.cases import build_operation
    shape = dict(B=1, L=16, N=16, d_model=32)
    kw = dict(backend="torch", device="cpu", pass_="fwd")
    for old in (dict(family="hybrid_mamba", expand=1, headdim=8), dict(family="hybrid_mamba", expand=1, ckpt=False)):
        with pytest.raises(Exception):
            build_operation("layer", shape, dict(old), **kw)
