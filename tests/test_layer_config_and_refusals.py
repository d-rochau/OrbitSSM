"""Pins: `layer.config` (the resolved family config, read-only) and CUDA refusals decided before any kernel work.

CPU only.  The CUDA entry point is replaced by a counter, and a tensor subclass reports ``is_cuda`` so the
``backend="auto"`` branch (warn once, run the torch route) is exercised without a device.

What the cells pin:
  contract  `layer.config == Config.from_dict(layer.to_recipe()["model"])`; every access is a fresh copy
            (mutating it changes nothing in the layer); it adds no state_dict key and survives deepcopy,
            torch.save / torch.load and `from_recipe`; `CausalLM.layer_configs` lists the layers' configs
  contract  a call the kernels refuse is named from its static properties BEFORE any kernel work: backend="cuda"
            raises, backend="auto" warns once and records `last_route()`, and the refused signature never
            enters the CUDA function again (a call counter)
  envelope  a refusal that depends on the configuration alone raises at construction and in `set_execution`
            under backend="cuda"
"""
import copy
import io
import warnings

import pytest
import torch

from orbit_ssm import CausalLM, HybridMambaConfig, SSDMimoConfig
from orbit_ssm.layers.routes import NonKernelRouteWarning
from orbit_ssm.ssm import CONFIGS

pytestmark = pytest.mark.ssm

_CFGS = {
    "ssd_mimo": SSDMimoConfig(d_model=32, d_state=16, d_inner=64, n_heads=2, n_groups=32),
    "hybrid_mamba": HybridMambaConfig(d_model=32, d_state=8, n_heads=4),
}


@pytest.fixture(autouse=True)
def _fresh_warned(monkeypatch):
    from orbit_ssm.layers import routes
    monkeypatch.setattr(routes, "_WARNED", set())


@pytest.fixture(params=sorted(_CFGS))
def layer(request):
    torch.manual_seed(0)
    return _CFGS[request.param].build()


def test_config_is_the_resolved_family_config(layer):
    """contract: layer.config is the family Config built from the recorded recipe (defaults filled)."""
    family = layer.config_family
    cfg = layer.config
    assert type(cfg) is CONFIGS[family]
    assert cfg == CONFIGS[family].from_dict(layer.to_recipe()["model"])
    assert cfg.d_model == 32


def test_config_is_a_fresh_copy_per_access(layer):
    """contract: mutating the returned config cannot be mistaken for changing the layer."""
    cfg = layer.config
    assert cfg is not layer.config and cfg == layer.config
    cfg.d_state = 4096
    assert layer.config.d_state != 4096 and layer.to_recipe()["model"]["d_state"] != 4096
    with pytest.raises(AttributeError):
        layer.config = cfg


def test_config_adds_no_state_and_survives_copies(layer):
    """contract: no new state_dict key; deepcopy, torch.save/load and from_recipe keep the config."""
    assert not [k for k in layer.state_dict() if "config" in k]
    assert copy.deepcopy(layer).config == layer.config
    buf = io.BytesIO()
    torch.save(layer, buf)
    buf.seek(0)
    assert torch.load(buf, weights_only=False).config == layer.config
    rebuilt = type(layer).from_recipe(layer.to_recipe())
    assert rebuilt.config == layer.config and rebuilt.to_recipe() == layer.to_recipe()


def test_causal_lm_layer_configs():
    """contract: lm.layer_configs is the tuple of the layers' configs, in order."""
    lm = CausalLM(vocab_size=64, layers=[_CFGS["ssd_mimo"], _CFGS["hybrid_mamba"]])
    got = lm.layer_configs
    assert isinstance(got, tuple) and [type(c) for c in got] == [SSDMimoConfig, HybridMambaConfig]
    assert got == tuple(m.config for m in lm.ssm_layers)


# ── refusals decided before any kernel work ──────────────────────────────────────────────────────────────

class _FakeCuda(torch.Tensor):
    """A CPU tensor that claims to live on a device, so ``backend="auto"`` resolves to the kernel route."""
    is_cuda = property(lambda self: True)


def _ssd(backend):
    torch.manual_seed(0)
    cfg = SSDMimoConfig(d_model=16, d_state=16, d_inner=32, n_heads=2, n_groups=16, bc_conv=False,
                        proj_source="main_branch",
                        row_blocks=[{"frac": 1.0, "decay": "head_dynamic", "complex": True}])
    return cfg.build(execution={"backend": backend})


def _counting_cuda(monkeypatch, m):
    calls = []

    def cuda(*a, **k):
        calls.append(1)
        raise NotImplementedError("kernel entered")
    monkeypatch.setattr(m, "_forward_cuda", cuda)
    return calls


def _refuse_by_environment(monkeypatch):
    """A call-level refusal: SSD_MIMO_ROT_WHERE=host leaves a per-head phase without a host form."""
    monkeypatch.setenv("SSD_MIMO_ROT_WHERE", "host")


def test_cuda_backend_raises_before_any_kernel_work(monkeypatch):
    """contract: backend='cuda' + a statically refused signature raises ValueError naming the reason; the CUDA
    function is never entered."""
    m = _ssd("cuda")
    assert m.rot_mode == "m3", "the fixture needs the in-kernel rotation law"
    calls = _counting_cuda(monkeypatch, m)
    _refuse_by_environment(monkeypatch)
    with pytest.raises(ValueError, match=r"SSD_MIMO_ROT_WHERE.*backend='torch'"):
        m.scan(torch.randn(1, 8, 32))
    assert calls == []


def test_auto_warns_once_and_never_retries_the_refused_signature(monkeypatch):
    """contract: backend='auto' on a CUDA input warns once, runs the torch route, records last_route(), and later
    calls with the same signature do not enter the CUDA function; another signature is decided on its own."""
    m = _ssd("auto")
    calls = _counting_cuda(monkeypatch, m)
    _refuse_by_environment(monkeypatch)
    x = torch.randn(2, 8, 32).as_subclass(_FakeCuda)
    with pytest.warns(NonKernelRouteWarning, match="SSD_MIMO_ROT_WHERE") as rec:
        y = m.scan(x)
    assert len(rec) == 1 and calls == [] and "SSD_MIMO_ROT_WHERE" in m.last_route()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(3):
            m.scan(x)
    assert calls == [] and torch.isfinite(y).all()
    # the decision is a property of the signature: with the lever gone the kernel is tried again
    monkeypatch.delenv("SSD_MIMO_ROT_WHERE")
    with pytest.warns(NonKernelRouteWarning, match="cuda route envelope"):
        m.scan(x)
    assert calls == [1]                  # the safety net caught the (test) refusal once ...
    m.scan(x)
    assert calls == [1]                  # ... and remembers the signature


def test_set_execution_change_reopens_the_decision(monkeypatch):
    """envelope: the cached decision belongs to the execution it was made under."""
    m = _ssd("auto")
    calls = _counting_cuda(monkeypatch, m)
    _refuse_by_environment(monkeypatch)
    x = torch.randn(1, 8, 32).as_subclass(_FakeCuda)
    with pytest.warns(NonKernelRouteWarning):
        m.scan(x)
    monkeypatch.delenv("SSD_MIMO_ROT_WHERE")
    m.set_execution({"precision": "ieee"})
    with pytest.warns(NonKernelRouteWarning):
        m.scan(x)
    assert calls == [1]


def test_config_only_refusal_raises_at_construction_and_set_execution():
    """envelope: d_state=48 has no in-kernel rotation (power of two <= 128): backend='cuda' refuses at
    construction and in set_execution; 'auto' / 'torch' accept it."""
    cfg = SSDMimoConfig(d_model=16, d_state=48, d_inner=32, n_heads=2, n_groups=16, bc_conv=False,
                        proj_source="main_branch",
                        row_blocks=[{"frac": 1.0, "decay": "head_dynamic", "complex": True}])
    with pytest.raises(ValueError, match=r"d_state=48.*backend='torch'"):
        cfg.build(execution={"backend": "cuda"})
    m = cfg.build(execution={"backend": "auto"})
    with pytest.raises(ValueError, match=r"d_state=48"):
        m.set_execution({"backend": "cuda"})
    assert m.backend == "auto"
    cfg.build(execution={"backend": "torch"})
