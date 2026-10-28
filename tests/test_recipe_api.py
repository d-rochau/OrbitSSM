"""Pins: the recipe API `to_recipe()` / `from_recipe(recipe, *, execution=None)` and `layer.get_execution()`.  CPU only.

  contract  `from_recipe(layer.to_recipe())` round-trips layers and `CausalLM`; the recipe format is unchanged
  contract  `execution=` REPLACES the recorded execution as a whole (never merged field by field): a layer recorded
            with non-default kernel_options loaded with `Execution(backend="torch")` has the default kernel_options
  contract  on a model recipe `execution=` is one Execution / dict for every layer or a list of one per layer
            (matching length); anything else raises
  contract  `layer.get_execution()` is the current Execution as an object, a fresh copy per access, read-only
"""
import pytest
import torch

from orbit_ssm import CausalLM, Execution, HybridMambaConfig, SSDMimoConfig

pytestmark = pytest.mark.ssm

_CFGS = {
    "ssd_mimo": SSDMimoConfig(d_model=32, d_state=16, d_inner=64, n_heads=2, n_groups=32),
    "hybrid_mamba": HybridMambaConfig(d_model=32, d_state=8, n_heads=4),
}


@pytest.fixture(params=sorted(_CFGS))
def cfg(request):
    return _CFGS[request.param]


def test_old_names_are_gone(cfg):
    """contract: no aliases for the former names."""
    layer = cfg.build()
    assert not hasattr(layer, "resolved_config") and not hasattr(type(layer), "from_config")
    assert not hasattr(CausalLM, "resolved_config") and not hasattr(CausalLM, "from_config")


def test_layer_recipe_round_trip_and_override_replaces_whole(cfg):
    """contract: round trip; the override is the whole execution (recorded kernel_options are not merged back)."""
    layer = cfg.build(execution={"backend": "sequential", "kernel_options": {"conv_kernel": "torch"}})
    recipe = layer.to_recipe()
    assert type(layer).from_recipe(recipe).to_recipe() == recipe
    assert recipe["execution"]["kernel_options"]["conv_kernel"] == "torch"
    moved = type(layer).from_recipe(recipe, execution=Execution(backend="torch"))
    assert moved.backend == "torch"
    assert moved.to_recipe()["model"] == recipe["model"]
    assert moved.to_recipe()["execution"]["kernel_options"]["conv_kernel"] == "auto"      # replaced, not merged
    assert type(layer).from_recipe(recipe, execution={"backend": "torch"}).to_recipe() == moved.to_recipe()
    with pytest.raises(TypeError):
        type(layer).from_recipe(recipe, execution="torch")


def test_get_execution_is_a_typed_fresh_copy(cfg):
    """contract: layer.get_execution() is the current Execution (set_execution's counterpart), a copy per call, and
    its repr shows every field."""
    layer = cfg.build(execution={"backend": "torch"})
    ex = layer.get_execution()
    assert isinstance(ex, Execution) and ex.backend == "torch"
    assert ex.fields() == {k: v for k, v in layer._execution_state().items() if v is not None}
    assert all(f"{name}=" in repr(ex) for name in ("backend", "precision", "chunk_size", "checkpoint", "kernel_options"))
    ex.backend = "sequential"
    assert layer.get_execution().backend == "torch"
    layer.set_execution(layer.get_execution())                      # symmetric: a no-op
    assert layer.get_execution() == layer.get_execution()
    assert not hasattr(layer, "execution_state")
    with layer.execution(Execution(backend="sequential")):          # the context manager is unchanged
        assert layer.get_execution().backend == "sequential"
    assert layer.get_execution().backend == "torch"


def _lm():
    return CausalLM(vocab_size=64, layers=[_CFGS["ssd_mimo"], _CFGS["hybrid_mamba"]],
                    execution={"backend": "sequential"})


def test_models_get_execution_when_layers_agree_and_refuse_when_not():
    """contract: a model's get_execution() is the common layer execution; layers that differ (another family, or
    another backend) raise, naming the per-layer call."""
    same = CausalLM(vocab_size=64, layers=[_CFGS["ssd_mimo"]] * 2, execution={"backend": "sequential"})
    assert same.get_execution().backend == "sequential"
    same.ssm_layers[0].set_execution({"backend": "torch"})
    with pytest.raises(ValueError, match="ssm_layers"):
        same.get_execution()
    assert same.ssm_layers[0].get_execution().backend == "torch"
    with pytest.raises(ValueError, match="ssm_layers"):
        _lm().get_execution()                       # two families do not share one Execution


def test_causal_lm_override_single_list_and_refusals():
    """contract: one Execution (or dict) applies to every layer; a per-layer list of matching length applies
    entry by entry; a wrong length or type raises; no override keeps the recorded executions."""
    lm = _lm()
    recipe = lm.to_recipe()
    assert CausalLM.from_recipe(recipe).to_recipe() == recipe
    one = CausalLM.from_recipe(recipe, execution=Execution(backend="torch"))
    assert [m.backend for m in one.ssm_layers] == ["torch", "torch"]
    assert CausalLM.from_recipe(recipe, execution={"backend": "torch"}).to_recipe() == one.to_recipe()
    two = CausalLM.from_recipe(recipe, execution=[Execution(backend="torch"), {"backend": "sequential"}])
    assert [m.backend for m in two.ssm_layers] == ["torch", "sequential"]
    assert one.to_recipe()["model"] == recipe["model"]
    for bad in ([Execution()], [Execution()] * 3, "torch", 3):
        with pytest.raises((TypeError, ValueError)):
            CausalLM.from_recipe(recipe, execution=bad)
    ids = torch.randint(0, 64, (1, 6))
    assert one(ids).shape == (1, 6, 64)
