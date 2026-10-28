"""Model assembly, public API, and end-to-end language-model tests."""

import argparse
import copy
import inspect
import json
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from orbit_ssm import ResidualBlock, CausalLM, BlockStack, SSDMimoSSM, HybridMambaSSM
from orbit_ssm.model import mark_residual_out, rescale_residual_out
from _tolerance import assert_close_terms
from test_state_lifecycle import argument_amplification, module_terms
from orbit_ssm import HybridMambaConfig, SSDMimoConfig


# Every family the LM registry serves.  `hybrid_mamba` is the mh_scan
# family — one of the two kernel families the repository names — and had
# no model-level coverage at all before the two-family consolidation.
_FAMILIES = ("ssd_mimo", "hybrid_mamba")


# Every family the registry can step (chunk S1: `HybridMambaSSM.step` is
# now implemented -- torch-native, both HM-law rows and const rows, in
# their NATIVE two-tap trapezoid form -- so it is no longer excluded; the
# config-space step-vs-forward parity cells live in
# the contract suite).
_STEP_FAMILIES = _FAMILIES


def _model_kwargs(ssm_type: str) -> dict:
    common = dict(
        vocab_size=41,
        d_model=16,
        n_layers=2,
        d_state=4,
        d_conv=0,
        ssm_type=ssm_type,
    )
    options = {
        "ssd_mimo": {"d_ssm_in": 2, "d_ssm_out": 2, "execution": {"backend": "cuda"}},
        # d_inner = 16 and n_heads = 2 gives the head width P = 8 the MH
        # CUDA backward needs (a power of two); "sequential" is the fp64
        # explicit recurrence, i.e. the family's CPU-capable route.
        # b_norm/c_norm pinned off: their default flipped to True, but this helper
        # feeds the state-dict manifest freeze below (among other tests) -- pin the earlier
        # value so the frozen shape stays put; a deliberate manifest
        # update is a separate, explicit decision.  The B/C bias is NOT
        # pinned: the spec default bc_bias="head"
        # is what the manifest freezes (beta_b/beta_c).
        "hybrid_mamba": {"n_heads": 2, "execution": {"backend": "sequential"},
                         "b_norm": False, "c_norm": False},
    }[ssm_type]
    common["expand"] = 1.0  # width expand * d_model = 16 = n_heads * d_head
    if ssm_type == "ssd_mimo":
        # The fused CUDA path uses tensor-core tiles: N and G are at least 16.
        common.update(d_model=32, d_state=16, expand=2.0)
        options["n_groups"] = 16
    common.update(options)
    top = {k: common.pop(k) for k in ("vocab_size", "d_model", "n_layers")}
    execution = common.pop("execution")
    common["type"] = common.pop("ssm_type")
    return dict(**top, layers=common, execution=execution)


@pytest.mark.parametrize(
    ("ssm_type", "mixer_cls"),
    [
        ("ssd_mimo", SSDMimoSSM),
        ("hybrid_mamba", HybridMambaSSM),
    ],
)
def test_all_model_families_have_one_shared_layer_structure(ssm_type, mixer_cls):
    model = CausalLM(**_model_kwargs(ssm_type))
    assert isinstance(model.backbone, BlockStack)
    assert len(model.backbone.layers) == 2
    assert all(isinstance(layer, ResidualBlock) for layer in model.backbone.layers)
    assert all(isinstance(layer.mixer, mixer_cls) for layer in model.backbone.layers)
    assert model.lm_head.weight is model.embedding.weight
    assert model.padded_vocab_size == 48
    assert model.embedding.num_embeddings == 48
    assert model.lm_head.out_features == 48
    assert all(
        getattr(layer.mixer.out_proj.weight, "_normalize_w_num_layers", False)
        for layer in model.backbone.layers
    )


@pytest.mark.parametrize("ssm_type", ["hybrid_mamba"])
@pytest.mark.parametrize("batch,length", [(1, 8), (3, 16)])
def test_cpu_models_forward_backward_across_shapes(ssm_type, batch, length):
    torch.manual_seed(batch * 100 + length)
    model = CausalLM(**_model_kwargs(ssm_type))
    ids = torch.randint(0, model.vocab_size, (batch, length))
    logits = model(ids)
    assert logits.shape == (batch, length, model.vocab_size)
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert model.embedding.weight.grad is not None
    assert torch.isfinite(model.embedding.weight.grad).all()


def test_heterogeneous_stack_and_embeddings_api():
    model = CausalLM(
        vocab_size=29,
        layers=[
            {
                "type": "hybrid_mamba", "d_state": 4, "d_conv": 0, "n_heads": 2,
            },
        ],
        d_model=16,
    )
    # one layer at a time: the layers address their own execution
    model.ssm_layers[0].set_execution({"backend": "sequential"})
    ids = torch.randint(0, 29, (2, 8))
    embeddings = model(ids, return_embeddings=True)
    assert embeddings.shape == (2, 8, 16)
    assert model(ids).shape == (2, 8, 29)


# Every step-capable family whose decode route runs on CPU.  `ssd_mimo`
# is step-capable too but needs the fused CUDA kernel, so it is covered
# by the kernel-backed step == forward check of the contract suite;
# `hybrid_mamba`'s "sequential" backend is CPU-capable (step() is
# torch-native regardless of `backend`, chunk S1) so it belongs here now.
@pytest.mark.parametrize(
    "ssm_type", [f for f in _STEP_FAMILIES if f != "ssd_mimo"])
def test_generation_uses_the_shared_stack_cache_contract(ssm_type):
    model = CausalLM(**_model_kwargs(ssm_type)).eval()
    prompt = torch.randint(0, model.vocab_size, (2, 4))
    generated = model.generate(prompt, max_new_tokens=3, temperature=0)
    assert generated.shape == (2, 7)
    torch.testing.assert_close(generated[:, :4], prompt)


_WRAP_D_MODEL = 16


def _as_state_list(module, state, wrap):
    """Normalise a block's single State and a stack's per-layer list to
    the same shape for this test's shared assertions."""
    return state if wrap == "stack" else [state]


@pytest.mark.gpu
@pytest.mark.parametrize("wrap", ["block", "stack"])
def test_block_and_stack_step_matches_forward_on_the_cuda_route(
        require_cuda, wrap):
    """The same wrapper contract where the mixer's scan is the fused
    SSD-MIMO CUDA kernel: the wrappers must hand the kernel's carry back
    unchanged, at fp32, which the fp64 CPU cases above cannot reach."""
    from _oracles import relative
    from _oracles import assert_fp64_routes_close, decode_by_stepping

    torch.manual_seed(3)
    # H*G*R == d_inner is the SSD geometry contract; G=16 lands on the
    # fused tile rather than the host-side padding route.
    d_model = 64

    def mixer():
        return SSDMimoSSM(SSDMimoConfig(d_model=d_model, d_state=16, expand=1, d_conv=4,
                          d_ssm_in=2, d_ssm_out=2, n_groups=16, n_heads=2,
                          bc_conv=True), execution={"backend": "cuda"})

    if wrap == "block":
        module = ResidualBlock(mixer(), d_model)
    else:
        module = BlockStack.from_mixers([mixer(), mixer()], d_model)
    module = module.cuda().eval()
    torch.manual_seed(5)
    x = torch.randn(2, 64, d_model, device="cuda")
    full, stepped = decode_by_stepping(
        module, x, device=torch.device("cuda"), dtype=torch.float32)
    assert relative(stepped, full) < 3e-3


def test_model_configuration_validation():
    with pytest.raises(ValueError, match="unknown type"):
        CausalLM(vocab_size=8, layers=dict(type="unknown"), n_layers=1, d_model=8)
    # `gate=` was removed (API unification): an unknown name is
    # refused by the family config's strict dict form, listing the valid names.
    with pytest.raises(TypeError, match="unknown key"):
        CausalLM(vocab_size=8, layers=dict(type="ssd_mimo", gate="glu"), n_layers=1, d_model=8)
    with pytest.raises(ValueError, match="pad_vocab_size_multiple"):
        CausalLM(vocab_size=8, layers=dict(type="ssd_mimo"), n_layers=1, d_model=8, pad_vocab_size_multiple=0)


def test_ssd_mimo_named_mixer_modes_share_parameter_structure():
    default = SSDMimoSSM(SSDMimoConfig(
        d_model=8, d_state=4, expand=1, d_conv=0,
        d_ssm_in=1, d_ssm_out=1, n_groups=4, n_heads=2), execution={"backend": "torch"})
    fused = SSDMimoSSM(SSDMimoConfig(
        d_model=8, d_state=4, expand=1, d_conv=0,
        d_ssm_in=1, d_ssm_out=1, n_groups=4, n_heads=2), execution={"backend": "torch", "kernel_options": {"mode": "fused"}})
    assert default.mode == "modular" and not default.combined
    assert fused.mode == "fused" and fused.combined
    assert default.state_dict().keys() == fused.state_dict().keys()
    with pytest.raises(ValueError, match="does not support value 'other'"):
        SSDMimoSSM(SSDMimoConfig(
            d_model=8, expand=1, 
            d_ssm_in=1, d_ssm_out=1, n_groups=4, n_heads=2), execution={"kernel_options": {"mode": "other"}})
    with pytest.raises(ValueError, match="only meaningful"):
        SSDMimoSSM(SSDMimoConfig(
            d_model=8, expand=1,
            
            d_ssm_in=1, d_ssm_out=1, n_groups=4, n_heads=2), execution={"kernel_options": {"mode": "fused", "modular_checkpoint": True}})


def test_ssd_mimo_modular_checkpoint_preserves_gradients():
    """Checkpointing changes saved activations, not the modular mixer math."""
    kwargs = dict(
        d_model=8, d_state=4, expand=1, d_conv=3,
        d_ssm_in=1, d_ssm_out=1, n_groups=4, n_heads=2)
    torch.manual_seed(3)
    plain = SSDMimoSSM(SSDMimoConfig(**kwargs), execution={"backend": "torch", "kernel_options": {"mode": "modular"}}).double()
    checked = copy.deepcopy(plain)
    checked.set_execution(dict(kernel_options={"modular_checkpoint": True}))
    torch.manual_seed(5)
    x = torch.randn(2, 7, 8, dtype=torch.float64)
    dy = torch.randn_like(x)

    results = []
    for mixer in (plain, checked):
        xi = x.clone().requires_grad_(True)
        y = mixer(xi)
        (y * dy).sum().backward()
        grads = {name: p.grad.detach().clone()
                 for name, p in mixer.named_parameters()}
        results.append((y.detach(), xi.grad.detach(), grads))

    torch.testing.assert_close(results[1][0], results[0][0], rtol=0, atol=0)
    torch.testing.assert_close(results[1][1], results[0][1], rtol=0, atol=0)
    assert results[1][2].keys() == results[0][2].keys()
    for name in results[0][2]:
        torch.testing.assert_close(
            results[1][2][name], results[0][2][name], rtol=0, atol=0,
            msg=lambda msg, name=name: f"{name}: {msg}")

    # Stateful calls stay on the ordinary prologue because the detached
    # convolution tail is part of the public carry.  Enabling the memory
    # option must therefore remain transparent to chunk continuation.
    continued = []
    for mixer in (plain, checked):
        state = mixer.init_state(2, dtype=torch.float64)
        y0, state = mixer(x[:, :3], state=state, return_state=True)
        y1, state = mixer(x[:, 3:], state=state, return_state=True)
        continued.append((torch.cat([y0, y1], dim=1), state))
    torch.testing.assert_close(
        continued[1][0], continued[0][0], rtol=0, atol=0)
    assert continued[1][1].keys() == continued[0][1].keys()
    for key in continued[0][1]:
        torch.testing.assert_close(
            continued[1][1][key], continued[0][1][key], rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("checkpoint", [False, True], ids=["saved", "recompute"])
def test_ssd_mimo_modular_mode_captures_with_all_gradients(
        require_cuda, checkpoint):
    """PC.3's two modular memory choices both support captured training."""
    from orbit_ssm import graphs as G

    torch.manual_seed(3)
    mixer = SSDMimoSSM(SSDMimoConfig(
        d_model=64, d_state=16, expand=1, d_conv=4,
        d_ssm_in=1, d_ssm_out=1, n_groups=16, n_heads=4), execution={"backend": "cuda", "kernel_options": {
            "mode": "modular", "modular_checkpoint": checkpoint}}).cuda()
    eager = copy.deepcopy(mixer)
    x = torch.randn(2, 64, 64, device="cuda", requires_grad=True)
    graph = G.graphed_step(
        mixer, (x.detach().clone().requires_grad_(True),), fwd_bwd=True,
        amp=lambda: G.autocast("bf16"), warmup=4)

    def eager_step(xi, dyi):
        eager.zero_grad(set_to_none=True)
        xi = xi.detach().clone().requires_grad_(True)
        with G.autocast("bf16"):
            yi = eager(xi)
        yi.backward(dyi)
        grads = {name: None if p.grad is None else p.grad.detach().clone()
                 for name, p in eager.named_parameters()}
        grads["input[0]"] = xi.grad.detach().clone()
        return yi.detach(), grads

    def worst_relative(left, right):
        worst, where = 0.0, ""
        for name, expected in left.items():
            actual = right[name]
            assert (actual is None) == (expected is None), name
            if actual is None:
                continue
            scale = max(float(expected.abs().max()), 1e-8)
            rel = float((actual - expected).abs().max()) / scale
            if rel > worst:
                worst, where = rel, name
        return worst, where

    worst_graph = worst_floor = worst_y = 0.0
    graph_where = floor_where = ""
    for iteration in range(2):
        torch.manual_seed(100 + iteration)
        xi = torch.randn_like(x)
        dyi = torch.randn_like(x)
        y_eager, eager_grads = eager_step(xi, dyi)
        _, eager_grads_2 = eager_step(xi, dyi)
        y_graph = graph.replay(xi, grad_outputs=dyi)
        torch.cuda.synchronize()
        graph_grads = {name: value.detach().clone()
                       for name, value in graph.grads.items()}
        worst_y = max(
            worst_y, float((y_graph.detach().float()
                            - y_eager.float()).abs().max()))
        rel, where = worst_relative(eager_grads, graph_grads)
        if rel > worst_graph:
            worst_graph, graph_where = rel, where
        rel, where = worst_relative(eager_grads, eager_grads_2)
        if rel > worst_floor:
            worst_floor, floor_where = rel, where

    assert worst_y == 0.0
    bar = max(5.0 * worst_floor, 5e-3)
    assert worst_graph <= bar, (
        f"captured gradient {graph_where} moved {worst_graph:.3g} > {bar:.3g}; "
        f"eager floor {worst_floor:.3g} on {floor_where}")


def test_residual_depth_scaling_policy():
    class ToyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(8, 16)
            self.out_proj = nn.Linear(16, 16, bias=False)
            mark_residual_out(self.out_proj.weight)

    torch.manual_seed(123)
    shallow = ToyLM()
    assert rescale_residual_out(shallow, n_residuals=1) == 1
    torch.manual_seed(123)
    deep = ToyLM()
    assert rescale_residual_out(deep, n_residuals=4) == 1
    torch.testing.assert_close(deep.out_proj.weight, shallow.out_proj.weight / 2)
    # the untagged embedding is not the stack rescale's business
    torch.testing.assert_close(deep.embedding.weight, shallow.embedding.weight,
                               rtol=0, atol=0)


def test_d_mlp_zero_is_the_historical_block():
    """The default must add NOTHING: no module, no key, no init change."""
    model = CausalLM(**_model_kwargs("ssd_mimo"))
    assert model.d_mlp == 0
    assert model.backbone.n_residuals == model.n_layers
    for block in model.backbone.layers:
        assert block.mlp is None and block.norm_mlp is None
    assert not [k for k in model.state_dict() if ".mlp." in k or "norm_mlp" in k]


def test_swiglu_sublayer_parameter_count_matches_mamba_ssm_gated_mlp():
    """One block's MLP is exactly Mamba-2/3's interleaved gated MLP.

    ``3 * d_mlp * d_model`` is the arithmetic; ``GatedMLP`` is the
    reference implementation the papers (and mamba_ssm) use. The two split the
    same matrices differently (fused ``fc1: 2*hidden x d`` vs separate
    ``gate``/``up``) and must carry the same number of parameters, or the arms
    are not on the same backbone.
    """
    d_model, d_mlp = 1024, 2048
    block = ResidualBlock(mixer=nn.Identity(), d_model=d_model,
                           d_mlp=d_mlp)
    n_mlp = sum(p.numel() for p in block.mlp.parameters())
    assert n_mlp == 3 * d_mlp * d_model

    mlp_mod = pytest.importorskip("mamba_ssm.modules.mlp")
    reference = mlp_mod.GatedMLP(d_model, hidden_features=d_mlp,
                                 bias=False)
    assert n_mlp == sum(p.numel() for p in reference.parameters())


def test_d_mlp_adds_one_mlp_per_block_and_doubles_the_init_depth():
    kwargs = _model_kwargs("ssd_mimo")
    bare = CausalLM(**kwargs)
    with_mlp = CausalLM(**kwargs, d_mlp=32)
    d_model, n_layers = kwargs["d_model"], kwargs["n_layers"]

    extra = sum(p.numel() for p in with_mlp.parameters()) \
        - sum(p.numel() for p in bare.parameters())
    # per block: 3 * d_mlp * d_model (SwiGLU) + d_model (its pre-norm)
    assert extra == n_layers * (3 * 32 * d_model + d_model)

    assert with_mlp.backbone.n_residuals == 2 * n_layers
    # Two residual streams per block => the 1/sqrt(depth) rescale uses 2*n_layers.
    assert with_mlp.backbone.layers[0].mlp.down.weight._normalize_w_num_layers


def test_mixer_registry_takes_precedence_and_reserves_the_builtin_names():
    class _Toy(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            mark_residual_out(self.out_proj.weight)

        def forward(self, x, state=None, return_state=False):
            return self.out_proj(x)

    seen = {}

    def factory(d_model, d_state=64, expand=2.0, d_conv=4, **kw):
        seen.update(d_model=d_model, d_state=d_state, expand=expand,
                    d_conv=d_conv, **kw)
        return _Toy(d_model)

    name = "_toy_registry_mixer"
    CausalLM.register_mixer(name, factory)
    try:
        assert name in CausalLM.registered_mixers()
        model = CausalLM(vocab_size=41, layers=dict(type=name, d_state=4, d_conv=0, flavour="x"), n_layers=2, d_model=16)
        assert seen == dict(d_model=16, d_state=4, expand=2.0, d_conv=0,
                            flavour="x")
        ids = torch.randint(0, model.vocab_size, (2, 5))
        assert model(ids).shape == (2, 5, model.vocab_size)

        with pytest.raises(ValueError, match="built-in SSM family"):
            CausalLM.register_mixer("ssd_mimo", factory)
    finally:
        from orbit_ssm import model as _model_mod
        _model_mod._MIXER_REGISTRY.pop(name, None)


@pytest.mark.parametrize("d_model", [256, 1024])
def test_llama3_attention_mixer_costs_exactly_four_d_model_squared(d_model):
    from orbit_ssm.layers.attention import Llama3Attention

    attn = Llama3Attention(d_model)
    assert attn.n_heads == d_model // 64
    assert sum(p.numel() for p in attn.parameters()) == 4 * d_model ** 2
    x = torch.randn(2, 6, d_model)
    assert attn(x).shape == x.shape


def test_llama3_attention_is_causal():
    from orbit_ssm.layers.attention import Llama3Attention

    torch.manual_seed(0)
    attn = Llama3Attention(64).eval()
    x = torch.randn(1, 7, 64)
    y = attn(x)
    x2 = x.clone()
    x2[:, 4:] = torch.randn(1, 3, 64)      # perturb the FUTURE only
    torch.testing.assert_close(attn(x2)[:, :4], y[:, :4])


@pytest.mark.gpu
def test_ssd_mimo_language_model_cuda_forward_backward(require_cuda):
    model = CausalLM(**_model_kwargs("ssd_mimo")).cuda()
    ids = torch.randint(0, model.vocab_size, (2, 64), device="cuda")
    logits = model(ids)
    assert logits.shape == (2, 64, model.vocab_size)
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert torch.isfinite(model.embedding.weight.grad).all()


def test_hybrid_mamba_step_defaults_to_fresh_state():
    layer = HybridMambaSSM(HybridMambaConfig(d_model=16, d_state=4, d_inner=16, d_conv=0, n_heads=2,
                           proj_source="main_branch",
                           row_blocks=[{"frac": 1.0, "decay": "coord_static"}],
                           bc_bias="none"), execution={"backend": "sequential"})
    model = CausalLM(**_model_kwargs("hybrid_mamba")).eval()
    # the bare recurrence (scan_step) defaults to a fresh State too
    x = torch.randn(2, 16)
    actual, state = layer.scan_step(x)
    expected, explicit = layer.scan_step(x, layer.init_state(2))
    torch.testing.assert_close(actual, expected)
    for key in state:
        torch.testing.assert_close(state[key], explicit[key])
    for module in (layer, model.backbone.layers[0].mixer):
        x = torch.randn(2, 16)
        actual, state = module.step(x)
        expected, explicit = module.step(x, module.init_state(2))
        torch.testing.assert_close(actual, expected)
        for key in state:
            torch.testing.assert_close(state[key], explicit[key])


@pytest.mark.gpu
def test_hybrid_mamba_language_model_cuda_forward_backward(require_cuda):
    """The mh_scan family end to end through the LM on its production
    route: in_proj -> conv+SiLU -> `mh_scan` -> SwiGLU -> out_proj, at
    fp32 and under bf16 autocast.

    `hybrid_mamba` is one of the two kernel families the repository
    names and had NO model-level coverage before this; `_model_kwargs`
    builds it on the CPU-capable "sequential" route, so the CUDA kernel
    itself is only reached here.
    """
    kwargs = dict(vocab_size=41, d_model=64, n_layers=2,
                  layers=dict(type="hybrid_mamba", d_state=32,
                                     d_conv=4, expand=2.0, n_heads=2))
    model = CausalLM(**kwargs).cuda()
    ids = torch.randint(0, model.vocab_size, (2, 64), device="cuda")

    logits = model(ids)
    assert logits.shape == (2, 64, model.vocab_size)
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert torch.isfinite(model.embedding.weight.grad).all()

    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(ids)
    assert torch.isfinite(out.float()).all()
    out.float().square().mean().backward()
    assert all(torch.isfinite(p.grad.float()).all()
               for p in model.parameters() if p.grad is not None)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16],
                         ids=["fp32", "bf16"])
@pytest.mark.parametrize("rot", [None, "m3"])
def test_ssd_mimo_mixer_dtypes_and_rot_modes(require_cuda, dtype, rot):
    """SSDMimoSSM end-to-end (in_proj+conv+fused gate+SSM) across the
    production dtypes and rotation modes."""
    torch.manual_seed(3)
    mixer = SSDMimoSSM(SSDMimoConfig(
        d_model=64, d_state=32, expand=2.0,
        d_ssm_in=1, n_groups=16, n_heads=8, row_blocks=[
            {"frac": 1.0, "decay": "head_dynamic", "complex": rot == "m3"}],
    )).cuda().to(dtype)
    x = torch.randn(2, 128, 64, device="cuda", dtype=dtype,
                    requires_grad=True)
    y = mixer(x)
    assert y.shape == x.shape and torch.isfinite(y.float()).all()
    y.float().square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad.float()).all()


class TestSSMConfigs:
    """The shared config base + family configs: build, refuse, and LM wiring."""

    def test_core_validation(self):
        from orbit_ssm.config import _CoreConfigBase
        with pytest.raises(ValueError):
            _CoreConfigBase(d_inner=0)
        cfg = _CoreConfigBase(d_inner=64)
        assert cfg.replace(d_ssm_in=2).d_ssm_in == 2

    def test_family_builds(self):
        from orbit_ssm import HybridMambaConfig, SSDMimoConfig
        for C, kw in [
            (SSDMimoConfig, dict(d_inner=128, d_state=32, d_ssm_in=2, n_groups=8,
                                 proj_source="main_branch")),
            # mh_scan: n_heads must leave a power-of-two head width P
            # (128 / 4 = 32); "sequential" is its CPU-capable route.
            (HybridMambaConfig, dict(d_inner=128, d_state=32, n_heads=4,
                                     proj_source="main_branch")),
        ]:
            m = C(**kw).build(execution={"backend": "torch"})
            assert sum(p.numel() for p in m.parameters()) > 0

    def test_loud_refusal(self):
        """A family must refuse unsupported core fields, never drop them."""
        from orbit_ssm.config import HybridMambaConfig
        # mh_scan is SISO by construction (its rank axis IS the head
        # width P), and its rotation is a per-ROW property of `rows`
        # (complex=True).
        with pytest.raises(NotImplementedError):
            HybridMambaConfig(d_inner=128, n_heads=4, d_ssm_in=4).build()

    def test_hybrid_mamba_requires_a_power_of_two_head_width(self):
        """The MH Gluon backward tiles P into a BlockedLayout whose
        threadsPerWarp must be a power of two.  The forward tolerates any
        divisor, so without the constructor check the failure would be a
        Triton CompilationError on the FIRST BACKWARD rather than at
        construction — refused loudly instead."""
        with pytest.raises(ValueError, match="power of two"):
            HybridMambaConfig(d_inner=96, d_state=32, n_heads=4).build()
        # ... and only for the routes that reach the kernel.
        assert HybridMambaConfig(d_inner=96, d_state=32, n_heads=4,
                                 proj_source="main_branch").build(
            execution={"backend": "sequential"}) is not None

    def test_lm_takes_config(self, require_cuda):
        import torch
        from orbit_ssm.model import CausalLM
        from orbit_ssm.config import SSDMimoConfig
        cfg = SSDMimoConfig(d_inner=128, d_state=32, d_ssm_in=2, n_groups=16)
        lm = CausalLM(vocab_size=128, layers=cfg, n_layers=2, d_model=64).cuda()
        x = torch.randint(0, 128, (2, 64), device="cuda")
        out = lm(x)
        y = out[0] if isinstance(out, tuple) else out
        assert y.shape == (2, 64, 128)
        y.float().sum().backward()

    def test_lm_takes_hybrid_mamba_config(self):
        """Typed core options reach the mixer and retain their declared width."""
        cfg = HybridMambaConfig(d_inner=64, d_state=32, n_heads=2,
                                proj_source="main_branch")
        lm = CausalLM(vocab_size=64, layers=cfg, n_layers=1, d_model=64, execution={"backend": "sequential"})
        assert isinstance(lm.backbone.layers[0].mixer, HybridMambaSSM)
        assert lm.backbone.layers[0].mixer.d_inner == cfg.d_inner
        with pytest.raises(TypeError, match="expand"):    # the LM takes no layer options besides layers
            CausalLM(vocab_size=64, layers=cfg, n_layers=1, d_model=64, expand=2)

    def test_lm_config_type_mismatch(self):
        """A dict names its family by "type" and holds that family's fields only; another family's
        field is refused by name (the config object's own type is its family)."""
        from orbit_ssm.model import CausalLM
        with pytest.raises(TypeError, match="d_rotation_block"):
            CausalLM(vocab_size=128, layers=dict(type="ssd_mimo", d_inner=64, d_state=32, d_rotation_block=8), n_layers=1, d_model=64)

    def test_autocast_policy(self, require_cuda):
        """T5: under torch.autocast(bf16) every family must run fwd+bwd
        and produce finite outputs (modules run in param dtype; the scan
        cores receive autocast-cast activations)."""
        import torch
        from orbit_ssm.model import CausalLM
        from orbit_ssm.config import SSDMimoConfig
        x = torch.randint(0, 128, (2, 64), device="cuda")
        cases = [
            ("ssd_mimo", SSDMimoConfig(d_inner=128, d_state=32, d_ssm_in=2,
                                       n_groups=16), 32),
        ]
        for name, cfg, _ in cases:
            lm = CausalLM(vocab_size=128, layers=cfg, n_layers=1, d_model=64).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = lm(x)
            y = out[0] if isinstance(out, tuple) else out
            assert torch.isfinite(y.float()).all(), name
            y.float().sum().backward()
            grads_finite = all(
                torch.isfinite(p.grad.float()).all()
                for p in lm.parameters() if p.grad is not None)
            assert grads_finite, name


@pytest.mark.gpu
def test_hybrid_mamba_language_model_cuda_forward_backward_with_rotation(
        require_cuda):
    """Same statement as `test_hybrid_mamba_language_model_cuda_forward_
    backward`, with a rotating `rows` override (module-level rotation uses
    the `"complex"` key, not the kernel ABI's `"rot"` -- `compile_rows`
    refuses the old spelling by name) reaching `HybridMambaSSM` ->
    `mh_scan` through the full embedding/conv/gating/
    lm_head stack -- the default LM test builds the plain A_FIX law with
    no rotating row at all.  Scored against an fp64 copy of the SAME
    weights on the `backend="sequential"` route (not finiteness only),
    matching how `test_hybrid_mamba_row_mixture_matches_the_fp64_oracle`
    scores the bare core."""
    from orbit_ssm.kernels.hybrid_mamba_mh.reference import rel_err

    rows = [{"frac": 0.5, "decay": "coord_dynamic", "complex": True},
            {"frac": 0.25, "decay": "coord_potential"}, {"frac": 0.25, "decay": "coord_static"}]
    kwargs = dict(vocab_size=41, d_model=64, n_layers=1,
                  layers=dict(type="hybrid_mamba", d_state=32,
                                     d_conv=4, expand=2.0, n_heads=2,
                                     row_blocks=rows, bc_bias="none"))
    torch.manual_seed(9)
    model = CausalLM(**kwargs).cuda()
    mixer = model.backbone.layers[0].mixer
    assert isinstance(mixer, HybridMambaSSM)
    ids = torch.randint(0, model.vocab_size, (2, 64), device="cuda")

    logits = model(ids)
    assert logits.shape == (2, 64, model.vocab_size)
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert torch.isfinite(model.embedding.weight.grad).all()

    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(ids)
    assert torch.isfinite(out.float()).all()
    out.float().square().mean().backward()
    assert all(torch.isfinite(p.grad.float()).all()
              for p in model.parameters() if p.grad is not None)

    # fp64 oracle: the SAME weights, `backend="sequential"` (mh_scan's own
    # CPU-capable explicit recurrence).
    model.zero_grad(set_to_none=True)
    oracle = copy.deepcopy(model).double()
    oracle.set_execution(dict(backend="sequential"))
    logits_cuda = model(ids)
    logits_ref = oracle(ids)
    assert rel_err(logits_cuda, logits_ref) < 5e-3, "logits vs sequential fp64"

    gy = torch.randn_like(logits_cuda)
    logits_cuda.backward(gy)
    logits_ref.backward(gy.double())
    err = rel_err(model.embedding.weight.grad,
                  oracle.embedding.weight.grad)
    assert err < 3e-2, f"d(embedding.weight) vs sequential fp64: {err:.1e}"


@pytest.mark.gpu
@pytest.mark.parametrize("rot", [None, "m3"])
def test_ssd_mimo_mixer_rotation_matches_the_torch_backend(require_cuda, rot):
    """`test_ssd_mimo_mixer_dtypes_and_rot_modes` checks `isfinite` only --
    this is an oracle-scored statement at the mixer level,
    for every rotation mode the module supports: `backend="cuda"` (the
    fused kernel, through the mixer's own in_proj/conv/gate wiring) against
    `backend="torch"` (the independent quadratic-dual reference,
    `unified_ssd_quadratic`, run at fp64) on the same weights and input.
    """
    from _oracles import relative

    torch.manual_seed(3)
    mixer = SSDMimoSSM(SSDMimoConfig(
        d_model=64, d_state=32, expand=2.0,
        d_ssm_in=1, n_groups=16, n_heads=8, row_blocks=[
            {"frac": 1.0, "decay": "head_dynamic", "complex": rot == "m3"}],
        
    ), execution={"backend": "cuda"}).cuda()
    oracle = copy.deepcopy(mixer).double()
    oracle.set_execution(dict(backend="torch"))

    torch.manual_seed(5)
    x = torch.randn(2, 128, 64, device="cuda", requires_grad=True)
    weight = torch.randn(2, 128, 64, device="cuda")
    y = mixer(x)
    (y * weight).sum().backward()

    x64 = x.detach().double().clone().requires_grad_(True)
    y_ref = oracle(x64)
    (y_ref * weight.double()).sum().backward()

    assert relative(y, y_ref) < 3e-3, f"rot={rot}: y vs the torch backend"
    assert relative(x.grad, x64.grad) < 1e-2, (
        f"rot={rot}: dx vs the torch backend")


# --- the compact "paper" bench --------------------
# The compact bench narrows every non-smoke grid to a "paper set" -- ONLY the cells a paper
# table or sentence quotes -- via each bench module's own
# `PAPER_SETS: dict[str, Callable[[Cell], bool]]`.  These are CPU-only,
# grid-BUILD-time checks (`Cell.build` is a lazy closure none of them
# call): every non-smoke grid has a non-empty paper set, the arms/rows
# the paper tables actually cite survive the filter, and the 4-shard
# balance plan the shard planner computes is within the
# 25% budget.
_PAPER_GRID_MODULES = ("bench_kernels", "bench_ssms", "bench_models")


# Grids that legitimately have NO paper-set predicate, and why: `smoke`
# ("never quote a smoke number") -- nothing else.  `paper_c3_lscale`
# (Figure 4) USED TO be here too: `grid_paper_c3_lscale` hardcoded
# precision="ieee" for every cell (the figure predates a bf16 column),
# and ieee is dropped from the paper's quoted columns outright, so
# the grid had nothing left to quote under the compact protocol -- B6's
# recorded gap, now closed: the grid also builds a
# bf16 column at the same shapes/arms and carries a `PAPER_SETS` entry
# (`bench_kernels._paper_accept_all_no_ieee`) selecting it, the same
# predicate `mh_siso`/`paper_c3_nscale` use; `ieee` keeps running under
# `--cells full` unaffected.
_NO_PAPER_SET_GRIDS = frozenset(("smoke",))


@pytest.mark.parametrize("precision", ["io", "tf32", "ieee"])
def test_ssd_precision_spelling_crosses_execution_and_module(precision):
    from orbit_ssm import Execution
    from orbit_ssm.ssm.ssd_mimo import SSDMimoSSM
    kw = dict(d_model=64, d_inner=64, d_state=64, n_groups=16)
    module = SSDMimoSSM(SSDMimoConfig(**kw), execution=Execution(backend="torch", precision=precision))
    direct = SSDMimoSSM(SSDMimoConfig(**kw), execution={"backend": "torch", "precision": precision})
    assert module.precision == direct.precision == precision
    direct.set_execution(dict(precision="ieee"))
    assert direct.precision == "ieee"


def test_retired_bf16_precision_alias_is_refused_on_both_surfaces():
    """PINS THE REFUSAL.  "bf16" was a legacy alias for the "io" OPERAND
    mode; the v1 back-compat cut removed it.  It has to raise at construction
    AND on a live module (`set_execution`), or a stale recipe silently picks the env default."""
    from orbit_ssm.ssm.ssd_mimo import SSDMimoSSM
    kw = dict(d_model=64, d_inner=64, d_state=64, n_groups=16)
    with pytest.raises(ValueError, match="precision must be one of"):
        SSDMimoSSM(SSDMimoConfig(**kw), execution={"backend": "torch", "precision": "bf16"})
    module = SSDMimoSSM(SSDMimoConfig(**kw), execution={"backend": "torch"})
    with pytest.raises(ValueError, match="precision must be one of"):
        module.set_execution(dict(precision="bf16"))


# ---------------------------------------------------------------------------
# The wrappers' own decode contract: ResidualBlock / BlockStack
# ---------------------------------------------------------------------------
# ResidualBlock.step computes x_t + mixer.step(norm(x_t)) and BlockStack.step threads layer i's cache to layer i
# and applies the final norm; `forward` has no caches at all, so it is the oracle and a divergence between the
# two is a wrapper bug.  The mixers' own step-vs-forward cells are in the contract suite.

_WRAP_D_MODEL = 16
_WRAP_KINDS = ("ssd_mimo", "hybrid_mamba")


def _wrap_depth(length: int, n_layers: int = 2) -> int:
    """A7 rounding depth of the fp64 route bound for `n_layers` wrapped mixers over `length` tokens: per layer the
    block norm and in_proj sums (d_model each), the scan's length-L and state sums (L + 16) and the out_proj sum (16)."""
    return n_layers * (2 * _WRAP_D_MODEL + length + 16 + 16)


def _wrap_mixer(kind: str, d_model: int = _WRAP_D_MODEL):
    """One CPU-capable, step-capable mixer per family."""
    if kind == "ssd_mimo":
        return SSDMimoSSM(SSDMimoConfig(d_model=d_model, d_state=8, d_inner=16, d_head=8, d_conv=0, d_ssm_in=2,
                                        d_ssm_out=2, n_groups=4, bc_conv=False), execution={"backend": "torch"})
    if kind == "hybrid_mamba":
        return HybridMambaSSM(HybridMambaConfig(d_model=d_model, d_state=8, d_inner=16, n_heads=2, d_conv=0,
                                                row_blocks=[{"frac": 1.0, "decay": "coord_static"}], bc_bias="none"),
                              execution={"backend": "sequential"})
    raise AssertionError(kind)


def _wrapped(kind: str, wrap: str, d_model: int = _WRAP_D_MODEL):
    if wrap == "block":
        return ResidualBlock(_wrap_mixer(kind, d_model), d_model)
    # two layers: a stack that reused one cache for every layer, or threaded them out of order, fails
    return BlockStack.from_mixers([_wrap_mixer(kind, d_model), _wrap_mixer(kind, d_model)], d_model)


@pytest.mark.parametrize("wrap", ["block", "stack"])
@pytest.mark.parametrize("kind", list(_WRAP_KINDS))
def test_block_and_stack_step_matches_forward(wrap, kind):
    """The residual/norm wrappers decode what they forward (fp64, two independent formulations)."""
    from _oracles import assert_fp64_routes_close, decode_by_stepping

    torch.manual_seed(3)
    module = _wrapped(kind, wrap).double().eval()
    torch.manual_seed(5)
    x = torch.randn(2, 16, _WRAP_D_MODEL, dtype=torch.float64)
    full, stepped = decode_by_stepping(module, x, device=torch.device("cpu"), dtype=torch.float64)
    assert stepped.shape == full.shape == x.shape
    assert_fp64_routes_close(stepped, full, _wrap_depth(x.shape[1], 1 if wrap == "block" else 2), f"{wrap} of {kind}")


def test_heterogeneous_stack_decodes_what_it_forwards():
    """A stack whose layers are DIFFERENT families still decodes: each layer's cache is its own object."""
    from _oracles import assert_fp64_routes_close, decode_by_stepping

    torch.manual_seed(3)
    stack = BlockStack.from_mixers([_wrap_mixer("ssd_mimo"), _wrap_mixer("hybrid_mamba")],
                                   _WRAP_D_MODEL).double().eval()
    torch.manual_seed(5)
    x = torch.randn(2, 16, _WRAP_D_MODEL, dtype=torch.float64)
    full, stepped = decode_by_stepping(stack, x, device=torch.device("cpu"), dtype=torch.float64)
    assert_fp64_routes_close(stepped, full, _wrap_depth(x.shape[1]), "heterogeneous stack")


@pytest.mark.parametrize("kind", list(_WRAP_KINDS))
def test_block_delegates_the_mixer_contract_and_stack_exposes_it(kind):
    """`ResidualBlock` carries the state lifecycle by delegation to its mixer (it is a `StateLifecycle`, not a
    `BaseMixer`), and the stack exposes the same decode entry points without being one."""
    from orbit_ssm.layers.state import BaseMixer, StateLifecycle

    block = ResidualBlock(_wrap_mixer(kind), _WRAP_D_MODEL)
    assert isinstance(block, StateLifecycle) and not isinstance(block, BaseMixer)
    stack = BlockStack.from_mixers([_wrap_mixer(kind)], _WRAP_D_MODEL)
    assert not isinstance(stack, BaseMixer)
    for module in (block, stack):
        assert callable(module.step) and callable(module.init_state)
    assert len(stack.init_state(2)) == len(stack.layers) == 1      # one state entry per layer, in layer order


def _as_state_list(state, wrap):
    """A block's single State and a stack's per-layer list, as one list."""
    return state if wrap == "stack" else [state]


@pytest.mark.parametrize("wrap", ["block", "stack"])
@pytest.mark.parametrize("kind", list(_WRAP_KINDS))
def test_block_and_stack_reset_set_and_get_state(wrap, kind):
    """`reset_state` / `set_state` / `get_state` at the wrapper level (the block delegates to its one mixer, the
    stack per layer): the live carry round-trips, and a mask resets exactly the selected batch rows."""
    torch.manual_seed(6)
    module = _wrapped(kind, wrap)
    x = torch.randn(3, 4, _WRAP_D_MODEL)
    state = module.init_state(3)
    for t in range(4):
        _, state = module.step(x[:, t], state)
    live = _as_state_list(state, wrap)

    for g, s in zip(_as_state_list(module.get_state(), wrap), live):
        for k, v in s.items():
            if torch.is_tensor(v):
                assert torch.equal(g[k], v)

    fresh = module.init_state(3)
    module.set_state(fresh, state)
    for f, s in zip(_as_state_list(fresh, wrap), live):
        for k, v in s.items():
            if torch.is_tensor(v):
                assert torch.equal(f[k], v)

    before = [{k: v.clone() for k, v in s.items() if torch.is_tensor(v)} for s in live]
    mask = torch.tensor([False, True, False])
    module.reset_state(state, mask)
    for s, b, r in zip(live, before, _as_state_list(module.init_state(3), wrap)):
        for k, v in s.items():
            if not torch.is_tensor(v) or v.dim() == 0 or v.shape[0] != 3:
                continue
            torch.testing.assert_close(v[~mask], b[k][~mask])
            torch.testing.assert_close(v[mask], r[k][mask])
