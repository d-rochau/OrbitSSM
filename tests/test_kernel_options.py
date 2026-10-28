"""The `kernel_options` config surface.

Each test names what it pins on its docstring's first line.  Everything here is CPU-only except the two
`@pytest.mark.gpu` cells at the bottom, which are the end-to-end
versions of claims the CPU cells pin at the mechanism.
"""

import json
import math
import os
import pytest
import torch
from _oracles import ssd_diff_reference
from _tolerance import assert_close_terms, terms_abs
import orbit_ssm.kernel_options as ko
from orbit_ssm.config import _CoreConfigBase
from orbit_ssm.execution import Execution
from orbit_ssm.kernels.ssd_mimo.autograd import _plan_record
from orbit_ssm.kernels.ssd_mimo.fused_fwd_kernel import _env
from orbit_ssm.model import CausalLM
from orbit_ssm.ssm import HybridMambaSSM, SSDMimoConfig, SSDMimoSSM, HybridMambaConfig


pytestmark = pytest.mark.contract


# Every SSD lever this surface names, with a non-default value that is
# admissible everywhere -- used by the "changes no parameter" invariant.
NON_DEFAULT = {
    "kernel_target": "portable",
    "autotune": "off",
    "fwd_cfg": "off",
    "fwd_cfg_iters": 5,
    "fwd_warps": 4,
    "fwd_stages": 2,
    "fwd_maxnreg": 128,
    "backward_states": "io",
    "chunk_size_bwd": 32,
    "smem_strategy": "t_halve",
    "backward_states_fp32": True,
    "theta_bwd_fp32": False,
    "rot_split_dtype": "fp32",
    "rot_operand_dtype": "io",
    "gluon_kbc_dalpha": True,
    "siso_adj_raw_common": True,
    "dt_fp32": False,
}


HM_NON_DEFAULT = {
    "kernel_target": "portable",
    "autotune": "pinned",
    "fwd_config": "BLOCK_P=4,BLOCK_L=256",
    "bwd_config": "TP_P=2",
    "carry_fwd_config": "BLOCK_C=16,BLOCK_T=128,NUM_WARPS=4",
    "carry_bwd_config": "BLOCK_C=32,BLOCK_T=64,NUM_WARPS=4",
    "gate_bias_partial": False,
    "rotation": "in_scan",
    "ckpt_bf16": False,
}


@pytest.fixture
def clean_env(monkeypatch):
    """Clear ONLY the levers this surface names, so precedence is read
    against the family default.

    Scoped to those levers deliberately: `conftest.py` pins the suite's
    autotune spaces through other `SSD_MIMO_` names,
    and a blanket wipe would silently widen every search this file's GPU
    cells run (a test owns what it asserts
    about, and no more).
    """
    for opt in ko.schema("ssd_mimo").values():
        if opt.env:
            monkeypatch.delenv("SSD_MIMO_" + opt.env, raising=False)


def test_contract_the_surface_covers_every_lever_the_contract_names():
    """contract: per-family key table.

    The table is normative, so it is asserted rather than read: a key
    added to the code without a contract row (or removed from the code
    while the row stands) fails here.
    """
    # kernel levers + the layer-level execution switches (LAYER_KEYS)
    assert ko.SUPPORTED_KERNEL_OPTIONS["ssd_mimo"] == (
        frozenset(NON_DEFAULT) | ko.LAYER_KEYS["ssd_mimo"])
    assert ko.SUPPORTED_KERNEL_OPTIONS["hybrid_mamba"] == (
        frozenset(HM_NON_DEFAULT) | ko.LAYER_KEYS["hybrid_mamba"])


def test_contract_hm_options_route_to_the_mh_readers(monkeypatch):
    """contract: hybrid_mamba rows -- tier names and
    aliases as `ssd_mimo`'s, an explicit
    option beats the environment, and a malformed config pin is refused."""
    from orbit_ssm.kernels.hybrid_mamba_mh import autotune as at
    for k in ("HYBRID_MH_AUTOTUNE", "HYBRID_MH_FWD_CONFIG",
              "HYBRID_MH_BWD_CONFIG", "HYBRID_MH_BWD_GATE_BIAS_PARTIAL"):
        monkeypatch.delenv(k, raising=False)
    assert at.tier() == "light" and at.pinned_config("fwd") is None
    for raw, want in (("0", "pinned"), ("1", "light"), ("off", "pinned"),
                      ("auto", "light"), ("full", "full")):
        monkeypatch.setenv("HYBRID_MH_AUTOTUNE", raw)
        assert at.tier() == want
    # default 'auto' = the (relaxed) atomics at every P; the env name turns partials ON
    # and is reported by resolve() as the kernel reads it
    assert not at.gate_bias_partial(16) and not at.gate_bias_partial(1)
    assert ko.resolve("hybrid_mamba")["gate_bias_partial"] == "auto"
    monkeypatch.setenv("HYBRID_MH_BWD_GATE_BIAS_PARTIAL", "1")
    assert at.gate_bias_partial(16) and at.gate_bias_partial(1)
    assert ko.resolve("hybrid_mamba")["gate_bias_partial"] is True
    monkeypatch.setenv("HYBRID_MH_BWD_GATE_BIAS_PARTIAL", "0")
    assert not at.gate_bias_partial(16)
    # a JSON-object pin is the same pin as its KEY=int spelling
    monkeypatch.setenv("HYBRID_MH_FWD_CONFIG", '{"BLOCK_P": 4, "BLOCK_L": 256}')
    assert at.pinned_config("fwd") == dict(BLOCK_P=4, BLOCK_L=256)
    monkeypatch.delenv("HYBRID_MH_FWD_CONFIG")
    tok = ko.push("hybrid_mamba", {"autotune": "light",
                                   "fwd_config": "BLOCK_P=4,BLOCK_L=256,PF=1",
                                   "gate_bias_partial": False})
    try:
        assert at.tier() == "light"
        assert at.pinned_config("fwd") == dict(BLOCK_P=4, BLOCK_L=256, PF=1)
        assert not at.gate_bias_partial(16)
    finally:
        ko.pop(tok)
    with pytest.raises(ko.KernelOptionError):
        ko.validate("hybrid_mamba", {"bwd_config": "BLOCK_P=4"})  # forward-only key


def test_contract_every_supported_key_declares_what_would_retire_it():
    """contract: not a dumping ground.

    A key with nothing that could delete it is a permanent knob, which
    is exactly what this surface refuses to become.  The registration
    check is run here as well as at first use so the failure names the
    offending key rather than an unrelated config construction.
    """
    ko._check_schema()
    for family in ko.FAMILIES:
        for key, opt in ko.schema(family).items():
            assert opt.open_until, f"{family}.{key} declares no open_until"
            assert opt.doc, f"{family}.{key} declares no doc"


def test_envelope_no_option_key_can_become_a_second_parameter_channel():
    """envelope: routing only, structurally.

    A key that collided with a config field or an `Execution` field would
    give one setting two channels, and the loser would be silent.
    """
    fields = (set(_CoreConfigBase.__dataclass_fields__)
              | set(Execution.__dataclass_fields__))
    for family in ko.FAMILIES:
        assert not (set(ko.schema(family)) & fields - {"kernel_options"})


def _ssd_layer(kernel_options=None, **kw):
    ex = {"backend": "torch"}
    if kernel_options is not None:
        ex["kernel_options"] = kernel_options
    return SSDMimoSSM(SSDMimoConfig(d_model=32, d_state=16, d_inner=64, d_ssm_in=1, **kw), execution=ex)


def test_contract_valid_options_round_trip_through_the_recipe():
    """contract: it serialises.

    The recipe's `execution` section is the round-trip entry
    (`from_recipe` and the LM build path both rebuild a layer through it),
    so a run is only reproducible if `kernel_options` survives it -- as a
    plain, JSON-serialisable dict.  The architecture config holds none.
    """
    opts = {"fwd_cfg": "measure", "fwd_warps": 4, "chunk_size_bwd": 32}
    layer = _ssd_layer(opts)
    assert layer.kernel_options == opts
    assert type(layer.kernel_options) is dict

    doc = json.loads(json.dumps(layer.to_recipe()))
    from orbit_ssm import kernel_options as _ko
    assert _ko.split("ssd_mimo", doc["execution"]["kernel_options"])[0] == opts   # + the layer switches
    again = SSDMimoSSM.from_recipe(doc)
    assert again.kernel_options == opts
    assert again.to_recipe() == layer.to_recipe()
    # the default is the empty map, i.e. "no option set"
    assert _ssd_layer().kernel_options == {}
    # a typed config is architecture only: no kernel_options field
    with pytest.raises(TypeError):
        SSDMimoConfig(d_inner=64, d_state=16, kernel_options=opts)


def test_envelope_an_unknown_key_is_refused_by_name_with_the_supported_set():
    """envelope: never silently ignored.

    The message must name the offending key AND list what is supported,
    because the caller's next action is to pick a real key.
    """
    with pytest.raises(ko.KernelOptionError) as e:
        _ssd_layer({"fwd_warp": 4})                        # typo
    msg = str(e.value)
    assert "'fwd_warp'" in msg
    for key in NON_DEFAULT:
        assert key in msg
    assert isinstance(e.value, ValueError)


@pytest.mark.parametrize("key,bad", [
    ("fwd_cfg", "sometimes"),
    ("backward_states", "store"),
    ("smem_strategy", "n_halve"),
    ("fwd_warps", 3),               # not a power of two the kernel lowers
    ("fwd_cfg_iters", 0),
    ("chunk_size_bwd", 24),         # not a power of two
    ("fwd_stages", True),           # a bool is not a stage count
])
def test_envelope_an_unsupported_value_is_refused_by_name(key, bad):
    """envelope: the VALUE is refused too."""
    with pytest.raises(ko.KernelOptionError) as e:
        _ssd_layer({key: bad})
    assert repr(key) in str(e.value) and repr(bad) in str(e.value)


@pytest.mark.parametrize("make", [
    lambda ex: HybridMambaSSM(HybridMambaConfig(d_model=32, d_state=16, d_inner=64, n_heads=2), execution=ex),
], ids=["hybrid_mamba"])
def test_envelope_a_family_refuses_the_other_familys_keys(make):
    """envelope: a key aimed at the wrong family is refused.

    HM publishes its own levers; the point of
    the per-family schema is that an SSD key aimed at it must LOOK
    refused instead of looking accepted and doing nothing.
    """
    assert make({"backend": "torch"}).kernel_options == {}      # empty is fine
    with pytest.raises(ko.KernelOptionError) as e:
        make({"backend": "torch", "kernel_options": {"fwd_warps": 4}})
    assert "'fwd_warps'" in str(e.value)


def test_contract_an_explicit_option_beats_the_env_lever(clean_env, monkeypatch):
    """contract: config > env.

    Asserted at `_env`, the single choke point every SSD lever reads
    through, because that is where the precedence is implemented.
    """
    monkeypatch.setenv("SSD_MIMO_SALL", "io")
    monkeypatch.setenv("SSD_MIMO_FWD_WARPS", "8")
    token = ko.push("ssd_mimo", {"backward_states": "recompute",
                                 "fwd_warps": 4})
    try:
        assert _env("SALL", "recompute") == "recompute"
        assert _env("FWD_WARPS", "") == "4"
    finally:
        ko.pop(token)
    # ... and the env lever is untouched once the scope closes
    assert _env("SALL", "recompute") == "io"


def test_contract_an_explicit_none_pin_shadows_an_env_pin(clean_env, monkeypatch):
    """contract: `None` is a value, not a gap.

    `fwd_warps=None` means "let the dispatcher choose"; a config that
    says so must not inherit a shell's pin.
    """
    monkeypatch.setenv("SSD_MIMO_FWD_WARPS", "8")
    token = ko.push("ssd_mimo", {"fwd_warps": None})
    try:
        assert _env("FWD_WARPS", "") == ""
    finally:
        ko.pop(token)


def test_contract_the_env_lever_still_beats_the_family_default(clean_env,
                                                               monkeypatch):
    """contract: env keeps its place above default.

    The ~135 un-migrated levers have no other channel, so an env value
    with no competing option must still be honoured; only an EXPLICIT
    option outranks it.
    """
    monkeypatch.setenv("SSD_MIMO_SMEM_STRATEGY", "n_split")
    assert _env("SMEM_STRATEGY", "measure") == "n_split"
    token = ko.push("ssd_mimo", {"fwd_cfg": "off"})        # a different key
    try:
        assert _env("SMEM_STRATEGY", "measure") == "n_split"
    finally:
        ko.pop(token)
    assert ko.resolve("ssd_mimo")["smem_strategy"] == "n_split"


def test_contract_the_scope_is_thread_local_and_pops_clean(clean_env):
    """contract: one model cannot steer another.

    Two models in one process is the normal case (a heterogeneous
    stack), so a scope that leaked would silently reroute a neighbour.
    """
    assert ko.current("ssd_mimo") == {}
    outer = ko.push("ssd_mimo", {"fwd_cfg": "off"})
    inner = ko.push("ssd_mimo", {"fwd_cfg": "measure"})
    assert ko.current("ssd_mimo") == {"fwd_cfg": "measure"}
    ko.pop(inner)
    assert ko.current("ssd_mimo") == {"fwd_cfg": "off"}
    ko.pop(outer)
    assert ko.current("ssd_mimo") == {} and _env("FWD_CFG", "auto") == "auto"


def test_contract_the_resolved_options_appear_in_the_dispatch_record(
        clean_env, monkeypatch):
    """contract: the routing is readable after.

    Timing cannot tell you which route a call took on a shared GPU, so
    the dispatcher says -- the same argument `strategy` and `fwd_cfg`
    are recorded for.  Pinned at `_plan_record` because the kernel entry
    itself needs CUDA; `test_gpu_*` below is the end-to-end cell.
    """
    monkeypatch.setenv("SSD_MIMO_SMEM_STRATEGY", "n_split")
    token = ko.push("ssd_mimo", {"fwd_cfg": "measure", "fwd_warps": 4})
    try:
        rec = _plan_record(16, [16], 64, 64, "native")
    finally:
        ko.pop(token)
    assert rec["kernel_options"] == {"fwd_cfg": "measure", "fwd_warps": 4,
                                     "smem_strategy": "n_split"}
    assert rec["kernel_options_src"] == {"fwd_cfg": "config",
                                         "fwd_warps": "config",
                                         "smem_strategy": "env"}


def test_contract_the_record_is_empty_when_every_option_is_at_its_default(
        clean_env):
    """contract: `{}` means "family defaults".

    Recording the eight defaults on every call would only invite the
    reader to believe the caller chose them; a record, like a benchmark
    number, means exactly what it says.  An
    option set explicitly TO its default therefore does not appear
    either -- nothing was steered.
    """
    assert _plan_record(16, [16], 64, 64, "native")["kernel_options"] == {}
    token = ko.push("ssd_mimo", {"fwd_cfg": "auto"})       # == the default
    try:
        rec = _plan_record(16, [16], 64, 64, "native")
    finally:
        ko.pop(token)
    assert rec["kernel_options"] == {} and rec["kernel_options_src"] == {}


def test_contract_an_absent_or_empty_kernel_options_changes_no_lever(clean_env):
    """contract: the no-behaviour-change bar, asserted not assumed.

    Every lever the SSD family reads goes through `_env`.  With no
    scope, with an empty scope, and with `None`, all three must return
    the identical string for every lever in the inventory -- otherwise
    this surface has moved a default.
    """
    levers = ["SALL", "FWD_CFG", "FWD_CFG_ITERS", "FWD_WARPS", "FWD_STAGES",
              "FWD_MAXNREG", "SMEM_STRATEGY", "CHUNK_BWD",
              "GATE_FUSED", "BWD_HB", "SISO_FAST", "SEG",
              "ROT_WHERE", "DCS_INK", "BC_GROUP_MAJOR", "SMEM_FLOOR"]
    base = {k: _env(k, "<default>") for k in levers}
    for empty in ({}, None):
        token = ko.push("ssd_mimo", empty)
        try:
            assert {k: _env(k, "<default>") for k in levers} == base
        finally:
            ko.pop(token)


def _lm(kernel_options, seed=0):
    torch.manual_seed(seed)
    return CausalLM(vocab_size=64, layers=SSDMimoConfig(d_inner=128, d_state=16, d_ssm_in=1, d_model=64), n_layers=2,
                    execution={"backend": "torch", "kernel_options": kernel_options})


@pytest.mark.parametrize("key", sorted(NON_DEFAULT))
def test_envelope_a_kernel_option_moves_no_parameter_name_or_shape(key):
    """envelope: routing only, measured.

    The rule that makes this surface safe to extend is that no key can
    reach a parameter.  Checked the way the LM state-dict manifest
    checks a default flip: same seed, same weights, one option
    different -- names, shapes and VALUES must all be identical, so a
    checkpoint written under one routing loads under any other.
    """
    ref = _lm({}).state_dict()
    alt = _lm({key: NON_DEFAULT[key]}).state_dict()
    assert list(ref) == list(alt)
    assert [t.shape for t in ref.values()] == [t.shape for t in alt.values()]
    for name in ref:
        assert torch.equal(ref[name], alt[name]), name


@pytest.mark.parametrize("key", sorted(HM_NON_DEFAULT))
def test_envelope_an_hm_kernel_option_moves_no_parameter_name_or_shape(key):
    """envelope: routing only, for `hybrid_mamba` too:
    same seed, one option different, identical state dict."""
    def build(opts):
        torch.manual_seed(0)
        return CausalLM(vocab_size=64, layers=dict(type="hybrid_mamba", d_state=16, d_head=4), n_layers=1, d_model=64, execution={"kernel_options": opts}).state_dict()
    ref, alt = build({}), build({key: HM_NON_DEFAULT[key]})
    assert list(ref) == list(alt)
    for name in ref:
        assert torch.equal(ref[name], alt[name]), name


def test_envelope_the_option_reaches_the_core_through_the_lm_build_path():
    """envelope: the recipe actually carries it.

    A surface that validated but never arrived would pass every test
    above and steer nothing.
    """
    lm = _lm({"fwd_cfg": "off", "chunk_size_bwd": 32})
    cores = [m for m in lm.modules() if hasattr(m, "kernel_options")]
    assert cores, "no SSDMimoSSM core found in the LM"
    for core in cores:
        assert core.kernel_options == {"fwd_cfg": "off", "chunk_size_bwd": 32}


@pytest.mark.gpu
@pytest.mark.kernel_ssd
def test_gpu_the_scan_entry_records_its_options_in_dispatch_log(require_cuda,
                                                                clean_env):
    """contract: through the real kernel entry.

    Needs CUDA: `ssd_mimo_scan` asserts `x_in.is_cuda` before any plan
    is recorded, so the CPU cell above can only reach `_plan_record`.
    """
    from orbit_ssm.kernels.ssd_mimo.autograd import dispatch_log, ssd_mimo_scan
    B, H, G, R, L, N = 1, 2, 16, 1, 64, 16
    dev = "cuda"
    x = torch.randn(B, H, G, R, L, device=dev)
    delta = torch.rand(B, H, L, device=dev) * 0.1
    b = torch.randn(B, L, H, N, R, device=dev)
    c = torch.randn(B, L, H, N, R, device=dev)
    a_log = torch.rand(H, device=dev) + 1.0
    ssd_mimo_scan(x, delta, b, c, a_log, chunk_size=32, bc_per_head=True,
                  precision="tf32", kernel_options={"fwd_cfg": "off"})
    rec = dispatch_log()[-1]
    assert rec["kernel_options"] == {"fwd_cfg": "off"}
    assert rec["kernel_options_src"] == {"fwd_cfg": "config"}
    ssd_mimo_scan(x, delta, b, c, a_log, chunk_size=32, bc_per_head=True,
                  precision="tf32")
    assert dispatch_log()[-1]["kernel_options"] == {}


@pytest.mark.gpu
@pytest.mark.kernel_ssd
def test_gpu_two_stage_counts_compute_the_same_y_bitwise(require_cuda,
                                                         clean_env):
    """law: through the config.

    The launch-config axes compute the same function BY CONSTRUCTION --
    three launch metadata fields the kernel body never reads.  This is the
    claim that lets a caller set them from a recipe at all; if it ever
    fails, the key must leave this surface.  Bitwise (`torch.equal`) only
    between configs that perform identical arithmetic: here both arms pin the SAME warp count and
    differ in `fwd_stages` only, a load-pipelining depth that leaves every
    operation and its order unchanged -- at tf32 too.  A warp-count change
    re-tiles the layouts and the MMAs, so it is not bitwise; it is bounded
    by a kernel test that every real-mode-4 forward config candidate
    computes the same function.
    """
    from orbit_ssm.kernels.ssd_mimo.autograd import ssd_mimo_scan
    B, H, G, R, L, N = 1, 2, 64, 1, 128, 32
    dev = "cuda"
    torch.manual_seed(0)
    args = (torch.randn(B, H, G, R, L, device=dev),
            torch.rand(B, H, L, device=dev) * 0.1,
            torch.randn(B, L, H, N, R, device=dev),
            torch.randn(B, L, H, N, R, device=dev),
            torch.rand(H, device=dev) + 1.0)
    kw = dict(chunk_size=32, bc_per_head=True, precision="tf32")
    y_ref, s_ref = ssd_mimo_scan(*args, **kw,
                                 kernel_options={"fwd_warps": 4,
                                                 "fwd_stages": 1})
    y_alt, s_alt = ssd_mimo_scan(*args, **kw,
                                 kernel_options={"fwd_warps": 4,
                                                 "fwd_stages": 2})
    assert torch.equal(y_ref, y_alt) and torch.equal(s_ref, s_alt)


@pytest.mark.gpu
@pytest.mark.kernel_ssd
def test_gpu_both_backward_state_policies_give_the_same_gradients(require_cuda,
                                                                  clean_env):
    """law: through the config.

    `backward_states` is the one key that reroutes the BACKWARD:
    `"recompute"` replays the forward in STATE_ONLY mode, `"io"` stores
    the per-chunk states.  the replay rebuilds them to the
    route bound (a separately compiled specialisation, not promised
    bitwise), and the backward's cross-CTA reductions are atomics --
    an allowed default -- so the computation is not bitwise reproducible
    and this cell asserts a TOLERANCE and says so.  The claim it pins is that the two policies compute the same
    function, not that they sum in the same order.
    """
    from orbit_ssm.kernels.ssd_mimo.autograd import ssd_mimo_scan
    B, H, G, R, L, N = 1, 2, 64, 1, 128, 32
    dev = "cuda"

    def grads(options):
        torch.manual_seed(0)
        leaves = [torch.randn(B, H, G, R, L, device=dev),
                  torch.rand(B, H, L, device=dev) * 0.1,
                  torch.randn(B, L, H, N, R, device=dev),
                  torch.randn(B, L, H, N, R, device=dev)]
        for t in leaves:
            t.requires_grad_(True)
        a_log = torch.rand(H, device=dev) + 1.0
        y, s = ssd_mimo_scan(*leaves, a_log, chunk_size=32, bc_per_head=True,
                             precision="tf32", kernel_options=options)
        (y.float().sum() + s.float().sum()).backward()
        return [t.grad for t in leaves]

    # sum|terms| of every leaf's gradient from the fp64 token-by-token
    # oracle (same draw as `grads`), loss = sum(y) + sum(final state)
    torch.manual_seed(0)
    leaves = [torch.randn(B, H, G, R, L, device=dev),
              torch.rand(B, H, L, device=dev) * 0.1,
              torch.randn(B, L, H, N, R, device=dev),
              torch.randn(B, L, H, N, R, device=dev)]
    a_log = torch.rand(H, device=dev) + 1.0
    for t in leaves:
        t.requires_grad_(True)

    def oracle_loss(*args):
        y, h = ssd_diff_reference(*args, True)
        return y.sum() + h.sum()

    res = terms_abs(oracle_loss, *leaves, a_log)
    # The STATE_ONLY replay is a
    # separately compiled specialisation, so at a tensor-core precision the
    # two policies differ at the operand format's rounding: u = tf32, two
    # arms, typical scale sqrt(2 * depth) with depth the longest sum feeding
    # each gradient: dx: chunk T + N; ddelta: G*N + T; dB, dC: G + T.
    depths = (32 + N, G * N + 32, G + 32, G + 32)
    for i, (ref, alt, t, depth) in enumerate(zip(
            grads({"backward_states": "recompute"}),
            grads({"backward_states": "io"}), res.grad_terms, depths)):
        assert_close_terms(alt, ref, t, "tf32", math.sqrt(2 * depth),
                           name=f"grad {i}")


@pytest.mark.parametrize("bad", ["bf16", "TF32", "fp32", "", "ieee "])
def test_ssd_mimo_scan_refuses_unknown_precision_by_name(bad):
    """contract: the bare scan validates `precision` at its entry, like the
    module (ssd_mimo.py) -- an unknown string must not run as tf32."""
    import torch
    from orbit_ssm.kernels.ssd_mimo.autograd import ssd_mimo_scan
    z = torch.zeros(1)
    with pytest.raises(ValueError, match="precision must be one of"):
        ssd_mimo_scan(z, z, z, z, z, precision=bad)
