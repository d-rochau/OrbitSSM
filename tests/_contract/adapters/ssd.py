"""SSD-MIMO (`SSDMimoSSM`, kernel entry `ssd_mimo_scan`) adapter of the contract suite.

LEVEL.  The module core: `SSDMimoSSM(x, sel_x=u, gate_z=z)` with the CUDA
backend, scored against the SAME module deep-copied to fp64 with
`backend="torch"` (the chunked torch recurrence: the INDEPENDENT oracle of
`_tolerance.Scorer`+`ssd_capture`, fed the same rounded inputs and parameters), forward,
no-grad forward and EVERY gradient (inputs and parameters).  Mixer-level
routes (the combined / registered-M2 node, the mixer's own conv and norm)
are not this level; the fused output norm IS (the core takes `out_norm=`).

FEASIBILITY AND ROUTE come from the dispatcher's own functions, never from
rules restated here:
  * `infeasible(cfg)` -- `orbit_ssm.config_spec.resolve` (rows / rotation /
    maps), `SSDMimoSSM.out_norm_refusal` (called on a stand-in carrying the
    cell's attributes), `chunkpar.cp_refusal` for a forced chunk-parallel
    forward, and the lever applicability the call sites state;
  * `route(cfg)` -- `siso_route.siso_enabled / siso_refusal /
    siso_extra_refusal / dth_rot_refusal`, `chunkpar.cp_refusal +
    chunkpar_wanted | real_cp_decision` (whichever the tree has), the module's
    `_scan_chunk` / `_gluon_siso_default` chunk rule, `autograd._rot_bwd_split`,
    evaluated under the cell's own environment on shape-only (`meta`) stand-ins
    for the scan operands.  The call records what it took in
    `autograd.LAST_ROUTE`; `test_oracle` compares the two.

TOLERANCE is `_tolerance.Scorer` (the two-part rule: per-token gate at
the contraction-depth bound, sum gate x kappa1 from the fp64 reference with
kappa1 > 10 marked uninformative, signed-bias per-token at bound/4 and the
sum-level t-test over >= 8 draws for kappa1 > 2): ONE implementation.  16-bit
cells draw 8 times (>= 5 everywhere, 8 where the t-test applies), fp32 cells 3.

CALIBRATION: every cell runs with SSD_MIMO_CALIBRATION=pinned (the legal
per-arch pin, no search); `prepare()` runs tools/check_calibration.py first.
"""
from __future__ import annotations

import contextlib
import copy
import importlib.util
import os
import types

import torch

from .. import register
from ..core import Factor, Space

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
B2 = (False, True)
CX = frozenset({"half", "all"})

# geometry classes: (B, heads, L, d_model).  `narrow` is the small odd-L cell (L % T != 0: the padded tail): every cell of the
# base space runs at it, so the kernels compile once per option set, not once per geometry.
GRID = {"narrow": dict(B=2, H=4, L=200, d_model=128)}
NORM_EPS = 1e-5
SEED = 22091


def _ceil16(g):
    return -(-g // 16) * 16


# ═══════════════════════════════════════════════════════════════════════
# the factor space (production default FIRST in every domain: recipes fill
# unspecified factors with the first value)
# ═══════════════════════════════════════════════════════════════════════
def _have(mod):
    return importlib.util.find_spec(mod) is not None


def _space():
    """The base space: the options of `SSDMimoConfig` / `Execution` a user of the library chooses, at one geometry class, over
    the values the shipped recipes use.  Production default FIRST in every domain."""
    CXS = {"cplx": CX}
    F = {
        # -- geometry / compile class
        "R": (1, 2),
        "O": Factor((1,), domains=(({"R": frozenset({2})}, (2,)),)),
        "N": (128, 16),
        "G": Factor((64,), domains=(({"N": frozenset({16})}, (32,)), ({"R": frozenset({2})}, (32,)))),
        "grid": ("narrow",),
        "T": ("auto",),
        "dtype": Factor(("bf16", "fp32"), domains=(({"R": frozenset({2})}, ("bf16",)),)),    # fp32 at rank 1: its kernels compile slowly
        "prec": Factor(("io",), domains=(({"dtype": frozenset({"fp32"})}, ("tf32",)),)),
        # -- rows / rotation
        "cplx": ("real", "half"),
        "layout": Factor(("interleaved",), when=(CXS,)),
        "omega": Factor(("pair",), when=(CXS,)),
        "theta_h": Factor((True, False), when=(CXS,)),
        "freq": Factor((True, False), when=(CXS,), recipe_only=True),
        "theta_init": Factor(("kaiming",), when=(CXS,)),
        # -- maps (B/C grouping at module level = the bias granularity)
        "bias": Factor(("head_bc", "none"), domains=((CXS, ("head_bc",)),), recipe_only=True),
        "bias_init": Factor(("auto",), when=({"bias": frozenset({"head_bc", "shared"})},), recipe_only=True),
        "norms": Factor((True, False), recipe_only=True),
        "gates": ("none", "bc"),
        # -- decay
        "decay": ("head_dynamic", "head_static"),
        "a_sib": Factor((False, True), when=({"decay": frozenset({"head_dynamic"}), "R": frozenset({2})},), recipe_only=True),
        "write": ("trapezoid", "euler"),
        # -- epilogue
        "gate": ("epilogue", "none"),
        "dskip": (True, False),
        "dskip_share": Factor(("input_output", "head"), recipe_only=True),
        "out_norm": Factor((False, True), domains=(({"R": frozenset({2})}, (False,)),
                                                   ({"gate": frozenset({"none"})}, (False,))), recipe_only=True),
        "proj_bias": Factor((True, False), recipe_only=True),
        "bc_conv": Factor((False, True), recipe_only=True),
    }
    groups = [["cplx", "layout", "omega", "theta_h", "freq", "theta_init", "bias"],
              ["decay", "a_sib", "write", "dtype", "prec"],
              ["R", "O", "G", "gate", "dtype", "prec", "cplx"],
              ["gate", "dskip", "out_norm", "gates", "bias", "norms", "R"]]
    return Space(F, groups=groups, infeasible=_infeasible, route=_route)


# ═══════════════════════════════════════════════════════════════════════
# cell -> module construction (also what feasibility consults)
# ═══════════════════════════════════════════════════════════════════════
def _rows(c):
    law = c["decay"]
    if c["cplx"] == "real":
        return [{"frac": 1.0, "decay": law, "complex": False}]
    if c["cplx"] == "half":
        return [{"frac": 0.5, "decay": law, "complex": True},
                {"frac": 0.5, "decay": law, "complex": False}]
    blk = {"frac": 1.0, "decay": law, "complex": True}
    if c["gates"] == "bc":
        blk["gate"] = True              # a gate must act on some block (config_spec)
    return [blk]


def ssm_kwargs(c):
    """The `SSDMimoConfig` fields of a cell (without d_inner / d_model) and, beside them, the run-time keys of its
    `Execution` (`precision`, `chunk_size`, `backend`, `m3_kernel`: `build_layer` splits them off)."""
    g = GRID[c["grid"]]
    cx = c["cplx"] != "real"
    kw = dict(d_ssm_in=c["R"], d_ssm_out=c["O"], n_groups=c["G"], n_heads=g["H"], row_blocks=_rows(c),
              b_norm=bool(c["norms"]), c_norm=bool(c["norms"]),
              bc_bias="head" if c["bias"] == "head_bc" else c["bias"],
              b_gate=c["gates"] == "bc", c_gate=c["gates"] == "bc",
              write_rule=c["write"], pair_layout=c["layout"] if cx else "interleaved",
              skip=bool(c["dskip"]), skip_sharing=c["dskip_share"],
              a_proj_bias=bool(c["proj_bias"]) or c["decay"] != "head_dynamic", trap_proj_bias=bool(c["proj_bias"]) or c["write"] != "trapezoid",
              a_dyn_per_sibling=bool(c["a_sib"] is True),
              precision=c["prec"], proj_source="layer_input", bc_conv=bool(c["bc_conv"]),
              chunk_size=c["T"], backend="cuda", m3_kernel=c.get("m3") if c.get("m3") in ("native", "upstream") else "native")
    if c["bias"] != "none":
        kw["bc_bias_init"] = c["bias_init"]
    if cx:
        kw.update(use_theta_h=bool(c["theta_h"]), learn_theta_freq=bool(c["freq"]),
                  theta_per_head=c["omega"] == "head", omega_layout=c["omega"], theta_init=c["theta_init"])
    return kw


def build_layer(c, geo, backend=None):
    """The cell's `SSDMimoSSM` (on the default device): the config fields of `ssm_kwargs` as `SSDMimoConfig`, the
    run-time keys as the `Execution`; `backend` overrides the cell's."""
    from orbit_ssm.ssm.ssd_mimo import SSDMimoConfig, SSDMimoSSM
    kw = ssm_kwargs(c)
    ex = {k: kw.pop(k) for k in ("precision", "chunk_size", "backend")}
    ex["kernel_options"] = {"m3_kernel": kw.pop("m3_kernel")}
    if backend is not None:
        ex["backend"] = backend
    return SSDMimoSSM(SSDMimoConfig(d_model=geo["d_model"], d_inner=geo["d_inner"], d_state=c["N"], **kw), execution=ex)


def _geometry(c):
    g = GRID[c["grid"]]
    return dict(g, d_inner=g["H"] * c["G"] * c["R"], d_out=g["H"] * c["G"] * c["O"])


_SPEC_CACHE = {}


def _config_refusal(c):
    """config_spec / geometry refusals of the constructor, by the resolver's own name."""
    key = (c["R"], c["O"], c["N"], c["G"], c["cplx"], c["decay"], c["gates"], c["bias"], c["bias_init"],
           c["norms"], c["omega"], c["theta_h"], c["freq"], c["theta_init"], c["grid"], c["layout"])
    if key not in _SPEC_CACHE:
        from orbit_ssm import config_spec
        kw = ssm_kwargs(c)
        try:
            config_spec.resolve("ssd_mimo", d_state=c["N"], d_ssm_in=c["R"], d_ssm_out=c["O"],
                                **{k: kw[k] for k in config_spec.SPEC_FIELDS if k in kw})
            _SPEC_CACHE[key] = None
        except Exception as e:                                  # noqa: BLE001 -- the resolver's refusal, named
            _SPEC_CACHE[key] = f"{type(e).__name__}: {e}"
    return _SPEC_CACHE[key]


@contextlib.contextmanager
def _applied(values):
    old = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def env_of(c):
    """The environment a cell pins: the calibration pinned (the legal per-arch pin: no search)."""
    return {"SSD_MIMO_CALIBRATION": "pinned"}


# ═══════════════════════════════════════════════════════════════════════
# the dispatcher bridge
# ═══════════════════════════════════════════════════════════════════════
def _device():
    """The CUDA device the route is decided for, or None (offline)."""
    try:
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
    except Exception:                                           # noqa: BLE001
        pass
    return None


def _io_dtype(c):
    return torch.bfloat16 if c["dtype"] == "bf16" else torch.float32


def _meta(*shape, dtype=torch.float32):
    return torch.empty(*shape, device="meta", dtype=dtype)


def scan_chunk(c, device=None, needs_bwd=True):
    """The chunk T the module resolves for a call: on a GPU the module's OWN `_resolve_chunk_size` run on a stand-in
    carrying the cell's attributes and a real kernel-layout slab (so the calibrated structural rules -- the real
    R = O = 1 T rule, the rotating route -- are the dispatcher's, not restated here); offline the rank table."""
    if c["T"] != "auto":
        return int(c["T"])
    from orbit_ssm.ssm import ssd_mimo as S
    R, O, Gk = c["R"], c["O"], _ceil16(c["G"])
    dev = device if device is not None else _device()
    if dev is not None:
        g = GRID[c["grid"]]
        io = _io_dtype(c)
        ns = types.SimpleNamespace(chunk_size="auto", R=R, O=O, G_kernel=Gk, allow_chunk_autotune=True,
                                   rot_mode="m3" if c["cplx"] != "real" else None)
        x_k = torch.empty(g["B"], g["H"] * R, Gk, R, g["L"], device=dev, dtype=io, requires_grad=needs_bwd)
        kw = {"x_k": x_k}
        if c["cplx"] != "real" and c["theta_h"] is True:
            kw["thh_k"] = torch.empty(1, device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=False):
            ns._resolve_chunk_size_impl = types.MethodType(S.SSDMimoSSM._resolve_chunk_size_impl, ns)
            return int(S.SSDMimoSSM._resolve_chunk_size(ns, kw)[0])
    from orbit_ssm.kernels.ssd_mimo.siso_route import siso_enabled
    if R == 1 and O == 1 and c["cplx"] != "real" and siso_enabled(
            None, rotating=True, io_bf16=c["dtype"] == "bf16", T=64):
        return 64
    return S._scan_chunk(R, Gk, O, needs_bwd)


def _kwargs_for(fn, **cand):
    """The candidate keyword arguments `fn` accepts (the dispatcher's signatures grow; the contract follows them)."""
    import inspect
    ps = inspect.signature(fn).parameters
    return {k: v for k, v in cand.items() if k in ps}


def _environment():
    return {"tilelang": _have("tilelang"), "mamba_ssm": _have("mamba_ssm")}


def _infeasible(c, device=None):
    """None, or why NO production path serves the cell."""
    why = _config_refusal(c)
    if why:
        return why
    R, O, N, Gk = c["R"], c["O"], c["N"], _ceil16(c["G"])
    with _applied(env_of(c)):
        if c["out_norm"] is True:
            why = _out_norm_refusal(c)
            if why:
                return "fused output norm: " + why
        # (a forced CHUNKPAR=1 that cp_refusal declines is a FALLBACK to the general route -- a route() value,
        #  checked against LAST_ROUTE -- never infeasibility)
    return None


def _out_norm_refusal(c):
    """`SSDMimoSSM.out_norm_refusal` on a stand-in with the cell's attributes."""
    from orbit_ssm.ssm.ssd_mimo import SSDMimoSSM
    cx = c["cplx"] != "real"
    ns = types.SimpleNamespace(
        m3_kernel=c.get("m3") if c.get("m3") in ("native", "upstream") else "native", backend="cuda", skip=bool(c["dskip"]), R=c["R"], O=c["O"],
        r_mix_mode="none" if c["R"] == 1 else "learned", G_kernel=_ceil16(c["G"]), G=c["G"], N=c["N"],
        use_trap=c["write"] == "trapezoid", rot_mode="m3" if cx else None,
        use_theta_h=bool(c["theta_h"] is True))
    ns._dskip_route = types.MethodType(SSDMimoSSM._dskip_route, ns)
    x = types.SimpleNamespace(is_cuda=True, dtype=torch.bfloat16)
    return SSDMimoSSM.out_norm_refusal(ns, x, gate_z=object() if c["gate"] != "none" else None)


def route_detail(c, device=None, needs_bwd=True, T_used=None):
    """(route label, siso refusal, dict of the decision) of a training call,
    by the dispatcher's own functions on shape-only operands."""
    from orbit_ssm.kernels.ssd_mimo import autograd as UA, chunkpar as CP, siso_route as SR
    if c.get("m3") == "upstream":
        return "upstream", None, dict(T=64, nc=0, real_cp=False)      # upstream's own kernels: no _SSDMimoFn
    dev = device if device is not None else _device()
    R, O, N, G = c["R"], c["O"], c["N"], c["G"]
    Gk = _ceil16(G)
    g = GRID[c["grid"]]
    io = _io_dtype(c)
    rot = c["cplx"] != "real"
    with _applied(env_of(c)):
        T = int(T_used) if T_used else scan_chunk(c, dev, needs_bwd)
        nc = -(-g["L"] // T)
        Hk = g["H"] * R
        # the scan operands, shapes only: x [B, Hk, G, R, L]; B/C [B, nc, T, N, R|O]
        x_raw = _meta(g["B"], Hk, Gk, R, nc * T, dtype=io)
        Ctil, Btil = _meta(1, nc, T, N, O), _meta(1, nc, T, N, R)
        theta = _meta(g["B"], nc * T, 1, N // 2) if rot else None
        theta_h = _meta(Hk, N // 2) if (rot and c["theta_h"] is True) else None
        beta = _meta(Hk, N) if c["bias"] != "none" else None
        gamma = beta
        has_gate = c["gate"] != "none" and R == 1 and O == 1
        in_kernel = has_gate and c["gate"] == "inkernel"
        pair_half = rot and c["layout"] == "half"
        real_cp = False
        if not rot:
            cpw = CP.cp_refusal(theta=None, theta_h=None, rot_bias=None, x0=None, fuse=None,
                                dcs_ink=False, T=T, N=N, G_SIZE=Gk) is None
            if cpw:
                rcd = getattr(CP, "real_cp_decision", None)
                try:
                    if rcd is not None:
                        flags = CP.plan_flags(beta=beta, gamma=gamma, dw=object(),
                                              trap_lam=object() if c["write"] == "trapezoid" else None,
                                              d_skip=object() if c["dskip"] else None,
                                              gate_z=object() if has_gate else None, in_kernel=in_kernel)
                        cpw = bool(rcd(n_serial_ctas=g["B"] * Hk, nc=nc, T=T, N=N, G_SIZE=Gk, x_dtype=io,
                                       precision=c["prec"], device=dev, needs_bwd=needs_bwd, flags=flags))
                    else:
                        cpw = bool(CP.chunkpar_wanted(g["B"] * Hk, nc, dev))
                except Exception:                               # noqa: BLE001 -- offline: no device to calibrate against
                    cpw = False
            real_cp = cpw
        why = (None if SR.siso_enabled(dev, rotating=rot, io_bf16=io == torch.bfloat16, T=T, real_cp=real_cp,
                                       **_kwargs_for(SR.siso_enabled, bh=g["B"] * Hk, L=nc * T, needs_bwd=needs_bwd,
                                                     thh=theta_h is not None))
               else "SSD_MIMO_SISO_FAST is off (or 'auto' does not apply)")
        if why is None:
            why = SR.siso_refusal(Ctil=Ctil, Btil=Btil, x_raw=x_raw, G_SIZE=Gk, N=N, bc_grouped=False,
                                  Gb=1, Gc=1, theta_off=None, in_kernel_gate=in_kernel, T=T,
                                  theta=theta, theta_h=theta_h, rot_bias=None)
        if why is None:
            xr = _on(dev, x_raw)
            try:
                why = SR.siso_extra_refusal(x_raw=xr, precision=c["prec"], T=T, N=N, G_SIZE=Gk,
                                            pair_half=pair_half, theta=theta)
            except Exception:                                   # noqa: BLE001 -- offline: the smem limit of no device
                why = SR.siso_extra_refusal(x_raw=xr, precision="tf32", T=T, N=N, G_SIZE=Gk,
                                            pair_half=pair_half, theta=theta)
        if why is None and rot:
            why = SR.dth_rot_refusal(needs_angle_grad=needs_bwd, device=dev, io_dtype=io,
                                     bias_pre_rot=True, beta=beta, gamma=gamma, theta=theta, N=N, T=T)
        use_siso = why is None
        use_cp = False
        if use_siso and not rot:
            use_cp = CP.cp_refusal(theta=None, theta_h=None, rot_bias=None, x0=None, fuse=None,
                                   dcs_ink=False, T=T, N=N, G_SIZE=Gk) is None and real_cp
        split = bool(rot and needs_bwd and not use_siso)
        if split:
            split = bool(UA._rot_bwd_split())
            if not split and hasattr(UA, "_rot_route_split") and dev is not None:
                # the calibrated structural route may pick ROTSPLIT for a rotating R = O = 1 bf16 call
                env_ = dict(Ctil=Ctil, Btil=Btil, x_raw=x_raw, G_SIZE=Gk, N=N, bc_grouped=False, Gb=1, Gc=1,
                            theta_off=None, in_kernel_gate=in_kernel, T=T, theta=theta, theta_h=theta_h,
                            rot_bias=None)
                split = bool(UA._rot_route_split(torch.empty(g["B"], 1, 1, R, 1, device=dev, dtype=io), Hk, nc * T,
                                                 T, R_sp=R, O_sp=O, envelope=env_, pair_half=pair_half,
                                                 thh=theta_h is not None))
    label = "chunkpar" if use_cp else "siso" if use_siso else "rotsplit" if split else "general"
    return label, why, dict(T=T, nc=nc, real_cp=real_cp)


def _on(dev, t):
    """A tiny CUDA tensor of t's dtype/shape class (siso_extra_refusal reads
    `element_size()` and `.device` only)."""
    return torch.empty(1, 1, 1, 1, 1, device="cpu" if dev is None else dev, dtype=t.dtype)


def _route(c):
    return route_detail(c)[0]


# ═══════════════════════════════════════════════════════════════════════
# draw pool: the shared Scorer, finished once per cell
# ═══════════════════════════════════════════════════════════════════════
_UPSTREAM_CALL = [False]
_LAST_CFG = [None]


def release(kw):
    """Break the reference cycle `model.proj_slab` (an instance attribute bound back to the model, left by
    `_SlabCapture.native`) and drop the capture's per-token tensors, so a finished cell's GPU memory is
    freed by refcount, not by a later GC pass (a full session otherwise OOMs by the 100th cell)."""
    m, cap = kw.get("model"), kw.get("cap")
    kw.pop("tmodel", None)
    kw.pop("umodel", None)
    if m is not None:
        m.__dict__.pop("proj_slab", None)
    if cap is not None:
        cap.u = cap.g = None
        cap.__dict__.pop("gn", None)


def _free(tag="", collect=True):
    import gc
    if collect:
        gc.collect()
    if torch.cuda.is_available():
        if collect:
            torch.cuda.empty_cache()
        log = os.environ.get("SSD_CONTRACT_MEMLOG")             # diagnostic: GPU bytes still held after a cell
        if log:
            with open(log, "a") as f:
                f.write(f"{tag}\t{torch.cuda.memory_allocated() / 2 ** 20:.0f} MiB\n")


def sgn(a, b):
    """Signed normwise error ((a - b) . b) / (b . b), the Scorer's own definition."""
    a, b = a.double(), b.double()
    return float(((a - b) * b).sum() / (b * b).sum().clamp_min(1e-300))


class SsdPool:
    """The draws of one cell, scored by one `Scorer`.  ``primary=False`` is the pool of a route-pair arm (an overriding env: its
    route differs by design, so the no-grad route is not checked against the cell's predicted route)."""

    def __init__(self, c, primary=True):
        self.c = c
        self.primary = primary
        self.scorer = None
        self.draws = 0

    def scorer_for(self, B, L, no_grad_only=False):
        if self.scorer is None:
            self.scorer = _ssd_check().Scorer(B, L, self.c["dtype"] == "bf16", no_grad_only, pool_bias=self.primary)
        return self.scorer

    def extra_seeds(self):
        """The seeds of the draws that settle an UNDECIDED floor verdict (median kernel/arm ratio above the factor on fewer than
        FLOOR_MIN_DRAWS draws, `Scorer.finish`): the continuation of the cell's own seed sequence up to FLOOR_MIN_DRAWS.
        Empty when the verdict is decided."""
        if self.scorer is None:
            return []
        self.scorer.finish(say=lambda s: None)
        if not self.scorer.floor_short:
            return []
        return [SEED + 7919 * k for k in range(self.draws, _ssd_check().FLOOR_MIN_DRAWS)]

    def assert_unbiased(self):
        """Apply the pooled signed-bias checks and fail on any recorded failure."""
        if self.scorer is None:
            return
        lines = []
        fails = self.scorer.finish(say=lines.append)
        _free("oracle-end")
        assert not fails, ("ssd_check two-part rule failed:\n  " + "\n  ".join(fails[:25])
                           + "\n" + "\n".join(lines))


_SSD_CHECK = None


def _ssd_check():
    """The scorer and the reference captures the adapter uses, as one namespace (formerly the tools/ssd_check.py
    module: the two-part `Scorer` lives in tests/_tolerance.py, the captures in tests/_contract/ssd_capture.py)."""
    global _SSD_CHECK
    if _SSD_CHECK is None:
        import _tolerance
        from .. import ssd_capture
        _SSD_CHECK = types.SimpleNamespace(
            Scorer=_tolerance.Scorer, FLOOR_MIN_DRAWS=_tolerance.FLOOR_MIN_DRAWS, accuracy_bound=_tolerance.accuracy_bound, prefix_allowance=_tolerance.prefix_allowance,
            U32=_tolerance.U32, _SlabCapture=ssd_capture._SlabCapture, _PrefixCapture=ssd_capture._PrefixCapture)
    return _SSD_CHECK


# ═══════════════════════════════════════════════════════════════════════
# the adapter
# ═══════════════════════════════════════════════════════════════════════
def _full(over):
    """A complete cell from `over` (first domain value = production default)."""
    return SPACE.fill(over, lambda n, d: d[0])


def _rms(y, w, eps=NORM_EPS):
    return y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps) * w


class SSD:
    family = "ssd"
    #: refusal paths (tests/test_contract_refusal.py): the constructor refuses what `infeasible` names with these prefixes
    constructor_reasons = ("ConfigSpecError:",)

    @staticmethod
    def construct(c):
        """The cell's module on the torch backend, built (no launch): the constructor's refusals are the config
        resolver's, by name."""
        with _quiet():
            build_layer(c, _geometry(c), backend="cuda" if c.get("m3") == "upstream" else "torch")

    #: the fallback guard of the family's launch sites (tests/test_contract_refusal.py): a device error is re-raised
    fatal_guards = ("orbit_ssm.kernels.ssd_mimo.common:reraise_fatal",)
    seed = 20261004
    #: SSD has no deterministic mode (reduction atomics):
    #: the forward is compared bitwise, the backward within the fp32 accumulation-order bound
    bitwise_backward = False
    #: double backward through the scan is unsupported (no create_graph): declared, not xfail
    double_backward = "unsupported"
    #: launch selection pinned for bit-identity pairs / correctness cells; the keys that select the launch MODE are
    #: unset in the default-mode production slice (test_production_mode) and in the cross-process check
    pinned_env = {"SSD_MIMO_CALIBRATION": "pinned"}
    mode_keys = ("SSD_MIMO_CALIBRATION",)
    #: tensors measured to vary run to run, WITH the reason (criterion C): declared ONLY after measuring (Blackwell
    #: smoke: recipes + regressions, default mode, 3 fresh processes x 10 repeats).  The backward accumulates the
    #: head-shared B/C maps, the D skip, theta_h, the sibling mix and everything behind them with fp32 atomic_add over
    #: head-block CTAs (no deterministic mode).  The clock / decay projections
    #: (grad:dt_proj.*, grad:a_dyn_proj.*) are in the set too: their cotangent dDelta / dDW is accumulated with fp32
    #: atomic_add across CTAs (Kb1 + Kb2 / K2: fused_bwd_kernel.py, gluon_k2.py, gluon_kbc_hb.py), so a tensor produced through kernel atomics gets a repeat bound (test_determinism holds every
    #: gradient to the kappa-aware bound / 4), never torch.equal.  Not declared: the forward `y`.
    _ATOMIC = ("grad:A_log", "grad:B_bias", "grad:C_bias", "grad:D", "grad:du", "grad:dx", "grad:theta_h", "grad:B_proj.weight",
               "grad:C_proj.weight", "grad:B_norm.weight", "grad:C_norm.weight", "grad:r_mix", "grad:r_mix_proj.weight",
               "grad:gB_proj.weight", "grad:gC_proj.weight", "grad:trap_proj.weight", "grad:trap_proj.bias",
               "grad:bc_conv1d.conv.weight", "grad:bc_conv1d.conv.bias",
               "grad:dt_proj.weight", "grad:dt_proj.bias", "grad:a_dyn_proj.weight", "grad:a_dyn_proj.bias")
    nondeterministic = {n: "head-block fp32 atomics in the backward (no deterministic mode); spread held inside "
                           "the kappa-aware accuracy bound / 4 by test_determinism" for n in _ATOMIC}
    DETERMINISTIC = ("y", "S_final")                     # forward outputs: bitwise run to run

    environment = staticmethod(_environment)

    def __init__(self):
        self.space = SPACE

    # -- recipes: the shipped production configs, EXACTLY ------------------
    @staticmethod
    def recipes():
        arm = dict(grid="narrow", gates="bc")
        out = [
            ("ssd_mimo_r1o1_real", dict(arm, cplx="real")),
            ("ssd_mimo_r1o1_complex", dict(arm, cplx="half")),
            ("ssd_mimo_r2o2_complex", dict(arm, R=2, O=2, G=32, cplx="half")),
            # official-Mamba-3 / Mamba-2 exact presets (presets.py)
            ("m3exact", dict(grid="narrow", cplx="half", theta_h=False, freq=False, proj_bias=False, dskip_share="head")),
            ("m2exact", dict(grid="narrow", cplx="real", decay="head_static", write="euler", bias="none",
                             norms=False, dskip_share="head", bc_conv=True, out_norm=True)),
        ]
        return [(n, _full(o)) for n, o in out]

    # -- environment / draws ---------------------------------------------
    @staticmethod
    def env(c):
        return env_of(c)

    @staticmethod
    def draws(c):
        n = 8 if c["dtype"] == "bf16" else 3
        return [SEED + 7919 * k for k in range(n)]

    @staticmethod
    def prepare(strict=False):
        """Session start: tools/check_calibration.py.  A GPU without a shipped calibration is served by the generic pinned
        configuration (speed only): the base suite says so and runs; `strict` (the repository's gate) refuses to run on it."""
        if not torch.cuda.is_available():
            return
        spec = importlib.util.spec_from_file_location("_check_calibration",
                                                      os.path.join(REPO, "tools", "check_calibration.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with _applied({"SSD_MIMO_CALIBRATION": "pinned"}):
            rc = mod.main()
        if rc and strict:
            raise RuntimeError(f"tools/check_calibration.py exited {rc}: calibration missing or stale for this GPU; "
                               "the SSD contract cells would measure an uncalibrated configuration")
        if rc:
            import warnings
            warnings.warn("no shipped calibration for this GPU: the cells run on the generic pinned launch configuration "
                          "(speed only); `python tools/calibrate_kernels.py` calibrates it", stacklevel=2)

    @staticmethod
    def new_pool(c, primary=True):
        return SsdPool(c, primary)

    # -- the torch route (CPU; tests/_contract/torch_route.py) -------------------
    @staticmethod
    def torch_cells():
        from . import ssd_torch
        return ssd_torch.cells()

    @staticmethod
    def torch_layer(c, dtype, seed):
        from . import ssd_torch
        return ssd_torch.layer(c, dtype, seed)

    @staticmethod
    def torch_state_dtype_exceptions(c, route):
        from . import ssd_torch
        return ssd_torch.state_dtype_exceptions(c, route)

    # -- inputs ------------------------------------------------------------
    @staticmethod
    def build_module(c, seed, device="cuda"):
        geo = _geometry(c)
        torch.manual_seed(seed)
        with _quiet():
            m = build_layer(c, geo).to(device)
        m.set_execution(dict(backend="cuda"))
        return m, geo

    @staticmethod
    def apply_regime(m, u, c):
        """Hook: a perturbation of the parameters / inputs BEFORE the fp64 copy is taken (both arms see the same model).
        The base space draws the plain case."""

    #: the production LM spine cell of the shipped recipes (shape set: (B, d_inner, L) = (6, 2048, 2048)),
    #: used only by test_recipes_fit: the recipe must run forward + backward on the device, an OOM there is a red
    PRODUCTION = dict(B=6, L=2048)

    @classmethod
    def build_production(cls, c, seed=SEED, device="cuda"):
        """Inputs of a recipe at the production geometry (no fp64 oracle: this checks that the shipped
        configuration runs, with the default launch selection)."""
        m, geo = cls.build_module(c, seed, device)
        geo = dict(geo, **cls.PRODUCTION)
        io = _io_dtype(c)
        g = torch.Generator(device="cpu").manual_seed(seed + 1)
        B, L = geo["B"], geo["L"]
        shapes = [(B, L, geo["d_inner"]), (B, L, geo["d_model"])] + ([(B, L, geo["d_out"])] if c["gate"] != "none" else [])
        args = [torch.randn(*sh, generator=g).to(device=device, dtype=io).requires_grad_() for sh in shapes]
        kw = dict(model=m, geo=geo, cfg=c)
        if c["out_norm"] is True:
            kw["norm_w"] = torch.ones(geo["d_out"], device=device, requires_grad=True)
        return args, kw

    @classmethod
    def build(cls, c, seed=SEED, device="cuda"):
        _free("build", collect=False)            # (memory log only: with the hook cycle gone refcounting frees a draw)
        m, geo = cls.build_module(c, seed, device)
        io = _io_dtype(c)
        g = torch.Generator(device="cpu").manual_seed(seed + 1)
        B, L = geo["B"], geo["L"]

        def draw(*shape):
            return torch.randn(*shape, generator=g).to(device)
        x, u = draw(B, L, geo["d_inner"]), draw(B, L, geo["d_model"])
        z = draw(B, L, geo["d_out"])
        cls.apply_regime(m, u, c)
        tmodel = copy.deepcopy(m)                 # the SAME-DTYPE torch-backend arm (an independent floor, see `_torch_floors`)
        tmodel.set_execution(dict(backend="torch", kernel_options={"m3_kernel": "native"}))   # upstream needs auto|cuda: the torch arm is our own recurrence
        ref = copy.deepcopy(m).double()
        ref.set_execution(dict(backend="torch", kernel_options={"m3_kernel": "native"}))   # the fp64 oracle is our own chunked recurrence, whatever the arm under test
        cap = _ssd_check()._SlabCapture(ref)
        args = [t.to(io).requires_grad_() for t in (x, u) + ((z,) if c["gate"] != "none" else ())]
        umodel = None
        if cls._upstream_ok(c, m):
            umodel = copy.deepcopy(tmodel)                 # OFFICIAL Mamba-3 (mamba_ssm's SISO kernels) on this module's parameters
            umodel.set_execution(dict(backend="cuda", kernel_options={"m3_kernel": "upstream"}))
        kw = dict(model=m, ref=ref, cap=cap, geo=geo, cfg=c, tmodel=tmodel, umodel=umodel)
        if c["out_norm"] is True:
            kw["norm_w"] = (1.0 + 0.2 * draw(geo["d_out"])).requires_grad_()
        return args, kw

    # -- the call under test -----------------------------------------------
    @staticmethod
    def _autocast(c):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=c["dtype"] == "bf16")

    @classmethod
    def call(cls, args, kw, state=None, return_state=False, **extra):
        from orbit_ssm.kernels.ssd_mimo import autograd as UA
        c, m, cap = kw["cfg"], kw["model"], kw.get("cap")
        gate = args[2] if c["gate"] != "none" else None
        norm = (kw["norm_w"], NORM_EPS) if "norm_w" in kw else None
        UA.LAST_ROUTE.clear()
        _UPSTREAM_CALL[0] = c.get("m3") == "upstream"
        _LAST_CFG[0] = c
        cls._last_call = (args, kw)
        m._route_note = None
        with cls._autocast(c), (cap.native(m) if cap is not None else contextlib.nullcontext()):
            out = m.scan(args[0], sel_x=args[1], gate_z=gate, out_norm=norm, state=state, return_state=return_state)
        m.__dict__.pop("proj_slab", None)        # the context left the bound method behind: a cycle with the model
        if return_state:
            y, st = out
            return [y, st.hidden_state]
        return [out]

    @staticmethod
    def served_route():
        from orbit_ssm.kernels.ssd_mimo import autograd as UA
        r = UA.LAST_ROUTE.get("route")
        if r is None and _UPSTREAM_CALL[0]:
            return "upstream"              # the upstream provider records no route of its own (it bypasses _SSDMimoFn)
        c = _LAST_CFG[0]
        if r and c is not None:
            # the shared-memory repair ladder (`_ssd_dispatch`) re-runs the scan at a smaller chunk when the requested
            # plan does not fit: the route served is then the route AT THE CHUNK USED.  If it is exactly that route,
            # report the cell's predicted label (the ladder is the dispatcher's documented repair, recorded in
            # dispatch_log as chunk_used != chunk_requested), else the truth.
            log = UA.dispatch_log()
            rec = log[-1] if log else {}
            used, req = rec.get("chunk_used"), rec.get("chunk_requested")
            if used and req and used != req:
                try:
                    if route_detail(c, T_used=used)[0] == r:
                        return route_detail(c)[0]
                except Exception:                                   # noqa: BLE001
                    pass
        return r or "none"

    @staticmethod
    def leaves(args, kw):
        return list(args) + list(kw["model"].parameters()) + ([kw["norm_w"]] if "norm_w" in kw else [])

    @staticmethod
    def _leaf_names(c, kw):
        act = ["dx", "du"] + (["dz"] if c["gate"] != "none" else [])
        return act + [n for n, _ in kw["model"].named_parameters()] + (["norm_w"] if "norm_w" in kw else [])

    # -- the oracle ----------------------------------------------------------
    @classmethod
    def oracle(cls, c, args, kw, outs, grads, pool=None, label=""):
        from orbit_ssm.kernels.ssd_mimo import autograd as UA
        m, ref, cap, geo = kw["model"], kw["ref"], kw["cap"], kw["geo"]
        B, L = geo["B"], geo["L"]
        T = UA.LAST_ROUTE.get("T") or 64
        route0 = UA.LAST_ROUTE.get("route")
        primary = pool is not None and getattr(pool, "primary", True)   # a route-pair arm runs under an overriding env: its route differs by design
        local = pool is None
        if local:
            pool = SsdPool(c)
        scorer = pool.scorer_for(B, L)
        d = pool.draws
        pool.draws += 1
        gate = args[2] if c["gate"] != "none" else None
        rl = [a.detach().double().requires_grad_() for a in args]
        rnw = kw["norm_w"].detach().double().requires_grad_() if "norm_w" in kw else None
        pc = _ssd_check()._PrefixCapture()
        with torch.enable_grad(), cap, pc:
            y_ref = ref.scan(rl[0], sel_x=rl[1], gate_z=rl[2] if gate is not None else None)
            if rnw is not None:
                y_ref = _rms(y_ref, rnw)
        w = outs.weights[0].double()
        want = torch.autograd.grad(y_ref, [*rl, *ref.parameters(), *([rnw] if rnw is not None else [])],
                                   w, allow_unused=True)
        kappas, tok_errs = cap.kappas(), cap.token_errors()
        # the no-grad forward takes the no-grad chunk rule (outside the Gluon envelope by design)
        UA.LAST_ROUTE.clear()
        _UPSTREAM_CALL[0] = c.get("m3") == "upstream"
        _LAST_CFG[0] = c
        with torch.no_grad(), cls._autocast(c):
            y_inf = m.scan(args[0], sel_x=args[1], gate_z=gate,
                      out_norm=(kw["norm_w"], NORM_EPS) if "norm_w" in kw else None)
        T_inf = UA.dispatch_log()[-1].get("chunk_used") if UA.dispatch_log() else None
        got_nograd = cls.served_route()
        want_nograd = route_detail(c, needs_bwd=False)[0] if primary else got_nograd
        if got_nograd != want_nograd:                    # the no-grad forward takes its own route (SISO inference misroute)
            scorer.fails.append(f"draw {d}: no-grad forward served route {got_nograd!r}, expected {want_nograd!r}")
        names = cls._leaf_names(c, kw)
        n_act = 2 + (gate is not None)
        cells = [("y_nograd", y_inf, y_ref.detach(), T_inf or 64), ("y", outs.values[0].detach(), y_ref.detach(), T)]
        for i, (name, ga, gb) in enumerate(zip(names, grads, want)):
            if (ga is None) != (gb is None):
                scorer.fails.append(f"{name}: one side has no gradient")
                continue
            if ga is None or not gb.abs().max() > 0:
                continue
            cells.append((name, ga, gb, T if i < n_act else B * L))
        u = cls.u_of(c)
        kd, kth = pc.kappas(T)
        floors = (lambda: cls._floors(c, args, kw, outs, want, names, y_ref, cap, route0)) if cls._uses_floor(c) else None
        scorer.draw(d, cells, T=T, u=u, kappas=kappas, tok_errs=tok_errs, prefix=(kd, kth, L), floors=floors)
        release(kw)
        if local:
            pool.assert_unbiased()

    # -- the same-dtype torch arm (floor) ---------------------------------------
    @staticmethod
    def _uses_floor(c):
        """The 16-bit arms (bf16 IO, or tf32 operands): the flat bound is not the arithmetic's own conditioning there,
        so the kernel is also held against the torch backend run at the same dtype (see `Scorer.draw`)."""
        return c["dtype"] == "bf16" or c["prec"] == "tf32"

    @staticmethod
    def _upstream_ok(c, m):
        """The cell lies in official Mamba-3's static envelope (R = O = 1 SISO, complex pair rotation without theta_h / learned frequency,
        N in {16..128}, G*R in {32, 64, 128}, no gate, no output norm, bf16 IO at precision io): `m3_kernel='upstream'` runs its
        kernels on this module's parameters, the independent 16-bit arm of the same function."""
        try:
            return (c["dtype"] == "bf16" and c["prec"] == "io" and c["gate"] == "none" and c["out_norm"] is not True
                    and m.R == 1 and m.O == 1 and m.rot_mode == "m3" and not m.use_theta_h and not m.learn_theta_freq
                    and m.omega_layout == "pair" and m.N in (16, 32, 64, 128) and m.G in (32, 64, 128))
        except AttributeError:
            return False

    @classmethod
    def _official_floors(cls, c, args, kw, outs, want, names, y_ref, cap):
        a2 = [a.detach().clone().requires_grad_() for a in args]
        mo = kw["umodel"]
        cap2 = _ssd_check()._SlabCapture(kw["ref"])
        with torch.enable_grad(), cls._autocast(c), cap2.native(mo):
            y2 = mo.scan(a2[0], sel_x=a2[1], gate_z=None)
        g2 = torch.autograd.grad([y2], list(a2) + list(mo.parameters()), outs.weights[0], allow_unused=True)

        def rel(a, b):
            return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-300))
        out = {}
        ey = rel(y2.detach(), y_ref.detach())
        out["y"], out["y_nograd"] = (ey, None, sgn(y2.detach(), y_ref.detach()), None), (ey, None, sgn(y2.detach(), y_ref.detach()), None)
        for nm, g, w in zip(names, g2, want):
            if g is not None and w is not None and float(w.abs().max()) > 0:
                out[nm] = (rel(g, w), None, sgn(g, w), None)
        gn = getattr(cap2, "gn", None)
        if gn is not None and cap.g is not None and gn.shape == cap.g.shape:
            off = 0
            for attr, wd in zip(cap.ref._proj_attrs, cap.ref._proj_widths):
                r, n = cap.g[..., off:off + wd], gn[..., off:off + wd]
                off += wd
                den = r.norm()
                if den > 0:
                    e = float((n - r).norm() / den)
                    tsg = float(((n - r) * r).sum() / (r * r).sum())
                    for key in (attr + ".weight", attr + ".bias"):
                        if key in out:
                            out[key] = (out[key][0], e, out[key][2], tsg)
        cap2.u = cap2.g = None
        cap2.__dict__.pop("gn", None)
        mo.__dict__.pop("proj_slab", None)
        return out

    @classmethod
    def _floors(cls, c, args, kw, outs, want, names, y_ref, cap, route0):
        """The independent 16-bit arm: OUR GENERAL ROUTE (the same module, SISO / chunk-parallel switched off) where the cell is
        served by another route -- an independent bf16 implementation of the same gradient -- else the fp32-internal torch
        backend, which the record names as the fallback (a cell already on the general route has no independent 16-bit arm)."""
        if kw.get("umodel") is not None:
            try:
                out = cls._official_floors(c, args, kw, outs, want, names, y_ref, cap)
                out["__arm__"] = "official Mamba-3 (m3_kernel='upstream')"
                return out
            except Exception as e:                                # noqa: BLE001  (an envelope clause the adapter does not know: fall back, named)
                kw["umodel"] = None
        if route0 in ("siso", "chunkpar"):
            out = cls._general_floors(c, args, kw, outs, want, names, y_ref, cap)
            out["__arm__"] = f"the general route (served route was {route0!r})"
            return out
        out = cls._torch_floors(c, args, kw, outs, want, names, y_ref, cap)
        out["__arm__"] = ("the bf16-operand-emulating torch backend" if os.environ.get("SSD_CONTRACT_FLOOR_ARM") == "emul" else "the fp32-internal torch backend")+" (no independent 16-bit arm: the cell is already on the general route)"
        return out

    @classmethod
    def _general_floors(cls, c, args, kw, outs, want, names, y_ref, cap):
        a2 = [a.detach().clone().requires_grad_() for a in args]
        kw2 = dict(kw, cap=None)
        if "norm_w" in kw:
            kw2["norm_w"] = kw["norm_w"].detach().clone().requires_grad_()
        cap2 = _ssd_check()._SlabCapture(kw["ref"])
        with _applied({"SSD_MIMO_SISO_FAST": "0", "SSD_MIMO_CHUNKPAR": "0"}), cap2.native(kw["model"]):
            y2 = cls.call(a2, kw2)[0]
            g2 = torch.autograd.grad([y2], cls.leaves(a2, kw2), outs.weights[0], allow_unused=True)

        def rel(a, b):
            return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-300))
        out = {}
        ey = rel(y2.detach(), y_ref.detach())
        out["y"], out["y_nograd"] = (ey, None, sgn(y2.detach(), y_ref.detach()), None), (ey, None, sgn(y2.detach(), y_ref.detach()), None)
        for nm, g, w in zip(names, g2, want):
            if g is not None and w is not None and float(w.abs().max()) > 0:
                out[nm] = (rel(g, w), None, sgn(g, w), None)
        gn = getattr(cap2, "gn", None)
        if gn is not None and cap.g is not None and gn.shape == cap.g.shape:
            off = 0
            for attr, wd in zip(cap.ref._proj_attrs, cap.ref._proj_widths):
                r, n = cap.g[..., off:off + wd], gn[..., off:off + wd]
                off += wd
                den = r.norm()
                if den > 0:
                    e = float((n - r).norm() / den)
                    tsg = float(((n - r) * r).sum() / (r * r).sum())
                    for key in (attr + ".weight", attr + ".bias"):
                        if key in out:
                            out[key] = (out[key][0], e, out[key][2], tsg)
        cap2.u = cap2.g = None
        cap2.__dict__.pop("gn", None)
        kw["model"].__dict__.pop("proj_slab", None)
        return out

    @classmethod
    def _torch_floors(cls, c, args, kw, outs, want, names, y_ref, cap):
        """{tensor name: (normwise error, per-token slab error | None)} of the SAME-DTYPE torch-backend arm (`kw['tmodel']`,
        same parameters, same autocast, same inputs and cotangent) against the same fp64 reference."""
        tm = kw["tmodel"]
        a2 = [a.detach().clone().requires_grad_() for a in args]
        gate = a2[2] if c["gate"] != "none" else None
        capt = _ssd_check()._SlabCapture(tm)
        # the torch arm runs at the cell's OPERAND precision: tf32 matmuls where the cell asks for tf32 (an fp32-IEEE torch arm would
        # be a different arithmetic: measured ratios of 10^3-10^4 against a tf32 kernel were that, not a defect)
        restore = _operand_precision("tf32" if c["prec"] == "tf32" else "ieee")
        emul = os.environ.get("SSD_CONTRACT_FLOOR_ARM") == "emul"
        import orbit_ssm.ssm.ssd_mimo_torch as _MT
        _orig_law = _MT.token_law
        if emul:
            # bf16-OPERAND emulation: the read / write maps round to bf16 exactly where the kernel's MMA operands do (the
            # backward then contracts the rounded values); everything else stays fp32.  Used for the trapezoid lam adjoint,
            # where no independent 16-bit implementation exists.
            def _law(*a, **k):
                law = _orig_law(*a, **k)
                rb = lambda t: t.to(torch.bfloat16).to(t.dtype)
                return law._replace(Bbar=rb(law.Bbar), Cbar=rb(law.Cbar))
            _MT.token_law = _law
        try:
            with torch.enable_grad(), capt, cls._autocast(c):
                y2 = tm.scan(a2[0], sel_x=a2[1], gate_z=gate)
            leaves = list(a2) + list(tm.parameters())
            if "norm_w" in kw:
                nw = kw["norm_w"].detach().clone().requires_grad_()
                y2 = _rms(y2.float(), nw.float()).to(y2.dtype)
                leaves.append(nw)
            g2 = torch.autograd.grad(y2, leaves, outs.weights[0], allow_unused=True)
        finally:
            _MT.token_law = _orig_law
            restore()
        def rel(a, b):
            return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-300))
        out = {}
        ey = rel(y2.detach(), y_ref.detach())
        out["y"], out["y_nograd"] = (ey, None, sgn(y2.detach(), y_ref.detach()), None), (ey, None, sgn(y2.detach(), y_ref.detach()), None)
        for nm, g, w in zip(names, g2, want):
            if g is not None and w is not None and float(w.abs().max()) > 0:
                out[nm] = (rel(g, w), None, sgn(g, w), None)
        if capt.g is not None and cap.g is not None and capt.g.shape == cap.g.shape:
            off = 0
            for attr, wd in zip(cap.ref._proj_attrs, cap.ref._proj_widths):
                r, n = cap.g[..., off:off + wd], capt.g[..., off:off + wd]
                off += wd
                den = r.norm()
                if den > 0:
                    e = float((n - r).norm() / den)
                    tsg = float(((n - r) * r).sum() / (r * r).sum())
                    for key in (attr + ".weight", attr + ".bias"):
                        if key in out:
                            out[key] = (out[key][0], e, out[key][2], tsg)
        capt.g = capt.u = None
        del y2, g2, a2, leaves
        return out

    # -- determinism ----------------------------------------------------------
    @staticmethod
    def determinism_kw():
        return {}

    @staticmethod
    def u_of(c):
        return 2.0 ** -8 if c["dtype"] == "bf16" else {"ieee": 2.0 ** -24, "tf32": 2.0 ** -11}.get(c["prec"], 2.0 ** -11)

    @classmethod
    def leaf_names(cls, c, kw):
        return cls._leaf_names(c, kw)

    @classmethod
    def determinism_context(cls, c, args, kw):
        """The conditioning the atomic-order spread is measured against, from the fp64 reference (never the kernel): the
        Scorer's per-tensor kappa of the token-map slab (`_SlabCapture.kappas`) and the fp32 prefix allowance's kappa_D,
        kappa_Theta (`_PrefixCapture`), at the cotangent the determinism test uses (ones)."""
        from orbit_ssm.kernels.ssd_mimo import autograd as UA
        ref, cap = kw["ref"], kw["cap"]
        gate = args[2] if c["gate"] != "none" else None
        rl = [a.detach().double().requires_grad_() for a in args]
        pc = _ssd_check()._PrefixCapture()
        with torch.enable_grad(), cap, pc:
            y = ref.scan(rl[0], sel_x=rl[1], gate_z=rl[2] if gate is not None else None)
        torch.autograd.grad(y, [*rl, *ref.parameters()], torch.ones_like(y), allow_unused=True)
        T = UA.LAST_ROUTE.get("T") or 64
        kd, kth = pc.kappas(T)
        return dict(kappas=cap.kappas(), prefix=(kd, kth, kw["geo"]["L"]), T=T)

    @classmethod
    def grad_repeatability(cls, c, label, a, b, ctx=None, name=None):
        """Two backward runs of the same cell differ only by the order of fp32 atomics.  That spread is part of the
        accuracy budget, so it must stay inside a quarter of it (evidence rule A7; a 16-bit gradient also moves by the ulp flips the reordering causes, sqrt(delta u16) ~ 1e-4, which is why
        the reference is the accuracy bound and not u32).  The budget is the Scorer's own, so the spread is bounded by

            (accuracy_bound + fp32 prefix allowance) / 4 * max(1, kappa_name)

        where kappa_name is the per-tensor conditioning of the sum the atomics reorder (the Scorer's kappa of a slab
        parameter, from the fp64 reference's per-token terms; 1 for a tensor without a captured decomposition) -- a sum of
        n reordered fp32 terms moves by ~u32 sum|terms| / |sum|, not by a flat amount (measured: every slab
        tensor's spread is 0.1-0.9 of u32 kappa, while the flat bound / 4 = 4.2 u32 was exceeded by 15 u32 on theta_freq)."""
        bound = _ssd_check().accuracy_bound(cls.u_of(c), scan_chunk(c), c["dtype"] == "bf16")
        kap = 1.0
        if ctx is not None:
            if c["dtype"] != "bf16" and cls.u_of(c) == _ssd_check().U32:
                bound += _ssd_check().prefix_allowance(ctx["prefix"][0], ctx["prefix"][1], ctx["T"], ctx["prefix"][2])
            kap = max(1.0, ctx["kappas"].get(name, 1.0)) if name else 1.0
        bound = bound / 4 * kap
        rel = float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-300))
        assert rel <= bound, f"{label}: run-to-run backward difference {rel:.2e} > {bound:.2e} (kappa {kap:.1f})"

    # -- continuation ------------------------------------------------------------
    @staticmethod
    def continuation(c):
        return c.get("m3") != "upstream"        # the upstream batched route takes no initial state (use the step API)

    @classmethod
    def check_continuation(cls, c, label=""):
        """Whole call == two calls joined by the returned State (hidden state, rotation clock, trapezoid lag,
        B/C-conv window).  Four scored tensors against the fp64 whole-sequence oracle -- the whole kernel call
        (y_whole, h_whole: the accuracy of the state output itself) and the joined calls (y_split, h_split) -- so a
        failure names whether the CARRY or the state's own accuracy is at fault."""
        from orbit_ssm.kernels.ssd_mimo import autograd as UA
        args, kw = cls.build(c)
        m, ref, geo = kw["model"], kw["ref"], kw["geo"]
        B, L = geo["B"], geo["L"]
        cut = L // 2 + 1
        gate = args[2] if c["gate"] != "none" else None

        def sl(a, lo, hi):
            return None if a is None else a[:, lo:hi]
        with torch.no_grad(), cls._autocast(c):
            y_w, st_w = m.scan(args[0], sel_x=args[1], gate_z=gate, return_state=True)
            T = UA.dispatch_log()[-1].get("chunk_used") or 64
            y1, st = m.scan(sl(args[0], 0, cut), sel_x=sl(args[1], 0, cut), gate_z=sl(gate, 0, cut), return_state=True)
            T1 = UA.dispatch_log()[-1].get("chunk_used") or 64
            y2, st2 = m.scan(sl(args[0], cut, L), sel_x=sl(args[1], cut, L), gate_z=sl(gate, cut, L), state=st,
                        return_state=True)
        for which, carry in (("first call", st), ("joined call", st2)):
            for key in ("hidden_state", "write_prev"):        # the state contract: a returned carry is fp32 under 16-bit IO
                v = carry.get(key)
                assert v is None or v.dtype == torch.float32, f"{label} {which}: returned carry {key} is {v.dtype}, not float32"
        rl = [a.detach().double() for a in args]
        pc = _ssd_check()._PrefixCapture()
        with torch.no_grad(), pc:
            y_ref, st_ref = ref.scan(rl[0], sel_x=rl[1], gate_z=rl[2] if gate is not None else None, return_state=True)
        scorer = _ssd_check().Scorer(B, L, c["dtype"] == "bf16", no_grad_only=True)
        cells = [("y_whole", y_w, y_ref, T), ("h_whole", st_w.hidden_state, st_ref.hidden_state, T),
                 ("y_split", torch.cat([y1, y2], 1), y_ref, max(T, T1)),
                 ("h_split", st2.hidden_state, st_ref.hidden_state, max(T, T1))]
        kd, kth = pc.kappas(max(T, T1))
        scorer.draw(0, cells, T=T, u=cls.u_of(c), kappas={}, tok_errs={}, prefix=(kd, kth, L))
        lines = []
        fails = scorer.finish(say=lines.append)
        release(kw)
        _free()
        assert not fails, f"{label} continuation vs the fp64 whole-sequence oracle:\n  " + "\n  ".join(fails)

    # -- route pairs: the same cell under two routes that must agree (none in the base space) ------------------
    @staticmethod
    def route_pairs(c):
        return []

    @staticmethod
    def compiled(cell):
        """No core-level torch.compile contract: compile coverage of the SSD module lives at mixer level
        (mixer-level compile tests)."""
        return False

    # -- hooks the framework's generic checks use ---------------------------------------
    @staticmethod
    def causal_outputs(outs):
        """Indices of the outputs that are causal in the inputs: y (and its per-token gradients); the returned
        final hidden state is not."""
        return [0]

    @staticmethod
    def slice_sample(args, kw, i):
        """The cell's inputs for ONE batch sample (batch independence); leaves are fresh."""
        args = [a.detach()[i:i + 1].clone().requires_grad_() for a in args]
        kw = dict(kw)
        st = kw.get("state")                              # an initial state is per-sample too
        if st is not None:
            kw["state"] = {k: (v[i:i + 1] if torch.is_tensor(v) and v.dim() and v.shape[0] > i else v)
                           for k, v in (st.items() if isinstance(st, dict) else vars(st).items())}
        return args, kw

    @staticmethod
    def variant(args, kw, kind):
        """The same VALUES under another layout (strided / permuted / non-contiguous views).

        kinds: x_channel_major  (x as a view of a [B, d, L] buffer: token stride 1 along channels' complement),
               u_channel_major, gate_channel_major, views (every input a slice of a wider buffer whose slack
               is NaN: an out-of-bounds read poisons the result)."""
        def chan_major(t):
            return t.detach().transpose(1, 2).contiguous().transpose(1, 2).requires_grad_()

        def view(t):
            big = torch.full((*t.shape[:-1], t.shape[-1] + 5), float("nan"), device=t.device, dtype=t.dtype)
            big[..., :t.shape[-1]] = t.detach()
            return big[..., :t.shape[-1]].requires_grad_()
        out = list(args)
        if kind in ("zero_x", "denormal_x"):
            f = 0.0 if kind == "zero_x" else 1e-39
            out[0] = (args[0].detach() * f).requires_grad_()
            return out, kw
        sel = {"x_channel_major": (0,), "u_channel_major": (1,), "gate_channel_major": (2,),
               "views": tuple(range(len(args)))}[kind]
        for i in sel:
            if i < len(out):
                out[i] = view(out[i]) if kind == "views" else chan_major(out[i])
        return out, kw


    # -- hooks of the generic checks added with the framework's metamorphic / inference / reproducibility tests ----
    time_axis = 1                                        # tensors are [B, L, d]
    layout_exact = False                                 # a layout may change the load path: normwise-close, not bitwise
    causal_exact = False
    variant_kinds = ("x_channel_major", "u_channel_major", "gate_channel_major", "views")
    per_token_leaves = ("grad:dx", "grad:du", "grad:dz")

    @staticmethod
    def is_sixteen_bit(c):
        return c["dtype"] == "bf16"

    @staticmethod
    def zero_input_underflows(c):
        return False

    @staticmethod
    def length(args, kw):
        return args[0].shape[1]

    @classmethod
    def tensor_names(cls, args, kw):
        return ["y"] + ["grad:" + n for n in cls._leaf_names(kw["cfg"], kw)]

    @classmethod
    def perturb_after(cls, args, kw, t0):
        """Every input from t0 on replaced by fresh draws of the same scale."""
        out = []
        for a in args:
            w = a.detach().clone()
            w[:, t0:] = torch.randn_like(w[:, t0:]) * w.float().std().to(w.dtype)
            out.append(w.requires_grad_())
        return out, kw

    @staticmethod
    def inference_route(c):
        """The route a no-grad call must take (the forward-only chunk table, no backward-dependent refusals)."""
        return route_detail(c, needs_bwd=False)[0]

    @classmethod
    def check_forward(cls, c, args, kw, values, label=""):
        """Forward values (no gradients) against the fp64 reference, by the Scorer's activation rule."""
        from orbit_ssm.kernels.ssd_mimo import autograd as UA
        ref, geo = kw["ref"], kw["geo"]
        gate = args[2] if c["gate"] != "none" else None
        rl = [a.detach().double() for a in args]
        pc = _ssd_check()._PrefixCapture()
        with torch.no_grad(), pc:
            y_ref = ref.scan(rl[0], sel_x=rl[1], gate_z=rl[2] if gate is not None else None)
            if "norm_w" in kw:
                y_ref = _rms(y_ref, kw["norm_w"].detach().double())
        T = UA.dispatch_log()[-1].get("chunk_used") or 64
        scorer = _ssd_check().Scorer(geo["B"], geo["L"], c["dtype"] == "bf16", no_grad_only=True)
        kd, kth = pc.kappas(T)
        scorer.draw(0, [("y_nograd", values[0], y_ref, T)], T=T, u=cls.u_of(c), kappas={}, tok_errs={},
                    prefix=(kd, kth, geo["L"]))
        fails = scorer.finish(say=lambda s: None)
        release(kw)
        assert not fails, f"{label} " + "; ".join(fails)

    @classmethod
    def close(cls, c, label, a, b):
        """Two kernel results of the same values that may differ in tiling / layout / batch size: normwise relative
        agreement within TWICE the accuracy bound at the chunk depth (each arm is within one bound of the truth)."""
        assert a is not None and b is not None, label
        bound = 2 * _ssd_check().accuracy_bound(cls.u_of(c), scan_chunk(c), c["dtype"] == "bf16")
        if "grad:" in label and getattr(cls, "_last_call", None) is not None:
            # a reduced gradient moves by u x its TERMS under another layout / batch, not by u x its value: the bound
            # carries the Scorer's per-tensor conditioning kappa of the sum (fp64 reference, as `grad_repeatability`)
            args_kw = cls._last_call
            if getattr(cls, "_kappa_key", None) != id(args_kw[1]):
                cls._kappa_key, cls._kappa_ctx = id(args_kw[1]), cls.determinism_context(c, *args_kw)
            bound *= max(1.0, cls._kappa_ctx["kappas"].get(label.split("grad:")[-1], 1.0))
        a, b = a.double(), b.double()
        ref = b.norm()
        rel = float((a - b).norm() / ref.clamp_min(1e-300)) if ref > 0 else float(a.norm())
        assert rel <= bound, f"{label}: differ by {rel:.2e} > {bound:.2e}"


def _operand_precision(setting):
    """Set the fp32 matmul precision of the torch arm ("tf32" / "ieee") and return the function that restores the caller's.  A
    caller who has set the new `fp32_precision` API cannot be asked the legacy question (torch refuses it), so the new API
    is used where it is the one in force."""
    try:
        prev = torch.get_float32_matmul_precision()
    except RuntimeError:                                  # the new API is set: it is the one to read and write
        be = torch.backends.cuda.matmul
        old = be.fp32_precision
        be.fp32_precision = setting
        return lambda: setattr(be, "fp32_precision", old)
    torch.set_float32_matmul_precision("high" if setting == "tf32" else "highest")
    return lambda: torch.set_float32_matmul_precision(prev)


@contextlib.contextmanager
def _quiet():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


SPACE = _space()
register(SSD())
