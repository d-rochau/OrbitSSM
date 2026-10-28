"""The two captures the SSD contract adapter hooks into the INDEPENDENT fp64 reference (moved out of tools/ssd_check.py
so the contract suite does not depend on a tool script; tools/ssd_check.py imports them from here):

  _SlabCapture    the reference's token-map slab: its input u and the per-token gradient of its output
  _PrefixCapture  the prefix conditioning (kappa_D, kappa_Theta) of the chunked scan, read from the fp64 reference
"""
from __future__ import annotations

import weakref


class _SlabCapture:
    """Hook the INDEPENDENT reference's token-map slab: its input u and the
    per-token gradient of its output (see the module docstring)."""

    def __init__(self, ref):
        self.ref, self.u, self.g = ref, None, None

    def __enter__(self):
        import orbit_ssm.ssm.ssd_mimo_torch as T
        self._T, self._orig = T, T.token_slab
        cap = weakref.ref(self)        # hooks live in the autograd graph: a strong reference to the capture
        #                                (-> the module -> its last forward) is a cycle no Python GC can see

        def token_slab(core, u, tok_raw):
            c = cap()
            raw = c._orig(core, u, tok_raw)
            if core is c.ref and raw.requires_grad:
                c.u = u.detach().double()
                raw.register_hook(lambda g: setattr(cap(), "g", g.detach().double()))
            return raw
        T.token_slab = token_slab
        return self

    def __exit__(self, *exc):
        self._T.token_slab = self._orig

    def native(self, model):
        """Context manager hooking the NATIVE model's slab gradient."""
        import contextlib
        cap = weakref.ref(self)

        @contextlib.contextmanager
        def ctx():
            orig = model.proj_slab

            def proj_slab(u, channel_major=False):
                out = orig(u, channel_major)
                if out.requires_grad:
                    out.register_hook(lambda g: setattr(cap(), "gn", g.detach().double()))
                return out
            model.proj_slab = proj_slab
            try:
                yield
            finally:
                model.__dict__.pop("proj_slab", None)      # back to the class method (no bound-method cycle)
        return ctx()

    def token_errors(self):
        """{param name: (per-token relative error, per-token signed error) of the native
        slab gradient vs the reference's, over that parameter's columns} or {} when
        the native slab gradient was not captured."""
        gn = getattr(self, "gn", None)
        if self.g is None or gn is None or gn.shape != self.g.shape:
            return {}
        out, off = {}, 0
        for attr, w in zip(self.ref._proj_attrs, self.ref._proj_widths):
            r, n = self.g[..., off:off + w], gn[..., off:off + w]
            off += w
            den = r.norm()
            if den > 0:
                e = float((n - r).norm() / den)
                sg = float(((n - r) * r).sum() / (r * r).sum())      # per-token signed error
                mod = getattr(self.ref, attr)
                if getattr(mod, "bias", None) is not None:
                    out[attr + ".bias"] = (e, sg)
                out[attr + ".weight"] = (e, sg)
        return out

    def kappas(self):
        """{param name: kappa >= 1} for every parameter of the slab."""
        import torch
        if self.g is None:
            return {}
        out, off = {}, 0
        g = self.g.reshape(-1, self.g.shape[-1])
        u = self.u.reshape(-1, self.u.shape[-1]) if self.u is not None else None
        for attr, w in zip(self.ref._proj_attrs, self.ref._proj_widths):
            gc = g[:, off:off + w]
            off += w
            mod = getattr(self.ref, attr)
            if getattr(mod, "bias", None) is not None and mod.bias.requires_grad:
                s = gc.sum(0).norm()
                out[attr + ".bias"] = max(1.0, float(gc.abs().sum(0).norm() / s)) if s > 0 else 1.0
            if u is not None and mod.weight.requires_grad and u.shape[-1] == mod.weight.shape[1]:
                s = (gc.T @ u).norm()
                out[attr + ".weight"] = (max(1.0, float((gc.abs().T @ u.abs()).norm() / s))
                                         if s > 0 else 1.0)
        return out


class _PrefixCapture:
    """The prefix conditioning of the chunked scan, read from the INDEPENDENT fp64 reference (never from the kernel).

    DERIVATION of the fp32 allowance (the A13 prefix term of the regime-grid bounds).  The kernels form the pair
    decay exp(D_i - D_j) and the rotation from fp32 PREFIX sums: D, the decay prefix from the chunk start, and Theta,
    the running phase from t = 0.  A prefix of k terms carries an absolute error ~ u32 sqrt(k) |prefix| at the typical
    (u-scaled) rounding, and an absolute error in D_i - D_j is a RELATIVE error of exp(D_i - D_j) and of every term
    it scales, likewise for the phase.  With kappa_D = the largest chunk-local prefix sum of the per-step log-decay
    c_t = -dA_t over the tokens that still pair with a representable decay (c_t <= 88: a token whose own step exceeds
    88 pairs with nothing but itself, so it adds no conditioning) and kappa_Theta = max_t |Theta_t| (the UNWRAPPED
    running phase), the allowance on a normwise-relative error is

        allow = 2 u32 (sqrt(T) kappa_D + sqrt(L) kappa_Theta),

    u-scaling: it is linear in u32 = 2^-24, so it is invisible at a 16-bit format's bound (u16 = 2^-8; measured:
    16 to 68,000 x the allowance on every failing 16-bit cell, tools/ssd_check.py history) and covers every
    fp32-ieee cell that exceeded the flat bound (27 of 27 tensor-cells).  Only the quantities the reference itself
    computes enter: `token_law`'s `dA` [B, L, H, R] and `incr` [B, L, H, R, K]."""

    def __init__(self):
        self.dA, self.incr = [], []

    def __enter__(self):
        import orbit_ssm.ssm.ssd_mimo_torch as T
        self._T, self._orig = T, T.token_law
        rec = self

        def token_law(*a, **k):
            law = rec._orig(*a, **k)
            if getattr(law, "dA", None) is not None:
                rec.dA.append(law.dA.detach())
            if getattr(law, "incr", None) is not None:
                rec.incr.append(law.incr.detach())
            return law
        T.token_law = token_law
        return self

    def __exit__(self, *exc):
        self._T.token_law = self._orig

    def kappas(self, T):
        """(kappa_D at chunk length T, kappa_Theta) of everything recorded."""
        import torch
        kd = 0.0
        for dA in self.dA:
            c = (-dA).double().clamp_min(0).movedim(1, -1).reshape(-1, dA.shape[1])      # [..., L]
            pad = (-c.shape[-1]) % T
            cs = torch.nn.functional.pad(c, (0, pad)).view(c.shape[0], -1, T).cumsum(-1)
            live = torch.nn.functional.pad(c, (0, pad), value=1e9).view(c.shape[0], -1, T) <= 88.0
            kd = max(kd, float((cs * live).amax()))
        kth = 0.0
        for inc in self.incr:
            kth = max(kth, float(inc.double().cumsum(dim=1).abs().amax()))
        return kd, kth
