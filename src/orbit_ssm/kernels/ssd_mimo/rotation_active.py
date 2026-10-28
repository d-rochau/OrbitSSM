
import os
import triton
import triton.language as tl

def rotation_ka(n, phase='fwd', guard=True):
    from . import levers
    from .fused_fwd_kernel import _env
    raw = _env('SISO_ROT_KA', '')
    if raw != '':
        ka = int(raw)
        if ka == 0:
            return n // 2
        if ka not in (n // 4, n // 2) or n % 4:
            raise ValueError('SISO_ROT_KA supports only N/4 or N/2 active pairs')
        return ka
    ka = levers.declared_active_pairs()
    if ka is None or ka != n // 4 or n % 4 or (not guard) or (not levers.on('rot_ka_' + phase, 'complex_R1O1', guard=True)):
        return n // 2
    return ka

@triton.jit
def prefix_half(x, T: tl.constexpr, N: tl.constexpr):
    a, b = tl.split(tl.permute(tl.reshape(x, (T, 2, N // 2)), (0, 2, 1)))
    return (a, b)

@triton.jit
def join_halves(a, b, T: tl.constexpr, N: tl.constexpr):
    return tl.reshape(tl.permute(tl.join(a, b), (0, 2, 1)), (T, N))

@triton.jit
def rotate_prefix(x, cs, sn, T: tl.constexpr, N: tl.constexpr, KA: tl.constexpr):
    if KA == N // 4:
        active, real = prefix_half(x, T, N)
    else:
        active = x
    e, o = tl.split(tl.reshape(active, (T, KA, 2)))
    rotated = tl.reshape(tl.join(cs * e + sn * o, cs * o - sn * e), (T, 2 * KA))
    if KA == N // 4:
        return join_halves(rotated, real, T, N)
    else:
        return rotated

@triton.jit
def rotate_block(x, cs, sn, active, T: tl.constexpr, N: tl.constexpr, INVERSE: tl.constexpr):
    if active:
        e, o = tl.split(tl.reshape(x, (T, N // 2, 2)))
        swapped = tl.reshape(tl.join(o, e), (T, N))
        if INVERSE:
            result = cs * x - sn * swapped
        else:
            result = cs * x + sn * swapped
    else:
        result = x
    return result
