
import math
from numbers import Real
from ..configuration import positive_int

def resolve_mixer_width(d_model, expand, d_inner=None):
    d_model = positive_int(d_model, 'd_model')
    width = None if d_inner is None else positive_int(d_inner, 'd_inner')
    if expand is None:
        if width is None:
            raise ValueError('expand=None requires an explicit d_inner')
        return width
    if isinstance(expand, bool) or not isinstance(expand, Real) or (not math.isfinite(expand)) or (expand <= 0):
        raise ValueError('expand must be a positive finite number, or None with d_inner')
    product = d_model * expand
    if not math.isfinite(product):
        raise ValueError('d_model * expand must be finite')
    derived = positive_int(int(product), 'expanded d_inner')
    if width is not None and width != derived:
        raise ValueError(f'd_inner={width} conflicts with int(d_model*expand)={derived}; use expand=None for explicit width')
    return derived
