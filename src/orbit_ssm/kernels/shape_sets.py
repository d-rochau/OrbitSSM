"""The ONE fixed shape set of the SSD-MIMO kernels: every calibration (families
`ssd_mimo`, `ssd_mimo_fused`, `ssd_mimo_chunkpar`, `ssd_mimo_levers`, every
group) AND the main benchmark files and scorecard scripts take their shapes
from here, so paper tables and calibrated defaults rest on the same cells.

Format (B, d_inner, L), head width P = 64, H = d_inner / 64.

GRID = size classes x shape axes.  Within a class the total size B * L *
d_inner is EQUAL (2^22, 2^25, 2^27), so the effect of an axis is isolated from
the effect of size:

    balanced   the compact reference cell of the class
    long       few channels, very long sequence (L = 32k)
    wide       few sequences, large d_inner
    batch      many short sequences (B*H up to 2048 CTAs)

LM = the application cells (the LM scorecard).

Each family fills in N / R / O / rotation / G per group.  A cell that does not
fit the device (out of memory) is DROPPED BY NAME at calibration time and the
names are recorded in the file's provenance (`dropped_shapes`).
"""
P = 64

GRID = {
    "small": {"balanced": (4, 512, 2048), "long": (1, 128, 32768),
              "wide": (1, 4096, 1024), "batch": (32, 128, 1024)},
    "medium": {"balanced": (8, 2048, 2048), "long": (1, 1024, 32768),
               "wide": (2, 8192, 2048), "batch": (64, 512, 1024)},
    "large": {"balanced": (16, 4096, 2048), "long": (2, 2048, 32768),
              "wide": (4, 8192, 4096), "batch": (128, 1024, 1024)},
}
LM = [(6, 2048, 2048), (32, 2048, 2048), (2, 2048, 16384), (6, 4096, 2048),
      (1, 4096, 32768), (8, 512, 8192)]

# tag -> (B, d_inner, L): grid tags "<class>_<axis>", LM tags "lm_b<B>_d<d>_l<L>"
SET = {"%s_%s" % (c, ax): s for c, axes in GRID.items() for ax, s in axes.items()}
LM_TAGS = tuple("lm_b%d_d%d_l%d" % s for s in LM)
SET.update({t: s for t, s in zip(LM_TAGS, LM)})
GRID_TAGS = tuple(t for t in SET if t not in LM_TAGS)


def geometry(tag):
    B, d_inner, L = SET[tag]
    return dict(B=B, L=L, H=d_inner // P, P=P)


def blocks(tag):
    B, d_inner, _ = SET[tag]
    return B * (d_inner // P)


def size(tag):
    B, d_inner, L = SET[tag]
    return B * d_inner * L


def bench_cells():
    """[(B, L, H)] of the whole set, in tag order (the benchmark / scorecard
    scripts' own cell format)."""
    return [(g["B"], g["L"], g["H"]) for g in map(geometry, SET)]


for _c, _axes in GRID.items():
    assert len({size("%s_%s" % (_c, a)) for a in _axes}) == 1, _c
assert len(SET) == 18
