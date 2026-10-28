
P = 64
GRID = {'small': {'balanced': (4, 512, 2048), 'long': (1, 128, 32768), 'wide': (1, 4096, 1024), 'batch': (32, 128, 1024)}, 'medium': {'balanced': (8, 2048, 2048), 'long': (1, 1024, 32768), 'wide': (2, 8192, 2048), 'batch': (64, 512, 1024)}, 'large': {'balanced': (16, 4096, 2048), 'long': (2, 2048, 32768), 'wide': (4, 8192, 4096), 'batch': (128, 1024, 1024)}}
LM = [(6, 2048, 2048), (32, 2048, 2048), (2, 2048, 16384), (6, 4096, 2048), (1, 4096, 32768), (8, 512, 8192)]
SET = {'%s_%s' % (c, ax): s for c, axes in GRID.items() for ax, s in axes.items()}
LM_TAGS = tuple(('lm_b%d_d%d_l%d' % s for s in LM))
SET.update({t: s for t, s in zip(LM_TAGS, LM)})
GRID_TAGS = tuple((t for t in SET if t not in LM_TAGS))

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
    return [(g['B'], g['L'], g['H']) for g in map(geometry, SET)]
for _c, _axes in GRID.items():
    assert len({size('%s_%s' % (_c, a)) for a in _axes}) == 1, _c
assert len(SET) == 18
