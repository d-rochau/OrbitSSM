
from __future__ import annotations
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
GLUE_ITEMS = ('bias', 'bc', 'wcat', 'addmm', 'dwin', 'inproj')
GLUE_OPT_IN = ('wcat',)

def dt_fp32_on() -> bool:
    return os.environ.get('SSD_MIMO_DT_FP32', '1') != '0'

def glue_on(item: str) -> bool:
    v = os.environ.get('SSD_MIMO_GLUE', '1')
    if v == '1' or v == '':
        return item not in GLUE_OPT_IN
    if v == 'all':
        return True
    if v == '0':
        return False
    if v.startswith('all:'):
        return item not in v[4:].split(',')
    return item not in v.split(',') and item not in GLUE_OPT_IN
HM_GLUE_ITEMS = ('inproj', 'sel', 'dtk', 'zk')
HM_GLUE_OPT_IN = ('sel', 'zk')

def hm_glue_on(item: str) -> bool:
    v = os.environ.get('HM_GLUE', '1')
    if v == '1' or v == '':
        return item not in HM_GLUE_OPT_IN
    if v == 'all':
        return True
    if v == '0':
        return False
    if v.startswith('all:'):
        return item not in v[4:].split(',')
    return item not in v.split(',') and item not in HM_GLUE_OPT_IN

def param_cache_key(params, *extra):
    return tuple(extra) + tuple(((p.data_ptr(), p._version, p.requires_grad) for p in params))
