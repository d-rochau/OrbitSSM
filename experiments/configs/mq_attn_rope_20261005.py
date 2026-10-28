
from __future__ import annotations
from experiments.configs.mq_attn_ref_20260925 import ATTN_KWARGS, DOCS, GUARD, GUARD_EVIDENCE, UPSTREAM_ATTN
from experiments.mqmkvar.campaign import Campaign
ROPE_KWARGS = {k: v for k, v in ATTN_KWARGS.items() if k != 'max_position_embeddings'}
ROPE_KWARGS['position'] = 'rope'
ATTN_ROPE = (dict(arm='zoology-attn-d392-rope', family='attention', budget_s=9000, supersedes=None, kwargs=ROPE_KWARGS, note='attn-ref zoology-attn-d392 with learned absolute positions replaced by RoPE (theta 10000, half-rotation, full head_dim 392, on q and k); everything else identical; frozen MQMKVAR recipe'),)
CAMPAIGNS = {'attn-rope-20261005': Campaign(id='attn-rope-20261005', manifest=DOCS / 'MQ_ATTN_ROPE_20261005.json', family='attention', prefix='attn_rope', tiers=('hard', 'insane'), allowed_top=frozenset(ROPE_KWARGS), module_defaults=False, definitions=ATTN_ROPE, guard=GUARD, guard_evidence=GUARD_EVIDENCE, header=dict(status='attention RoPE ablation 2026-10-05: zoology-attn-d392 with RoPE instead of learned absolute positions, frozen MQMKVAR recipe (owner-requested clean ablation)'), size_rule='floor', upstream_modules=UPSTREAM_ATTN + ('experiments.models.rope_attention',))}
