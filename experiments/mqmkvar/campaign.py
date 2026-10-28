
from __future__ import annotations
from dataclasses import dataclass, field, replace
import importlib
import inspect
import json
from functools import lru_cache
from pathlib import Path
from typing import Any
REPO = Path(__file__).resolve().parents[2]
REGISTRY = {'ssd-complex-20260925': 'experiments.configs.mq_recollect_20260925', 'ssd-real-20260925': 'experiments.configs.mq_recollect_20260925', 'hm-gates-20260925': 'experiments.configs.mq_recollect_20260925', 'hm-gateform-20260925': 'experiments.configs.mq_recollect_20260925', 'ssd-r1o1real-20260925': 'experiments.configs.mq_recollect_20260925', 'hm-screen2-20260925': 'experiments.configs.mq_recollect_20260925', 'hm-screen2b-20260925': 'experiments.configs.mq_recollect_20260925', 'hm-amu-20260925': 'experiments.configs.mq_recollect_20260925', 'attn-ref-20260925': 'experiments.configs.mq_attn_ref_20260925', 'attn-rope-20261005': 'experiments.configs.mq_attn_rope_20261005', 'attn-diag-20260925': 'experiments.configs.mq_attn_ref_20260925', 'rot-longgap-20260926': 'experiments.configs.mq_rot_longgap_20260926', 'r4o4-insane-20260926': 'experiments.configs.mq_r4o4_insane_20260926', 'gates-off-insane-20260926': 'experiments.configs.mq_gates_off_insane_20260926', 'hm-mixc-insane-20260926': 'experiments.configs.mq_hm_mixc_insane_20260926', 'blockgates-insane-20260926': 'experiments.configs.mq_blockgates_insane_20260926', 'p1final-ssd-20260927': 'experiments.configs.mq_p1final_20260927', 'p1final-hm-20260927': 'experiments.configs.mq_p1final_20260927', 'euler-insane-ssd-20260927': 'experiments.configs.mq_euler_insane_20260927', 'euler-insane-hm-20260927': 'experiments.configs.mq_euler_insane_20260927', 'p1extra-ssd-20260927': 'experiments.configs.mq_p1extra_20260927', 'p1extra-hm-20260927': 'experiments.configs.mq_p1extra_20260927', 'lrsweep-ssd-20260927': 'experiments.configs.mq_lrsweep_20260927', 'lrsweep-hm-20260927': 'experiments.configs.mq_lrsweep_20260927', 'lrsweep-attn-20260927': 'experiments.configs.mq_lrsweep_20260927', 'lrsweep-m12-20260927': 'experiments.configs.mq_lrsweep_20260927', 'lrsweep-m3-20260927': 'experiments.configs.mq_lrsweep_20260927', 'lrsweep2-ssd-20260928': 'experiments.configs.mq_lrsweep2_20260928', 'lrsweep2-hm-20260928': 'experiments.configs.mq_lrsweep2_20260928'}
REFERENCE_FAMILIES = ('attention', 'llama3')
OFFICIAL_MAMBA = 'official_mamba'
OFFICIAL_FAMILIES = ('official_mamba1', 'official_mamba2', 'official_mamba3')

def _is_reference(family: str) -> bool:
    return family in REFERENCE_FAMILIES or family == OFFICIAL_MAMBA
REMOVED_KEYS = {'rot_mode': 'a complex block in rows', 'rope_fraction': 'the size of the complex block in rows', 'a_dyn': 'rows[i].decay "head_dynamic"/"head_static"', 'fracs': 'rows', 'gates': 'use_g_b + use_g_c', 'bc_bias': 'bias_gran', 'bc_bias_heads': 'bias_gran', 'bc_norm': 'core maps (b_norm / c_norm) -- removed at mixer / LM / harness level'}
RECIPE_STEPS = {'insane': 50000, 'hard': 50016, 'easy': 50016, 'longgap': 50000}
LONGGAP_DATA = dict(train_layout=[(2048, 100000, 8)], test_layout=[(2048, 10000, 8)], vocab_size=8192, num_kv_queries=0, min_key_len=2, max_key_len=4, min_list_len=2, max_list_len=6, interleaving_frac=0.4, query_mode='streaming', random_non_queries=False, power_a=0.01, shared_token_prob=0.0)
SMOKE_STEPS, RESUME_SMOKE_STEPS = (60, 64)

def longgap_data(insane_kwargs: dict, root, smoke: bool=False) -> dict:
    dk = dict(insane_kwargs)
    dk.update(json.loads(json.dumps(LONGGAP_DATA)))
    dk['train_layout'] = [tuple(x) for x in dk['train_layout']]
    dk['test_layout'] = [tuple(x) for x in dk['test_layout']]
    dk['cache_dir'] = None if smoke else str(Path(root) / 'data_cache' / 'longgap')
    if smoke:
        dk.update(train_layout=[(2048, 256, 8)], test_layout=[(2048, 32, 8)])
    return dk

def get_campaign(campaign_id: str) -> 'Campaign':
    if campaign_id not in REGISTRY:
        raise KeyError(f'unknown MQMKVAR campaign {campaign_id!r}; registered: {sorted(REGISTRY)}')
    return importlib.import_module(REGISTRY[campaign_id]).CAMPAIGNS[campaign_id]

def _removed_in(kwargs: dict, where: str) -> list[str]:
    bad = []
    for key, value in kwargs.items():
        if key in REMOVED_KEYS:
            bad.append(f'{where}.{key} is removed (use {REMOVED_KEYS[key]})')
        if isinstance(value, dict):
            bad += _removed_in(value, f'{where}.{key}')
    return bad

def _defaults(fn) -> dict:
    return {k: p.default for k, p in inspect.signature(fn).parameters.items() if p.default is not inspect.Parameter.empty}

@dataclass(frozen=True)
class Campaign:
    id: str
    manifest: Path
    family: str
    prefix: str
    tiers: tuple
    seeds: tuple = (101, 202, 303)
    allowed_top: frozenset = frozenset()
    allowed_mixer: frozenset = frozenset()
    arm_axes: frozenset = frozenset()
    module_defaults: bool = True
    max_steps: int | None = None
    collection: tuple = ()
    definitions: tuple = field(default=(), compare=False, repr=False)
    guard: dict = field(default_factory=dict, compare=False, repr=False)
    guard_evidence: str = ''
    header: dict = field(default_factory=dict, compare=False, repr=False)
    size_rule: str = 'band'
    upstream_modules: tuple = ()

    def load(self) -> dict:
        return _load_manifest(str(self.manifest))

    @property
    def arms(self) -> tuple:
        return tuple((a['arm'] for a in self.load()['arms']))

    @property
    def jobs(self) -> tuple:
        arms = self.collection or self.arms
        return tuple(((t, a, s) for t in self.tiers for s in self.seeds for a in arms))

    @property
    def smokes(self) -> tuple:
        return tuple((j for j in self.jobs if j[2] == self.seeds[0]))

    def arm_for(self, arm: str) -> dict:
        for a in self.load()['arms']:
            if a['arm'] == arm:
                return a
        raise KeyError(f'{self.id}: no arm {arm!r} (arms: {self.arms})')

    def arm_for_spec(self, spec) -> dict:
        return self.arm_for(next((tag for tag in spec.tags if tag in self.arms)))

    def guard_config(self):
        from experiments.mqmkvar.divergence_guard import GuardConfig
        return GuardConfig.from_dict(self.load()['divergence_guard']['config'])

    def expected_steps(self, tier: str, smoke: bool=False) -> int:
        if smoke:
            return SMOKE_STEPS
        return self.max_steps if self.max_steps is not None else RECIPE_STEPS[tier]

    def spec_problems(self, kwargs: dict) -> list[str]:
        bad = _removed_in(kwargs, 'kwargs')
        bad += [f'top-level kwarg {k} not allowed' for k in sorted(set(kwargs) - self.allowed_top)]
        bad += [f'mixer kwarg {k} not allowed' for k in sorted(set(kwargs.get('mixer_kwargs', {})) - self.allowed_mixer)]
        return bad

    def build_spec(self, tier, arm, seed, root, *, smoke=False, resume_smoke=False):
        from experiments.configs.mqmkvar_baselines_20260920 import build_spec as baseline_spec
        from experiments.specs import ModelSpec
        if tier not in self.tiers or arm not in self.arms or seed not in self.seeds:
            raise ValueError(f'{self.id}: not a job {(tier, arm, seed)}')
        a = self.arm_for(arm)
        bad = self.spec_problems(a['kwargs'])
        if bad:
            raise ValueError(f'{self.id} arm {arm}: {bad}')
        s = baseline_spec('insane' if tier == 'longgap' else tier, 'official_mamba1', seed, root, smoke=smoke, resume_smoke=resume_smoke)
        if tier == 'longgap':
            s = replace(s, data=replace(s.data, kwargs=longgap_data(s.data.kwargs, root, smoke)))
        name = f'{self.prefix}_{tier}_{arm}_s{seed}' + ('_smoke' if smoke else '')
        train = replace(s.train, kwargs={**s.train.kwargs, 'default_root_dir': str(Path(root) / 'trials' / name)})
        if a.get('learning_rate') is not None:
            train = replace(train, learning_rate=float(a['learning_rate']))
        return replace(s, name=name, model=ModelSpec(a['family'], json.loads(json.dumps(a['kwargs']))), train=train, tags=('paper1', self.id, tier, arm))

    def size_problems(self, counts) -> list[str]:
        m = self.load()
        ne, budget = (counts['non_embedding'], m['budget_non_embedding'])
        if self.size_rule == 'band' and abs(ne / budget - 1) > m['tolerance']:
            return [f"non_embedding {ne} outside ±{m['tolerance']:.0%} of {budget}"]
        if self.size_rule == 'floor' and ne < budget:
            return [f'non_embedding {ne} below the floor {budget}']
        return []

    def module_default_deviations(self, model, kwargs) -> list[str]:
        from orbit_ssm import config_spec
        bad = self.spec_problems(kwargs)
        if self.family == 'ssd_mimo':
            from orbit_ssm.ssm.ssd_mimo import SSDMimoMixer as Mixer, SSDMimoSSM as Core
        else:
            from orbit_ssm.ssm.hybrid_mamba import HybridMambaMixer as Mixer, HybridMambaSSM as Core
        core_d, mix_d = (_defaults(Core.__init__), _defaults(Mixer.__init__))
        mk = kwargs.get('mixer_kwargs', {})
        spec_keys = set(config_spec.FLAT) | set(config_spec.SHORTHANDS) | {'rows', 'rotation', 'maps'}
        rc = model.resolved_config()
        for i, layer in enumerate(rc['model']['layers']):
            if layer['family'] != self.family:
                bad.append(f"layer {i} family {layer['family']}")
                continue
            core = layer['model']['core']
            want = config_spec.resolve(self.family, d_state=core['d_state'], R=core.get('R', 1), O=core.get('O'), **{k: v for k, v in mk.items() if k in spec_keys and k in self.arm_axes})
            for part in ('rows', 'rotation', 'maps'):
                if core.get(part) != want[part]:
                    bad.append(f'layer {i} core.{part}={core.get(part)!r} (from arm spec alone {want[part]!r})')
            ex = rc['execution']['layer_execution'][i]
            checks = [('core', core, core_d), ('mixer', {k: v for k, v in layer['model'].items() if k != 'core'}, mix_d), ('execution', ex['core_execution'], core_d), ('mixer-execution', {k: v for k, v in ex.items() if k != 'core_execution'}, mix_d)]
            for where, resolved, defaults in checks:
                for key, value in resolved.items():
                    if key in self.arm_axes or key in ('rows', 'rotation', 'maps') or key not in defaults:
                        continue
                    default = defaults[key]
                    if default is None or default == 'auto':
                        continue
                    if value != default:
                        bad.append(f'layer {i} {where}.{key}={value!r} (module default {default!r})')
        return bad

    def model_problems(self, model, arm: dict) -> list[str]:
        from experiments.models.zoo import count_parameter_breakdown
        bad = []
        counts = count_parameter_breakdown(model)
        for k in ('total', 'embedding', 'non_embedding'):
            if counts[k] != arm[k]:
                bad.append(f'count {k} {counts[k]} != manifest {arm[k]}')
        if _is_reference(self.family):
            if reference_record(model) != arm['resolved_model']:
                bad.append('reference model record (parameter shapes / module tree) drift vs manifest')
            if reference_geometry(self.family, arm['kwargs'], model) != arm['geometry']:
                bad.append('reference geometry drift vs manifest')
            return bad + self.size_problems(counts)
        rc = model.resolved_config()
        if rc['model'] != arm['resolved_model']:
            bad.append('resolved model record drift vs manifest')
        if rc['execution'] != arm['resolved_execution']:
            bad.append('resolved execution record drift vs manifest')
        g = arm['geometry']
        for i, layer in enumerate(model.backbone.layers):
            ssm = layer.mixer.ssm
            got = dict(d_inner=ssm.d_inner, N=ssm.N if hasattr(ssm, 'N') else ssm.d_state, heads=ssm.H)
            if self.family == 'ssd_mimo':
                got.update(R=ssm.R, O=ssm.O, P=ssm.P)
            for k, v in got.items():
                if g.get(k) != v:
                    bad.append(f'layer {i} geometry {k}={v} (manifest {g.get(k)})')
        bad += self.size_problems(counts)
        if self.module_defaults:
            bad += self.module_default_deviations(model, arm['kwargs'])
        return bad

def reference_record(model) -> dict:
    import hashlib
    return dict(cls=type(model).__name__, parameters={n: list(p.shape) for n, p in model.named_parameters()}, module_tree_sha256=hashlib.sha256(repr(model).encode()).hexdigest())

def reference_geometry(family: str, kwargs: dict, model) -> dict:
    kw = dict(kwargs)
    geometry = dict(family=family, d_model=kw.get('d_model'), n_layers=kw.get('n_layers'))
    if family == 'attention':
        mixer = model.backbone.layers[0].sequence_mixer
        geometry.update(num_heads=mixer.num_heads, head_dim=mixer.head_dim, state_mixer=type(model.backbone.layers[0].state_mixer).__name__, max_position_embeddings=model.backbone.embeddings.max_position_embeddings, tied_lm_head=model.lm_head.weight is model.backbone.embeddings.word_embeddings.weight)
    elif family == OFFICIAL_MAMBA:
        mixer = model.blocks[0].mixer
        geometry.update(d_state=kw.get('d_state'), expand_factor=kw.get('expand_factor'), headdim=kw.get('headdim'), ngroups=kw.get('ngroups'), d_inner=getattr(mixer, 'd_inner', None), nheads=getattr(mixer, 'nheads', None))
    elif family == 'llama3':
        c = model.config
        geometry.update(num_heads=c.num_attention_heads, head_dim=c.hidden_size // c.num_attention_heads, intermediate_size=c.intermediate_size, max_position_embeddings=c.max_position_embeddings, rope=dict(c.rope_parameters) if getattr(c, 'rope_parameters', None) else None, tied_lm_head=model.lm_head.weight is model.llama.embed_tokens.weight)
    return json.loads(json.dumps(geometry))

@lru_cache(maxsize=None)
def _load_manifest(path: str) -> dict:
    return json.loads(Path(path).read_text())
ANCHOR = dict(anchor='official_mamba1 N128 d256 two layers (M1), frozen baseline recipe', budget_non_embedding=1220352, tolerance=0.02, matching_rule='TASK_P1_MQ_STOCK_ABOVE_20260922: our SSMs within ±2% of the M1 backbone')
VOLATILE = ('source_git_sha',)

def _build(definition: dict):
    import warnings
    from experiments.models.zoo import build_mqmkvar_model
    from experiments.specs import DataSpec, ModelSpec
    import torch
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return build_mqmkvar_model(ModelSpec(definition['family'], json.loads(json.dumps(definition['kwargs']))), DataSpec('mqmkvar', 'lightning', {'vocab_size': 8192}))

def arm_record(campaign: Campaign, definition: dict) -> dict:
    from experiments.models.zoo import count_parameter_breakdown
    bad = campaign.spec_problems(definition['kwargs'])
    if bad:
        raise ValueError(f"{campaign.id} arm {definition['arm']}: {bad}")
    model = _build(definition)
    counts = count_parameter_breakdown(model)
    if _is_reference(campaign.family):
        old = definition['supersedes']
        record = dict(arm=definition['arm'], family=definition['family'], kwargs=definition['kwargs'], geometry=reference_geometry(campaign.family, definition['kwargs'], model), total=counts['total'], embedding=counts['embedding'], non_embedding=counts['non_embedding'], backbone=counts.get('backbone'), position_embedding=counts.get('position_embedding'), gap_percent=100 * (counts['non_embedding'] / ANCHOR['budget_non_embedding'] - 1), budget_s=definition['budget_s'], note=definition['note'], supersedes=old, count_delta_vs_superseded=None if old is None else {k: counts[k] - old[k] for k in ('total', 'embedding', 'non_embedding')}, resolved_model=reference_record(model), resolved_execution=None, module_default_deviations=None)
        if 'learning_rate' in definition:
            record['learning_rate'] = definition['learning_rate']
        return json.loads(json.dumps(record))
    rc = model.resolved_config()
    ssm = model.backbone.layers[0].mixer.ssm
    kw = definition['kwargs']
    geometry = dict(d_model=kw['d_model'], d_inner=ssm.d_inner, heads=ssm.H, P=ssm.P, N=getattr(ssm, 'N', getattr(ssm, 'd_state', None)), n_layers=kw['n_layers'])
    if campaign.family == 'ssd_mimo':
        geometry.update(R=ssm.R, O=ssm.O, G=ssm.G if hasattr(ssm, 'G') else None, expand=kw['mixer_kwargs'].get('expand'))
    core = rc['model']['layers'][0]['model']['core']
    geometry.update(rows=core['rows'], rotation=core['rotation'], maps=core['maps'])
    old = definition['supersedes']
    record = dict(arm=definition['arm'], family=definition['family'], kwargs=definition['kwargs'], geometry=geometry, total=counts['total'], embedding=counts['embedding'], non_embedding=counts['non_embedding'], backbone=counts.get('backbone'), gap_percent=100 * (counts['non_embedding'] / ANCHOR['budget_non_embedding'] - 1), budget_s=definition['budget_s'], note=definition['note'], supersedes=old, count_delta_vs_superseded=None if old is None else {k: counts[k] - old[k] for k in ('total', 'embedding', 'non_embedding')}, resolved_model=rc['model'], resolved_execution=rc['execution'])
    record['module_default_deviations'] = campaign.module_default_deviations(model, definition['kwargs']) if campaign.module_defaults else None
    if 'learning_rate' in definition:
        record['learning_rate'] = definition['learning_rate']
    return json.loads(json.dumps(record))

def _module_sha256(modules) -> dict:
    import hashlib
    import inspect
    out = {}
    for name in modules:
        path = Path(inspect.getfile(importlib.import_module(name)))
        out[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out

def generate_manifest(campaign: Campaign, *, git_sha: str | None=None) -> dict:
    from experiments.mqmkvar.divergence_guard import GuardConfig
    GuardConfig.from_dict(campaign.guard)
    manifest = dict(campaign=campaign.id, **campaign.header, source_git_sha=git_sha, spec='implementation.md implementation.3 rows / rotation / maps (main anonymous-revision-08)', recipe='experiments/configs/mqmkvar_baselines_20260920.build_spec (frozen baseline recipe)', **ANCHOR, tiers=list(campaign.tiers), seeds=list(campaign.seeds), max_steps=campaign.max_steps, collection_arms=list(campaign.collection or [d['arm'] for d in campaign.definitions]), module_defaults_rule=campaign.module_defaults, allowed_top=sorted(campaign.allowed_top), allowed_mixer=sorted(campaign.allowed_mixer), arm_axes=sorted(campaign.arm_axes), **{} if campaign.size_rule == 'band' else {'size_rule': campaign.size_rule}, upstream_module_sha256=_module_sha256(campaign.upstream_modules), divergence_guard=dict(config=dict(campaign.guard), evidence=campaign.guard_evidence), arms=[arm_record(campaign, d) for d in campaign.definitions])
    return json.loads(json.dumps(manifest))
