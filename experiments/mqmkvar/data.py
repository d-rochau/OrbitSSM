
from __future__ import annotations
import sys
from pathlib import Path
from typing import Any
from lightning import LightningDataModule
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT) + '/src')
from experiments.tasks.mqmkvar.builder import mqmkvar
from zoology.config import DataConfig, DataSegmentConfig
from zoology.data.utils import prepare_data

class MQMKVARSegmentConfig(DataSegmentConfig):
    name: str = 'mqmkvar'
    num_kv_pairs: int = 8
    num_kv_queries: int = 0
    num_aliases: int = 1
    num_occurrences: int = 1
    min_key_len: int = 1
    max_key_len: int = 3
    key_len: int | tuple[int, int] | None = None
    val_len: int | tuple[int, int] | None = None
    interleaving_frac: float = 0.0
    prefix_pair_prob: float = 0.0
    shared_token_prob: float = 0.0
    max_list_len: int = 4
    min_list_len: int = 1
    list_len_distribution: str = 'uniform'
    list_len_geometric_p: float = 0.5
    globally_unique_values: bool = True
    query_mode: str = 'streaming'
    value_order_mode: str = 'identity'
    value_order_family: str = 'dihedral'
    power_a: float = 0.01
    random_non_queries: bool = True
    include_slices: bool = True
    mask_answer_span: bool = True
    data_seed: int = 0
    cache_version: int = 2

    def build(self, seed: int):
        params = self.model_dump()
        params.pop('data_seed', None)
        params.pop('cache_version', None)
        return mqmkvar(seed=int(seed), **params)

def _make_mqmkvar_segment_configs(layout: list[tuple[int, int, int]], *, data_seed: int, vocab_size: int, random_non_queries: bool, power_a: float, min_key_len: int=1, max_key_len: int=3, min_list_len: int=1, max_list_len: int=4, key_len: int | tuple[int, int] | None=None, val_len: int | tuple[int, int] | None=None, num_kv_queries: int=0, interleaving_frac: float=0.0, globally_unique_values: bool=True, prefix_pair_prob: float=0.0, shared_token_prob: float=0.0, list_len_distribution: str='uniform', list_len_geometric_p: float=0.5, query_mode: str='streaming', value_order_mode: str='identity', value_order_family: str='dihedral', num_aliases: int=1, num_occurrences: int=1, generator: str='mqmkvar') -> list[DataSegmentConfig]:
    if generator == 'mqar':
        from zoology.data.multiquery_ar import MQARConfig
        if key_len not in (None, 1) or val_len not in (None, 1) or interleaving_frac != 0 or (shared_token_prob != 0) or (prefix_pair_prob != 0) or (num_aliases != 1) or (num_occurrences != 1) or (query_mode != 'streaming') or (value_order_mode != 'identity') or (not globally_unique_values):
            raise ValueError("generator='mqar' requires ordinary single-key/single-value MQAR")
        if any((num_kv_queries not in (0, kv) for _, _, kv in layout)):
            raise ValueError("generator='mqar' queries each stored key once; use num_kv_queries=0")
        return [MQARConfig(vocab_size=vocab_size, input_seq_len=length, num_examples=examples, num_kv_pairs=kv, power_a=power_a, random_non_queries=random_non_queries, include_slices=True) for length, examples, kv in layout]
    if generator != 'mqmkvar':
        raise ValueError(f'Unknown recall generator {generator!r}')
    return [MQMKVARSegmentConfig(name='mqmkvar', vocab_size=vocab_size, input_seq_len=input_seq_len, num_examples=num_examples, num_kv_pairs=num_kv_pairs, num_kv_queries=num_kv_queries, num_aliases=num_aliases, num_occurrences=num_occurrences, min_key_len=min_key_len, max_key_len=max_key_len, key_len=key_len, val_len=val_len, interleaving_frac=interleaving_frac, prefix_pair_prob=prefix_pair_prob, shared_token_prob=shared_token_prob, min_list_len=min_list_len, max_list_len=max_list_len, list_len_distribution=list_len_distribution, list_len_geometric_p=list_len_geometric_p, globally_unique_values=globally_unique_values, query_mode=query_mode, value_order_mode=value_order_mode, value_order_family=value_order_family, power_a=power_a, random_non_queries=random_non_queries, include_slices=True, mask_answer_span=True, data_seed=data_seed) for input_seq_len, num_examples, num_kv_pairs in layout]

class MQMKVARDataModule(LightningDataModule):

    def __init__(self, data_spec, train_spec, seed_spec):
        super().__init__()
        self.data_spec = data_spec
        self.train_spec = train_spec
        self.seed_spec = seed_spec
        self.data_config = None
        self.train_loader = None
        self.val_loader = None
        self.metadata: dict[str, Any] = {}

    def setup(self, stage: str | None=None) -> None:
        kwargs = dict(self.data_spec.kwargs)
        train_layout = kwargs.pop('train_layout')
        test_layout = kwargs.pop('test_layout')
        vocab_size = int(kwargs.pop('vocab_size', 8192))
        num_kv_queries = int(kwargs.pop('num_kv_queries', 0))
        interleaving_frac = float(kwargs.pop('interleaving_frac', 0.0))
        globally_unique_values = bool(kwargs.pop('globally_unique_values', True))
        random_non_queries = bool(kwargs.pop('random_non_queries', False))
        power_a = float(kwargs.pop('power_a', 0.01))
        cache_dir = kwargs.pop('cache_dir', None)
        force_cache = bool(kwargs.pop('force_cache', False))
        final_test_seed = kwargs.pop('final_test_seed', None)
        data_seed = int(self.seed_spec.data_seed)
        data_config_kwargs = dict(train_configs=_make_mqmkvar_segment_configs(train_layout, data_seed=data_seed, random_non_queries=random_non_queries, power_a=power_a, vocab_size=vocab_size, num_kv_queries=num_kv_queries, interleaving_frac=interleaving_frac, globally_unique_values=globally_unique_values, **kwargs), test_configs=_make_mqmkvar_segment_configs(test_layout, data_seed=data_seed, random_non_queries=random_non_queries, power_a=power_a, vocab_size=vocab_size, num_kv_queries=num_kv_queries, interleaving_frac=interleaving_frac, globally_unique_values=globally_unique_values, **kwargs), batch_size=(int(self.train_spec.batch_size), max(1, int(self.train_spec.batch_size) // 8)), seed=data_seed, force_cache=force_cache)
        if cache_dir is not None:
            data_config_kwargs['cache_dir'] = str(cache_dir)
        self.data_config = DataConfig(**data_config_kwargs)
        self.train_loader, self.val_loader = prepare_data(self.data_config)
        self.final_test_loader = None
        if final_test_seed is not None and int(final_test_seed) != data_seed:
            final_cfg = DataConfig(**{**data_config_kwargs, 'seed': int(final_test_seed)})
            _train_unused, self.final_test_loader = prepare_data(final_cfg)
            self.final_test_seed = int(final_test_seed)
        self.metadata = {'generator': self.data_spec.kwargs.get('generator', 'mqmkvar'), 'num_train_segments': len(self.data_config.train_configs), 'num_val_segments': len(self.data_config.test_configs), 'train_examples': sum((config.num_examples for config in self.data_config.train_configs)), 'val_examples': sum((config.num_examples for config in self.data_config.test_configs)), 'batch_size': int(self.train_spec.batch_size), 'vocab_size': vocab_size, 'final_test_seed': int(final_test_seed) if final_test_seed is not None else None}

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.final_test_loader
