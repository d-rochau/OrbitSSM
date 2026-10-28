
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import torch
try:
    from zoology.data.utils import DataSegment
    from zoology.config import DataSegmentConfig
    _HAS_ZOOLOGY = True
except Exception:
    _HAS_ZOOLOGY = False

    @dataclass
    class DataSegment:
        inputs: torch.Tensor
        labels: torch.Tensor
        slices: Optional[Dict[str, Any]] = None

        def __len__(self) -> int:
            return len(self.inputs)

    class DataSegmentConfig:
        name: str = ''

        def model_dump(self) -> Dict[str, Any]:
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
PAD_TOKEN = 0
FIRE_TOKEN = 1
ORDER_TOKEN_BASE = 1
OCC_TOKEN_BASE = 1

def _value_order_permutations(width: int, family: str) -> tuple[tuple[int, ...], ...]:
    if width < 2:
        raise ValueError(f'permutation value width must be >= 2, got {width}')
    if family != 'dihedral':
        raise ValueError(f"value_order_family must be 'dihedral', got {family!r}")
    identity = tuple(range(width))
    reverse = tuple(reversed(identity))
    permutations = []
    for base in (identity, reverse):
        for shift in range(width):
            permutations.append(base[shift:] + base[:shift])
    return tuple(dict.fromkeys(permutations))
LengthSpec = int | list[int] | tuple[int, int] | None

def _parse_length_spec(spec: LengthSpec, name: str) -> Optional[tuple[int, int]]:
    if spec is None:
        return None
    if isinstance(spec, int):
        lo = hi = int(spec)
    elif isinstance(spec, (list, tuple)):
        if len(spec) != 2:
            raise ValueError(f'{name} must be an int or a length-2 list/tuple [a, b], got {spec!r}.')
        lo, hi = spec
    else:
        raise TypeError(f'{name} must be an int or a length-2 list/tuple [a, b], got {type(spec).__name__}.')
    if not isinstance(lo, int) or not isinstance(hi, int):
        raise TypeError(f'{name} bounds must be integers, got {type(lo).__name__} and {type(hi).__name__}.')
    if lo < 1 or hi < 1 or lo > hi:
        raise ValueError(f'{name} must satisfy 1 <= a <= b, got [{lo}, {hi}].')
    return (int(lo), int(hi))

def _resolve_length_bounds(spec: LengthSpec, *, current_min: int, current_max: int, spec_name: str) -> tuple[int, int]:
    parsed = _parse_length_spec(spec, spec_name)
    if parsed is None:
        return (current_min, current_max)
    return parsed

@dataclass
class MQMKVARConfig(DataSegmentConfig):
    name: str = 'mqmkvar'
    vocab_size: int = 256
    num_examples: int = 100000
    input_seq_len: int = 256
    num_kv_pairs: int = 8
    num_kv_queries: int = 0
    key_len: LengthSpec = None
    min_key_len: int = 1
    max_key_len: int = 3
    interleaving_frac: float = 0.0
    prefix_pair_prob: float = 0.0
    shared_token_prob: float = 0.0
    val_len: LengthSpec = None
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

    def build(self, seed: int) -> DataSegment:
        return mqmkvar(seed=self.data_seed, **self.model_dump())

def _make_subkey(parent: Tuple[int, ...], min_key_len: int, rng: np.random.RandomState) -> Optional[Tuple[int, ...]]:
    L = len(parent)
    if L <= min_key_len:
        return None
    sub_len = rng.randint(min_key_len, L)
    max_start = L - sub_len
    start = rng.randint(0, max_start + 1)
    return parent[start:start + sub_len]

def _make_substring_overlap(keys: List[Tuple[int, ...]], min_key_len: int, max_key_len: int, key_token_choices: np.ndarray, rng: np.random.RandomState) -> Optional[Tuple[int, ...]]:
    parent = keys[rng.randint(len(keys))]
    if len(parent) < 2:
        return None
    sub_len = rng.randint(2, len(parent) + 1)
    sub_start = rng.randint(0, len(parent) - sub_len + 1)
    substring = parent[sub_start:sub_start + sub_len]
    klen = rng.randint(max(min_key_len, sub_len), max_key_len + 1)
    place = rng.randint(0, klen - sub_len + 1)
    tokens = list(rng.choice(key_token_choices, size=klen))
    tokens[place:place + sub_len] = list(substring)
    return tuple((int(t) for t in tokens))

def _generate_keys(num_keys: int, min_key_len: int, max_key_len: int, interleaving_frac: float, shared_token_prob: float, key_token_choices: np.ndarray, rng: np.random.RandomState) -> List[Tuple[int, ...]]:
    keys: List[Tuple[int, ...]] = []
    key_set: Set[Tuple[int, ...]] = set()

    def _random_key(lo: int=min_key_len, hi: int=max_key_len) -> Tuple[int, ...]:
        klen = rng.randint(lo, hi + 1)
        return tuple((int(t) for t in rng.choice(key_token_choices, size=klen)))

    def _add_unique(candidate: Tuple[int, ...]) -> bool:
        if candidate not in key_set:
            keys.append(candidate)
            key_set.add(candidate)
            return True
        return False
    if interleaving_frac > 0.0 and max_key_len > 1:
        n_large = round(num_keys * interleaving_frac / 2)
        n_large = max(0, min(n_large, num_keys // 2))
        n_random = num_keys - 2 * n_large
        large_keys: List[Tuple[int, ...]] = []
        sub_keys: List[Tuple[int, ...]] = []
        lo_large = max(min_key_len, 2)
        for _ in range(n_large):
            for _a in range(500):
                c = _random_key(lo_large, max_key_len)
                if c not in key_set:
                    key_set.add(c)
                    large_keys.append(c)
                    break
        for parent in large_keys:
            placed = False
            for _a in range(200):
                c = _make_subkey(parent, min_key_len, rng)
                if c is not None and c not in key_set:
                    key_set.add(c)
                    sub_keys.append(c)
                    placed = True
                    break
            if not placed:
                for _a in range(500):
                    c = _random_key()
                    if c not in key_set:
                        key_set.add(c)
                        sub_keys.append(c)
                        break
        random_keys: List[Tuple[int, ...]] = []
        for _ in range(n_random):
            for _a in range(500):
                c = _random_key()
                if c not in key_set:
                    key_set.add(c)
                    random_keys.append(c)
                    break
        all_keys = large_keys + sub_keys + random_keys
        perm = rng.permutation(len(all_keys))
        keys = [all_keys[i] for i in perm]
    else:
        for i in range(num_keys):
            for _a in range(200):
                candidate: Optional[Tuple[int, ...]] = None
                if i > 0 and rng.random() < shared_token_prob:
                    candidate = _make_substring_overlap(keys, min_key_len, max_key_len, key_token_choices, rng)
                if candidate is None:
                    candidate = _random_key()
                if _add_unique(candidate):
                    break
            else:
                while True:
                    c = _random_key()
                    if _add_unique(c):
                        break
    return keys

def _find_prefix_ambiguous_keys(keys: List[Tuple[int, ...]]) -> Set[int]:
    ambiguous: Set[int] = set()
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            if i != j and len(ki) < len(kj) and (kj[:len(ki)] == ki):
                ambiguous.add(i)
                break
    return ambiguous

def _sample_list_lens(n: int, distribution: str, min_len: int, max_len: int, p: float, rng: np.random.RandomState) -> np.ndarray:
    if min_len < 1:
        raise ValueError(f'min_list_len must be >= 1, got {min_len}.')
    if max_len < min_len:
        raise ValueError(f'max_list_len ({max_len}) < min_list_len ({min_len}).')
    if distribution == 'constant':
        return np.full(n, max_len, dtype=np.int64)
    if distribution == 'uniform':
        return rng.randint(min_len, max_len + 1, size=n).astype(np.int64)
    if distribution == 'geometric':
        raw = rng.geometric(p=p, size=n)
        return np.clip(raw, min_len, max_len).astype(np.int64)
    raise ValueError(f'Unknown list_len_distribution: {distribution!r}')

def _sample_powerlaw_gaps(n: int, total: int, power_a: float, rng: np.random.RandomState) -> np.ndarray:
    if total == 0:
        return np.zeros(n, dtype=np.int64)
    space = np.arange(1, total + 2)
    p = power_a * space ** (power_a - 1)
    p = p / p.sum()
    w = rng.choice(space, size=n, replace=True, p=p).astype(np.float64)
    w = w / w.sum() * total
    int_part = np.floor(w).astype(np.int64)
    rem = total - int(int_part.sum())
    if rem > 0:
        order = np.argsort(-(w - int_part))[:rem]
        int_part[order] += 1
    return int_part

def mqmkvar(vocab_size: int, num_examples: int, input_seq_len: int, seed: int, num_kv_pairs: int=8, num_kv_queries: int=0, num_aliases: int=1, num_occurrences: int=1, key_len: LengthSpec=None, min_key_len: int=1, max_key_len: int=3, prefix_pair_prob: float=0.0, shared_token_prob: float=0.0, interleaving_frac: float=0.0, interleaving_types: Optional[List[str]]=None, val_len: LengthSpec=None, max_list_len: int=4, min_list_len: int=1, list_len_distribution: str='uniform', list_len_geometric_p: float=0.5, globally_unique_values: bool=True, query_mode: str='streaming', value_order_mode: str='identity', value_order_family: str='dihedral', power_a: float=0.01, random_non_queries: bool=True, include_slices: bool=True, mask_answer_span: bool=True, record_distances: bool=False, **kwargs: Any) -> DataSegment:
    min_key_len, max_key_len = _resolve_length_bounds(key_len, current_min=min_key_len, current_max=max_key_len, spec_name='key_len')
    min_list_len, max_list_len = _resolve_length_bounds(val_len, current_min=min_list_len, current_max=max_list_len, spec_name='val_len')
    if query_mode not in ('streaming', 'fire'):
        raise ValueError(f"query_mode must be 'streaming' or 'fire', got {query_mode!r}")
    if value_order_mode not in ('identity', 'permutation'):
        raise ValueError(f"value_order_mode must be 'identity' or 'permutation', got {value_order_mode!r}")
    if value_order_mode == 'permutation' and query_mode != 'fire':
        raise ValueError("value_order_mode='permutation' requires query_mode='fire': the order token replaces FIRE and unambiguously starts the answer")
    if num_aliases < 1:
        raise ValueError(f'num_aliases must be >= 1, got {num_aliases}.')
    if num_aliases > 1 and query_mode != 'fire':
        raise ValueError("num_aliases > 1 requires query_mode='fire': streaming prefix disambiguation is undefined over concatenated alias keys")
    if num_occurrences < 1:
        raise ValueError(f'num_occurrences must be >= 1, got {num_occurrences}.')
    if num_occurrences > 1:
        if query_mode != 'fire':
            raise ValueError("num_occurrences > 1 requires query_mode='fire': the OCC_j token replaces FIRE and unambiguously starts the answer")
        if num_aliases > 1 or value_order_mode == 'permutation':
            raise ValueError("num_occurrences > 1 does not compose with num_aliases > 1 or value_order_mode='permutation' (token reservations clash)")
    if num_examples < 1:
        raise ValueError(f'num_examples must be >= 1, got {num_examples}.')
    if input_seq_len < 1:
        raise ValueError(f'input_seq_len must be >= 1, got {input_seq_len}.')
    if num_kv_pairs < 1:
        raise ValueError(f'num_kv_pairs must be >= 1, got {num_kv_pairs}.')
    if max_key_len < min_key_len or min_key_len < 1:
        raise ValueError(f'Need 1 <= min_key_len <= max_key_len, got [{min_key_len}, {max_key_len}]')
    if not 0.0 <= interleaving_frac <= 1.0:
        raise ValueError(f'interleaving_frac must lie in [0, 1], got {interleaving_frac}.')
    if not 0.0 <= shared_token_prob <= 1.0:
        raise ValueError(f'shared_token_prob must lie in [0, 1], got {shared_token_prob}.')
    if power_a <= 0.0:
        raise ValueError(f'power_a must be > 0, got {power_a}.')
    if list_len_distribution == 'geometric' and (not 0.0 < list_len_geometric_p <= 1.0):
        raise ValueError(f'list_len_geometric_p must lie in (0, 1] for geometric distribution, got {list_len_geometric_p}.')
    order_permutations: tuple[tuple[int, ...], ...] = ()
    if value_order_mode == 'permutation':
        if min_list_len != max_list_len:
            raise ValueError(f"value_order_mode='permutation' requires a fixed value-list length, got [{min_list_len}, {max_list_len}]")
        order_permutations = _value_order_permutations(max_list_len, value_order_family)
    num_order_tokens = len(order_permutations)
    min_vocab_size = num_order_tokens + 3 if value_order_mode == 'permutation' else num_occurrences + 3 if num_occurrences > 1 else 4 if query_mode == 'fire' else 3
    if vocab_size < min_vocab_size:
        raise ValueError(f'vocab_size={vocab_size} is too small for query_mode={query_mode!r}; need at least {min_vocab_size} tokens to reserve special tokens and leave at least one key token and one value token.')
    if num_kv_queries == 0:
        num_kv_queries = num_kv_pairs
    elif num_kv_queries < -1:
        raise ValueError(f'num_kv_queries={num_kv_queries} is invalid. Use 0 for the default (num_kv_pairs), -1 for the maximum safe value, or an explicit value >= num_kv_pairs.')
    _max_context_entry_len = num_aliases * max_key_len + max_list_len
    _max_query_span_len = max_key_len + max_list_len + (1 if query_mode == 'fire' else 0)
    _max_context_len = num_kv_pairs * num_occurrences * _max_context_entry_len
    _query_budget = input_seq_len - _max_context_len - 1
    _safe_max_queries = max(num_kv_pairs, _query_budget // _max_query_span_len) if _query_budget > 0 else num_kv_pairs
    if num_kv_queries == -1:
        num_kv_queries = _safe_max_queries
    if num_kv_queries < num_kv_pairs:
        raise ValueError(f'num_kv_queries={num_kv_queries} must be >= num_kv_pairs={num_kv_pairs}.')
    if num_kv_queries > _safe_max_queries:
        raise ValueError(f"input_seq_len={input_seq_len} is too short for num_kv_queries={num_kv_queries}: worst-case context takes {_max_context_len} tokens, leaving {_query_budget} for queries; at max_query_span_len={_max_query_span_len} (max_key_len={max_key_len} + max_list_len={max_list_len}{(' + 1(FIRE)' if query_mode == 'fire' else '')}) the sequence can accommodate at most {_safe_max_queries} queries.  Reduce num_kv_queries or increase input_seq_len.")
    if interleaving_frac > 0.0 and min_key_len >= max_key_len:
        raise ValueError(f'interleaving_frac={interleaving_frac} > 0 requires a strict range of key lengths (min_key_len < max_key_len), so that parent keys of length L can have subkeys of length < L that still satisfy the min_key_len constraint.  Got min_key_len={min_key_len}, max_key_len={max_key_len}.  Either set interleaving_frac=0 or widen the key-length range.')
    rng = np.random.RandomState(seed)
    num_special = 1 + num_order_tokens if value_order_mode == 'permutation' else 1 + num_occurrences if num_occurrences > 1 else 2 if query_mode == 'fire' else 1
    available = np.arange(num_special, vocab_size, dtype=np.int64)
    half = len(available) // 2
    key_token_choices = available[:half]
    value_choices = available[half:]
    max_unique_keys = sum((len(key_token_choices) ** k for k in range(min_key_len, max_key_len + 1)))
    if num_kv_pairs * num_aliases > max_unique_keys:
        raise ValueError(f'num_kv_pairs*num_aliases={num_kv_pairs}*{num_aliases} exceeds the number of unique keys possible with vocab_size={vocab_size} and key_len=[{min_key_len}, {max_key_len}] ({max_unique_keys}). Increase vocab_size or max_key_len.')
    if max_list_len > len(value_choices):
        raise ValueError(f'max_list_len={max_list_len} exceeds value vocab size {len(value_choices)} (vocab_size={vocab_size}, query_mode={query_mode!r}). Increase vocab_size or reduce max_list_len.')
    _worst_vals = num_kv_pairs * num_occurrences * max_list_len
    if globally_unique_values and _worst_vals > len(value_choices):
        raise ValueError(f'globally_unique_values=True requires num_kv_pairs * num_occurrences * max_list_len ({num_kv_pairs} * {num_occurrences} * {max_list_len} = {_worst_vals}) <= len(value_vocab) = {len(value_choices)} (vocab_size={vocab_size}). Increase vocab_size or reduce them.')
    inputs_np = np.zeros((num_examples, input_seq_len), dtype=np.int64)
    labels_np = np.full((num_examples, input_seq_len), -100, dtype=np.int64)
    if record_distances and num_occurrences != 1:
        raise ValueError('record_distances needs num_occurrences == 1 (one write per key)')
    dist_np = np.full((num_examples, input_seq_len), -1, dtype=np.int32) if record_distances else None
    list_len_running_sum = 0
    key_len_running_sum = 0
    for ex in range(num_examples):
        _pool = _generate_keys(num_kv_pairs * num_aliases, min_key_len, max_key_len, interleaving_frac, shared_token_prob, key_token_choices, rng)
        alias_keys = [_pool[i * num_aliases:(i + 1) * num_aliases] for i in range(num_kv_pairs)]
        keys = [group[0] for group in alias_keys]
        key_len_running_sum += sum((len(k) for k in _pool))
        list_lens = _sample_list_lens(num_kv_pairs, list_len_distribution, min_list_len, max_list_len, list_len_geometric_p, rng)
        list_len_running_sum += int(list_lens.sum())
        context_tokens: List[int] = []
        write_end: Dict[int, int] = {}
        value_lists: List[np.ndarray] = []
        value_lists_occ: List[List[np.ndarray]] = []
        if num_occurrences > 1:
            O = num_occurrences
            if globally_unique_values:
                total_vals = int(list_lens.sum()) * O
                all_values = rng.choice(value_choices, size=total_vals, replace=False)
            offset = 0
            for i in range(num_kv_pairs):
                L = int(list_lens[i])
                per_key: List[np.ndarray] = []
                for _j in range(O):
                    if globally_unique_values:
                        v_list = all_values[offset:offset + L]
                        offset += L
                    else:
                        v_list = rng.choice(value_choices, size=L, replace=False)
                    per_key.append(v_list)
                value_lists_occ.append(per_key)
            value_lists = [per_key[-1] for per_key in value_lists_occ]
            write_order = rng.permutation(np.repeat(np.arange(num_kv_pairs), O))
            _occ_counter = np.zeros(num_kv_pairs, dtype=np.int64)
            for i in write_order:
                _j = int(_occ_counter[i])
                _occ_counter[i] += 1
                context_tokens.extend(alias_keys[i][0])
                context_tokens.extend((int(v) for v in value_lists_occ[i][_j]))
        elif globally_unique_values:
            total_vals = int(list_lens.sum())
            all_values = rng.choice(value_choices, size=total_vals, replace=False)
            offset = 0
            for i in range(num_kv_pairs):
                L = int(list_lens[i])
                v_list = all_values[offset:offset + L]
                offset += L
                value_lists.append(v_list)
                for ak in alias_keys[i]:
                    context_tokens.extend(ak)
                context_tokens.extend((int(v) for v in v_list))
                write_end[i] = len(context_tokens) - 1
        else:
            for i in range(num_kv_pairs):
                v_list = rng.choice(value_choices, size=int(list_lens[i]), replace=False)
                value_lists.append(v_list)
                for ak in alias_keys[i]:
                    context_tokens.extend(ak)
                context_tokens.extend((int(v) for v in v_list))
                write_end[i] = len(context_tokens) - 1
        context_size = len(context_tokens)
        ambiguous_indices = _find_prefix_ambiguous_keys(keys) if query_mode == 'streaming' else set()
        _base = rng.permutation(num_kv_pairs)
        if value_order_mode == 'permutation':
            repetitions, remainder = divmod(num_kv_queries, num_kv_pairs)
            _combined = np.tile(np.arange(num_kv_pairs), repetitions)
            if remainder:
                _combined = np.concatenate([_combined, rng.permutation(num_kv_pairs)[:remainder]])
            rng.shuffle(_combined)
            query_perm = _combined
        elif num_kv_queries > num_kv_pairs:
            _extra = rng.choice(num_kv_pairs, size=num_kv_queries - num_kv_pairs, replace=True)
            _combined = np.concatenate([_base, _extra])
            rng.shuffle(_combined)
            query_perm = _combined
        else:
            query_perm = _base
        query_order_ids = np.zeros(num_kv_queries, dtype=np.int64)
        if value_order_mode == 'permutation':
            order_cycles = [rng.permutation(num_order_tokens) for _ in range(num_kv_pairs)]
            order_counts = np.zeros(num_kv_pairs, dtype=np.int64)
            for q, key_idx in enumerate(query_perm):
                count = int(order_counts[key_idx])
                if count > 0 and count % num_order_tokens == 0:
                    order_cycles[key_idx] = rng.permutation(num_order_tokens)
                query_order_ids[q] = order_cycles[key_idx][count % num_order_tokens]
                order_counts[key_idx] += 1
        query_alias_ids = np.zeros(num_kv_queries, dtype=np.int64)
        if num_aliases > 1:
            alias_cycles = [rng.permutation(num_aliases) for _ in range(num_kv_pairs)]
            alias_counts = np.zeros(num_kv_pairs, dtype=np.int64)
            for q, key_idx in enumerate(query_perm):
                count = int(alias_counts[key_idx])
                if count > 0 and count % num_aliases == 0:
                    alias_cycles[key_idx] = rng.permutation(num_aliases)
                query_alias_ids[q] = alias_cycles[key_idx][count % num_aliases]
                alias_counts[key_idx] += 1
        query_occ_ids = np.zeros(num_kv_queries, dtype=np.int64)
        if num_occurrences > 1:
            occ_cycles = [rng.permutation(num_occurrences) for _ in range(num_kv_pairs)]
            occ_counts = np.zeros(num_kv_pairs, dtype=np.int64)
            for q, key_idx in enumerate(query_perm):
                count = int(occ_counts[key_idx])
                if count > 0 and count % num_occurrences == 0:
                    occ_cycles[key_idx] = rng.permutation(num_occurrences)
                query_occ_ids[q] = occ_cycles[key_idx][count % num_occurrences]
                occ_counts[key_idx] += 1
        if query_mode == 'fire':
            span_lens = np.array([len(keys[i]) + 1 + int(list_lens[i]) for i in range(num_kv_pairs)], dtype=np.int64)
        else:
            span_lens = np.array([len(keys[i]) + int(list_lens[i]) for i in range(num_kv_pairs)], dtype=np.int64)
        if num_aliases > 1:
            total_span = int(sum((len(alias_keys[kq][int(query_alias_ids[q])]) + 1 + int(list_lens[kq]) for q, kq in enumerate(query_perm))))
        else:
            total_span = int(span_lens[query_perm].sum())
        post_context = input_seq_len - context_size
        slack = post_context - total_span - 1
        if slack < 0:
            raise ValueError(f'input_seq_len={input_seq_len} too small. Context={context_size}, query spans={total_span}, need at least {context_size + total_span + 1}.')
        gaps = _sample_powerlaw_gaps(num_kv_queries + 1, slack, power_a, rng)
        seq = np.zeros(input_seq_len, dtype=np.int64)
        seq[:context_size] = np.array(context_tokens, dtype=np.int64)
        labels = np.full(input_seq_len, -100, dtype=np.int64)
        cursor = context_size + int(gaps[0])
        for j, kq_idx in enumerate(query_perm):
            span_start = cursor
            key = alias_keys[kq_idx][int(query_alias_ids[j])]
            vs = value_lists[kq_idx]
            L_k = len(key)
            L_v = int(list_lens[kq_idx])
            for t, tok in enumerate(key):
                seq[cursor + t] = tok
            if query_mode == 'fire':
                fire_pos = cursor + L_k
                if value_order_mode == 'permutation':
                    order_id = int(query_order_ids[j])
                    seq[fire_pos] = ORDER_TOKEN_BASE + order_id
                    permutation = order_permutations[order_id]
                    output_vs = vs[np.asarray(permutation, dtype=np.int64)]
                elif num_occurrences > 1:
                    occ_id = int(query_occ_ids[j])
                    seq[fire_pos] = OCC_TOKEN_BASE + occ_id
                    output_vs = value_lists_occ[kq_idx][num_occurrences - 1 - occ_id]
                else:
                    seq[fire_pos] = FIRE_TOKEN
                    output_vs = vs
                labels[fire_pos] = int(output_vs[0])
                for vi in range(1, L_v):
                    labels[fire_pos + vi] = int(output_vs[vi])
                if mask_answer_span:
                    seq[fire_pos + 1:fire_pos + 1 + L_v] = PAD_TOKEN
                else:
                    seq[fire_pos + 1:fire_pos + 1 + L_v] = output_vs
                if dist_np is not None:
                    span = np.arange(span_start, fire_pos + L_v)
                    lab = span[labels[span] != -100]
                    dist_np[ex, lab] = lab - write_end[kq_idx]
                cursor += L_k + 1 + L_v + int(gaps[j + 1])
            else:
                last_key_pos = cursor + L_k - 1
                is_ambiguous = kq_idx in ambiguous_indices
                if is_ambiguous:
                    labels[last_key_pos] = -100
                    val_start = last_key_pos + 1
                    for vi in range(L_v):
                        labels[val_start + vi] = int(vs[vi])
                    ans_start, ans_count = (val_start, L_v)
                else:
                    labels[last_key_pos] = int(vs[0])
                    for vi in range(1, L_v):
                        labels[last_key_pos + vi] = int(vs[vi])
                    ans_start, ans_count = (last_key_pos + 1, L_v - 1)
                if mask_answer_span:
                    seq[ans_start:ans_start + ans_count] = PAD_TOKEN
                elif is_ambiguous:
                    seq[ans_start:ans_start + L_v] = vs
                elif L_v > 1:
                    seq[ans_start:ans_start + L_v - 1] = vs[1:]
                if dist_np is not None:
                    span = np.arange(span_start, span_start + int(span_lens[kq_idx]))
                    lab = span[labels[span] != -100]
                    dist_np[ex, lab] = lab - write_end[kq_idx]
                cursor += int(span_lens[kq_idx]) + int(gaps[j + 1])
        inputs_np[ex] = seq
        labels_np[ex] = labels
    inputs = torch.from_numpy(inputs_np)
    labels = torch.from_numpy(labels_np)
    if random_non_queries:
        zero_mask = (inputs == 0) & (labels == -100)
        if zero_mask.any():
            rand = torch.from_numpy(rng.choice(available, size=int(zero_mask.sum().item())))
            inputs[zero_mask] = rand
    slices_dict: Optional[Dict[str, Any]] = None
    if include_slices:
        total_keys = max(1, num_examples * num_kv_pairs)
        slices_dict = {'input_seq_len': input_seq_len, 'num_kv_pairs': num_kv_pairs, 'max_key_len': max_key_len, 'min_key_len': min_key_len, 'avg_key_len': key_len_running_sum / total_keys, 'max_list_len': max_list_len, 'min_list_len': min_list_len, 'avg_list_len': list_len_running_sum / total_keys, 'query_mode': query_mode, 'num_aliases': num_aliases, 'num_occurrences': num_occurrences, 'occ_token_base': OCC_TOKEN_BASE if num_occurrences > 1 else None, 'value_order_mode': value_order_mode, 'value_order_family': value_order_family, 'num_order_tokens': num_order_tokens, 'order_token_base': ORDER_TOKEN_BASE if num_order_tokens else None, 'order_permutations': order_permutations, 'interleaving_frac': interleaving_frac, 'shared_token_prob': shared_token_prob, 'num_kv_queries': num_kv_queries}
        if dist_np is not None:
            slices_dict['write_read_distance'] = torch.from_numpy(dist_np)
    return DataSegment(inputs=inputs, labels=labels, slices=slices_dict)

def mqmkvar_parallel(num_workers: int=4, **kwargs: Any) -> DataSegment:
    num_examples = kwargs['num_examples']
    base_seed = kwargs.get('seed', 0)
    if num_workers <= 1 or num_examples < 200:
        return mqmkvar(**kwargs)
    from concurrent.futures import ProcessPoolExecutor
    chunk_sizes = [num_examples // num_workers] * num_workers
    for i in range(num_examples % num_workers):
        chunk_sizes[i] += 1
    futures = []
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for i, cs in enumerate(chunk_sizes):
            chunk_kw = dict(kwargs)
            chunk_kw['num_examples'] = cs
            chunk_kw['seed'] = base_seed * 1000 + i
            chunk_kw['include_slices'] = i == 0
            futures.append(pool.submit(mqmkvar, **chunk_kw))
    segments = [f.result() for f in futures]
    inputs = torch.cat([s.inputs for s in segments], dim=0)
    labels = torch.cat([s.labels for s in segments], dim=0)
    return DataSegment(inputs=inputs, labels=labels, slices=segments[0].slices)

def build_mqmkvar_data(*, train_examples: int=100000, test_examples: int=3000, input_seq_len: int=256, vocab_size: int=256, num_kv_pairs: int=8, num_kv_queries: int=0, key_len: LengthSpec=None, min_key_len: int=1, max_key_len: int=3, prefix_pair_prob: float=0.0, shared_token_prob: float=0.0, interleaving_frac: float=0.0, val_len: LengthSpec=None, max_list_len: int=4, min_list_len: int=1, list_len_distribution: str='uniform', globally_unique_values: bool=True, query_mode: str='streaming', value_order_mode: str='identity', value_order_family: str='dihedral', power_a: float=0.01, random_non_queries: bool=True, mask_answer_span: bool=True, seed_train: int=0, seed_test: int=1) -> Dict[str, DataSegment]:
    common = dict(vocab_size=vocab_size, input_seq_len=input_seq_len, num_kv_pairs=num_kv_pairs, num_kv_queries=num_kv_queries, key_len=key_len, min_key_len=min_key_len, max_key_len=max_key_len, prefix_pair_prob=prefix_pair_prob, shared_token_prob=shared_token_prob, interleaving_frac=interleaving_frac, val_len=val_len, max_list_len=max_list_len, min_list_len=min_list_len, list_len_distribution=list_len_distribution, globally_unique_values=globally_unique_values, query_mode=query_mode, value_order_mode=value_order_mode, value_order_family=value_order_family, power_a=power_a, random_non_queries=random_non_queries, mask_answer_span=mask_answer_span, include_slices=True)
    return {'train': mqmkvar(num_examples=train_examples, seed=seed_train, **common), 'test': mqmkvar(num_examples=test_examples, seed=seed_test, **common)}

def make_mqmkvar_loaders(batch_size: int, *, shuffle_train: bool=True, num_workers: int=0, pin_memory: bool=False, **data_kwargs) -> Dict[str, 'torch.utils.data.DataLoader']:
    from torch.utils.data import TensorDataset, DataLoader
    data = build_mqmkvar_data(**data_kwargs)
    return {split: DataLoader(TensorDataset(seg.inputs, seg.labels), batch_size=batch_size, shuffle=shuffle_train and split == 'train', num_workers=num_workers, pin_memory=pin_memory) for split, seg in data.items()}

def visualize_mqmkvar(*, seed: int=0, example_idx: int=0, vocab_size: int=64, input_seq_len: int=64, num_kv_pairs: int=3, key_len: LengthSpec=None, min_key_len: int=1, max_key_len: int=2, val_len: LengthSpec=None, max_list_len: int=2, min_list_len: int=1, list_len_distribution: str='uniform', interleaving_frac: float=0.0, query_mode: str='streaming', value_order_mode: str='identity', value_order_family: str='dihedral', power_a: float=0.01, random_non_queries: bool=True, mask_answer_span: bool=True, globally_unique_values: bool=True, col_width: int=5, block_size: int=16, max_display: int=64) -> None:
    seg = mqmkvar(vocab_size=vocab_size, num_examples=example_idx + 1, input_seq_len=input_seq_len, seed=seed, num_kv_pairs=num_kv_pairs, key_len=key_len, min_key_len=min_key_len, max_key_len=max_key_len, val_len=val_len, max_list_len=max_list_len, min_list_len=min_list_len, list_len_distribution=list_len_distribution, interleaving_frac=interleaving_frac, query_mode=query_mode, value_order_mode=value_order_mode, value_order_family=value_order_family, power_a=power_a, random_non_queries=random_non_queries, mask_answer_span=mask_answer_span, globally_unique_values=globally_unique_values, include_slices=False)
    inp = seg.inputs[example_idx].tolist()
    lab = seg.labels[example_idx].tolist()
    L = min(len(inp), max_display)
    order_permutations = _value_order_permutations(max_list_len, value_order_family) if value_order_mode == 'permutation' else ()
    num_order_tokens = len(order_permutations)
    num_special = 1 + num_order_tokens if value_order_mode == 'permutation' else 2 if query_mode == 'fire' else 1
    available = list(range(num_special, vocab_size))
    half = len(available) // 2
    key_set = set(available[:half])
    val_set = set(available[half:])

    def _tok(t: int, l: int) -> str:
        if t == PAD_TOKEN:
            return 'PO' if l != -100 else 'PAD'
        if value_order_mode == 'permutation' and ORDER_TOKEN_BASE <= t < num_special:
            return f'OP{t - ORDER_TOKEN_BASE}'
        if query_mode == 'fire' and t == FIRE_TOKEN:
            return 'FIR'
        if t in key_set:
            return f'k{t}'
        if t in val_set:
            return f'v{t}'
        return str(t)

    def _lbl(l: int) -> str:
        if l == -100:
            return '·' * col_width
        if l in val_set:
            return f'v{l}'
        if l in key_set:
            return f'k{l}'
        return str(l)
    w = col_width
    title = f"MQMKVAR ex={example_idx}  mode={query_mode}/{value_order_mode}  vocab={vocab_size}  seq={input_seq_len}  kv={num_kv_pairs}  key=[{min_key_len},{max_key_len}]  list=[{min_list_len},{max_list_len}]  uniq_val={('Y' if globally_unique_values else 'N')}" + (f'  ilv={interleaving_frac:.0%}' if interleaving_frac > 0 else '') + (f'  [showing first {L}/{len(inp)} tokens]' if L < len(inp) else '')
    print(f'\n{title}')
    print('─' * len(title))
    print('  inp annotation:  k<N>=key  v<N>=val  PAD=filler  PO=answer-slot  FIR=fire  OP<N>=order')
    print('  lbl annotation:  v<N>=expected output  ·····=ignored (-100)')
    print()
    for start in range(0, L, block_size):
        end = min(start + block_size, L)
        idxs = range(start, end)
        print('pos: ' + ''.join((f'{i:<{w}}' for i in idxs)))
        print('inp: ' + ''.join((f'{_tok(inp[i], lab[i]):<{w}}' for i in idxs)))
        print('lbl: ' + ''.join((f'{_lbl(lab[i]):<{w}}' for i in idxs)))
        print()
