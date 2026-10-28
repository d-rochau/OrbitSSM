
from __future__ import annotations
import json
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator
import numpy as np
import torch
DTYPE = np.uint32
_READ_THREADS_DEFAULT = 8
_PREFETCH_DEFAULT = 4

def _load_shard_names(token_dir: str, split: str, val_shards: int) -> list[str]:
    man_path = os.path.join(token_dir, 'manifest.json')
    if os.path.exists(man_path):
        names = [s['file'] for s in json.load(open(man_path))['shards']]
    else:
        names = sorted((f for f in os.listdir(token_dir) if f.startswith('tok_') and f.endswith('.bin')))
    if not names:
        raise FileNotFoundError(f'no token shards in {token_dir}')
    if split == 'train':
        return names[:-val_shards] if val_shards else names
    held_out = names[-val_shards:] if val_shards else []
    if split == 'val':
        return held_out[:-1] if len(held_out) > 1 else held_out
    if split == 'test':
        return held_out[-1:]
    raise ValueError(f"split must be 'train', 'val' or 'test', got {split!r}")

class FineWebTokens:

    def __init__(self, token_dir: str, seq_len: int=2048, split: str='train', val_shards: int=2, read_threads: int=_READ_THREADS_DEFAULT):
        self.token_dir = token_dir
        self.seq_len = seq_len
        self.split = split
        self.read_threads = int(read_threads)
        self._fds: list[int] | None = None
        self._pool: ThreadPoolExecutor | None = None
        self._reader_lock = threading.Lock()
        names = _load_shard_names(token_dir, split, val_shards)
        self.shard_names = names
        self.memmaps = [np.memmap(os.path.join(token_dir, n), dtype=DTYPE, mode='r') for n in names]
        self.lengths = np.array([m.size for m in self.memmaps], dtype=np.int64)
        self.total_tokens = int(self.lengths.sum())
        self.usable = self.lengths - (seq_len + 1)
        keep = self.usable > 0
        if not keep.any():
            raise ValueError(f'no shard has >= seq_len+1={seq_len + 1} tokens')
        self.idx_random = np.nonzero(keep)[0]
        self.weights = self.usable[self.idx_random].astype(np.float64)
        self.weights /= self.weights.sum()
        self.windows_per_shard = np.maximum(0, (self.lengths - 1) // seq_len).astype(np.int64)
        self.total_windows = int(self.windows_per_shard.sum())
        self.cumwin = np.cumsum(self.windows_per_shard)
        print(f'[FineWebTokens:{split}] {len(names)} shards, {self.total_tokens:,} tokens, {self.total_windows:,} unique windows (seq_len={seq_len})')

    def get_batch(self, batch_size: int, device: str | torch.device='cpu', generator: torch.Generator | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        L = self.seq_len
        seed = None if generator is None else int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator).item())
        rng = np.random.default_rng(seed)
        chosen = rng.choice(self.idx_random, size=batch_size, p=self.weights)
        xs = np.empty((batch_size, L), dtype=np.int64)
        ys = np.empty((batch_size, L), dtype=np.int64)
        for b, si in enumerate(chosen):
            start = int(rng.integers(0, self.usable[si] + 1))
            w = np.asarray(self.memmaps[si][start:start + L + 1], dtype=np.int64)
            xs[b] = w[:-1]
            ys[b] = w[1:]
        return self._to_torch(xs, ys, device)

    def _to_torch(self, xs: np.ndarray, ys: np.ndarray, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(xs)
        y = torch.from_numpy(ys)
        if str(device) != 'cpu':
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        return (x, y)

    def _ensure_readers(self) -> None:
        if self._fds is not None:
            return
        with self._reader_lock:
            if self._fds is not None:
                return
            fds = [os.open(os.path.join(self.token_dir, n), os.O_RDONLY) for n in self.shard_names]
            if self.read_threads > 1:
                self._pool = ThreadPoolExecutor(max_workers=self.read_threads, thread_name_prefix=f'fwread-{self.split}')
            self._fds = fds

    def _locate(self, window_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        shard_idx = np.searchsorted(self.cumwin, window_ids, side='right')
        local_w = window_ids - np.where(shard_idx > 0, self.cumwin[shard_idx - 1], 0)
        return (shard_idx, local_w)

    def _read_window_ids(self, window_ids: np.ndarray, use_pread: bool=True) -> tuple[np.ndarray, np.ndarray]:
        B = len(window_ids)
        L = self.seq_len
        xs = np.empty((B, L), dtype=np.int64)
        ys = np.empty((B, L), dtype=np.int64)
        shard_idx, local_w = self._locate(window_ids)
        if not use_pread:
            for b in range(B):
                arr = np.asarray(self.memmaps[int(shard_idx[b])][int(local_w[b]) * L:int(local_w[b]) * L + L + 1], dtype=np.int64)
                xs[b] = arr[:-1]
                ys[b] = arr[1:]
            return (xs, ys)
        self._ensure_readers()
        nbytes = (L + 1) * DTYPE().itemsize

        def one(b: int) -> None:
            buf = os.pread(self._fds[int(shard_idx[b])], nbytes, int(local_w[b]) * L * DTYPE().itemsize)
            if len(buf) != nbytes:
                raise IOError(f'short read: window {int(window_ids[b])} got {len(buf)} of {nbytes} bytes')
            arr = np.frombuffer(buf, dtype=DTYPE).astype(np.int64)
            xs[b] = arr[:-1]
            ys[b] = arr[1:]
        if self._pool is not None and B > 1:
            list(self._pool.map(one, range(B)))
        else:
            for b in range(B):
                one(b)
        return (xs, ys)

    def plan_sequential(self, *, world_size: int, batch_size: int) -> dict:
        if world_size < 1 or batch_size < 1:
            raise ValueError('world_size and batch_size must be >= 1')
        align = world_size * batch_size
        padded_total = self.total_windows // align * align
        return dict(world_size=int(world_size), batch_size=int(batch_size), total_windows_padded=int(padded_total), windows_per_rank_per_epoch=int(padded_total // world_size), batches_per_rank_per_epoch=int(padded_total // (world_size * batch_size)), dropped_windows_per_epoch=int(self.total_windows - padded_total))

    def iter_batches_sequential(self, *, rank: int, world_size: int, batch_size: int, base_seed: int=0, start_epoch: int=0, start_batch_in_epoch: int=0, device: str | torch.device='cpu', prefetch: int=_PREFETCH_DEFAULT) -> Iterator[tuple[torch.Tensor, torch.Tensor, int, int]]:
        ids = self._iter_window_ids(rank=rank, world_size=world_size, batch_size=batch_size, base_seed=base_seed, start_epoch=start_epoch, start_batch_in_epoch=start_batch_in_epoch)
        if prefetch and prefetch > 0:
            src = self._prefetched(ids, prefetch)
        else:
            src = ((self._read_window_ids(w), e, bi) for w, e, bi in ids)
        for (xs, ys), epoch, bi in src:
            x, y = self._to_torch(xs, ys, device)
            yield (x, y, epoch, bi)

    def _iter_window_ids(self, *, rank: int, world_size: int, batch_size: int, base_seed: int, start_epoch: int, start_batch_in_epoch: int) -> Iterator[tuple[np.ndarray, int, int]]:
        plan = self.plan_sequential(world_size=world_size, batch_size=batch_size)
        K = plan['windows_per_rank_per_epoch']
        BPE = plan['batches_per_rank_per_epoch']
        N_padded = plan['total_windows_padded']
        epoch = start_epoch
        batch_in_epoch = start_batch_in_epoch
        while True:
            perm = np.random.default_rng(base_seed + epoch).permutation(self.total_windows)
            perm = perm[:N_padded]
            my_slice = perm[rank * K:(rank + 1) * K]
            for bi in range(batch_in_epoch, BPE):
                yield (my_slice[bi * batch_size:(bi + 1) * batch_size], epoch, bi)
            epoch += 1
            batch_in_epoch = 0

    def _prefetched(self, ids: Iterator[tuple[np.ndarray, int, int]], depth: int):
        q: queue.Queue = queue.Queue(maxsize=depth)
        stop = threading.Event()
        _DONE = object()

        def produce() -> None:
            try:
                for window_ids, epoch, bi in ids:
                    if stop.is_set():
                        break
                    item = (self._read_window_ids(window_ids), epoch, bi)
                    while not stop.is_set():
                        try:
                            q.put(item, timeout=0.5)
                            break
                        except queue.Full:
                            continue
            except BaseException as exc:
                try:
                    q.put((exc, None, None), timeout=5.0)
                except queue.Full:
                    pass
                return
            try:
                q.put((_DONE, None, None), timeout=5.0)
            except queue.Full:
                pass
        t = threading.Thread(target=produce, daemon=True, name=f'fwprefetch-{self.split}')
        t.start()
        try:
            while True:
                first, epoch, bi = q.get()
                if first is _DONE:
                    return
                if isinstance(first, BaseException):
                    raise first
                yield (first, epoch, bi)
        finally:
            stop.set()
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            t.join(timeout=2.0)
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=os.path.expandvars('./data/fineweb-edu/llama3_tokens'))
    ap.add_argument('--seq-len', type=int, default=2048)
    ap.add_argument('--mode', choices=['random', 'sequential'], default='sequential')
    ap.add_argument('--world-size', type=int, default=2)
    ap.add_argument('--rank', type=int, default=0)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--steps', type=int, default=3)
    a = ap.parse_args()
    n_shards = len(json.load(open(os.path.join(a.dir, 'manifest.json')))['shards'])
    val_shards = 1 if n_shards >= 2 else 0
    ds = FineWebTokens(a.dir, seq_len=a.seq_len, split='train', val_shards=val_shards)
    if a.mode == 'random':
        x, y = ds.get_batch(a.batch_size, generator=torch.Generator().manual_seed(0))
        print('random x', x.shape, 'y', y.shape, 'x[0,:6]=', x[0, :6].tolist())
        assert (x[:, 1:] == y[:, :-1]).all()
        print('OK (random)')
    else:
        plan = ds.plan_sequential(world_size=a.world_size, batch_size=a.batch_size)
        print('plan:', plan)
        it = ds.iter_batches_sequential(rank=a.rank, world_size=a.world_size, batch_size=a.batch_size, base_seed=0)
        for s in range(a.steps):
            x, y, ep, bi = next(it)
            assert (x[:, 1:] == y[:, :-1]).all()
            print(f'step {s}: epoch={ep} batch_in_epoch={bi} x{tuple(x.shape)} x[0,:6]={x[0, :6].tolist()}')
        print('OK (sequential)')
