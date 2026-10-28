
from __future__ import annotations
import argparse
import math
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from experiments.lm.data.fineweb_dataset import FineWebTokens
from experiments.lm.model.lm import MambaLM
from experiments.lm.model.sizes import RECIPE, SIZES
from experiments.lm.train import ckpt as ckpt_mod
from experiments.lm.train.optim import LRSchedule, make_optimizer

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--size', choices=list(SIZES.keys()), required=True)
    p.add_argument('--ssm', default='official_mamba3', help='mixer for every block: one of the five registered LM-contract mixers (official_mamba1, official_mamba2, official_mamba3, official_mamba3_mimo, llama3_attention), or a JSON dict naming a HybridSSM family and its kwargs (families: ssd_mimo, hybrid_mamba).')
    p.add_argument('--d-state', type=int, default=0, help='override SSM state size N (0 = size default 128)')
    p.add_argument('--mlp-dim', type=int, default=0, help="override the size recipe's MLP inner dim (0 = size default). NOT the preferred parameter-matching knob: trimming the MLP makes MLP capacity co-vary INVERSELY with mixer cost, so a cheap SSM keeps a big MLP and an expensive one pays for itself -- a confound aimed straight at the variable under study. Use --n-layers instead. Each unit is ~3*d_model*n_layers params (gated MLP).")
    p.add_argument('--min-lr', type=float, default=0.0, help='LR floor the cosine decays to; 0 => the recipe default (1e-5). Sweeping PEAK alone silently varies the floor RATIO (1e-5 is 0.67%% of 1.5e-3 but 0.22%% of 4.5e-3), which confounds the two axes.')
    p.add_argument('--peak-lr', type=float, default=0.0, help="override the size recipe's peak LR (0 = recipe default, 5*gpt3_base_lr = 1.5e-3 at 440M). The harness trains EVERY architecture at one fixed LR, so a single-LR comparison measures 'who likes this LR' as much as 'who is better' -- and 1.5e-3 is aggressive here (conv-less arms hit grad-norm 1e5 and diverged). Use to test whether the mamba3 gap is partly an LR-optimum mismatch rather than architecture.")
    p.add_argument('--n-layers', type=int, default=0, help="override the size recipe's depth (0 = size default). THE parameter-matching knob: the 440M harness matches backbone SHAPE, not params, so M1-family mixers silently carry +2.3%% (hm_m1) to +5.9%% (mhr_r1_complex) over the mamba3 target (439.38M). Dropping layers keeps the MLP:mixer ratio identical across arms and follows the Mamba convention (a Mamba block is ~6d^2 vs a Transformer layer's ~12d^2, so Mamba-1/2 use 2x the layers to hit GPT-3 param targets). Matched depths at 440M/N64: hm_m1 23 (436.19M, -0.72%%), hm_m1_complex 23 (439.21M, -0.04%%), mhr_r1_complex 22 (437.40M, -0.45%%). Rule: at most baseline params, within 1%% below.")
    p.add_argument('--decay-from-tokens', type=int, default=0, help='Decay branch: start of the cosine decay (tokens), normally the stable-phase milestone a long run continues from (e.g. 2e9). Passed on a resume it starts the branch; recorded in the ckpt.')
    p.add_argument('--decay-to-tokens', type=int, default=0, help="Decay branch: token count at which the LR reaches min_lr (the run's declared final budget, e.g. 10e9).")
    p.add_argument('--lr-constant', action='store_true', help='Hold peak LR after warmup and never decay (the STABLE phase of warmup-stable-decay). Trunk runs share one trajectory at any length; anneal later from a ckpt. Passed on a --resume it OVERRIDES the schedule stored in the checkpoint: a run started on a cosine continues constant-after-warmup from that point (and says so in the log).')
    p.add_argument('--lr-horizon-tokens', type=int, default=0, help='anchor the cosine to THIS many tokens (e.g. 100e9 = paper protocol) instead of the run length. With this set, a short --stop-tokens chunk is a faithful prefix of the full-horizon schedule, so resumes continue the SAME cosine with no LR restart. 0 = adaptive (anneal over the run length).')
    p.add_argument('--stop-tokens', type=int, default=0, help='halt THIS chunk at this many tokens_seen (resumable milestone). 0 = use --max-tokens / --max-epochs.')
    p.add_argument('--tokens-dir', default=os.path.expandvars('./data/fineweb-edu/llama3_tokens'))
    p.add_argument('--run-dir', required=True, help='where to write logs + checkpoints')
    p.add_argument('--seq-len', type=int, default=RECIPE['seq_len'])
    p.add_argument('--batch-tokens', type=int, default=RECIPE['batch_tokens'])
    p.add_argument('--micro-batch', type=int, default=8, help='per-GPU micro batch (sequences)')
    p.add_argument('--max-epochs', type=int, default=1, help='stop at the end of this many full epochs (default 1 = paper-faithful)')
    p.add_argument('--max-tokens', type=int, default=10 ** 18, help='hard token-budget cap (default: effectively unbounded; use max_epochs)')
    p.add_argument('--warmup-tokens', type=int, default=500000000, help='linear warmup length in tokens (default 0.5B)')
    p.add_argument('--val-shards', type=int, default=2)
    p.add_argument('--val-every-tokens', type=int, default=500000000)
    p.add_argument('--val-batches', type=int, default=50)
    p.add_argument('--ckpt-every-tokens', type=int, default=2000000000)
    p.add_argument('--log-every-steps', type=int, default=10)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--seed', type=int, default=0, help='permutation seed: epoch e uses seed+e (deterministic across ranks)')
    p.add_argument('--smoke-steps', type=int, default=0, help='if >0, run only this many macro-steps then exit')
    p.add_argument('--data-prefetch', type=int, default=4, help='batches the loader reads AHEAD of the consumer, in a background thread (0 disables). Order and contents are unchanged -- one producer walking the same index schedule into a FIFO -- so runs stay bit-comparable with every past run; only wall-clock moves.')
    p.add_argument('--data-read-threads', type=int, default=8, help='threads issuing os.pread within one batch. The windows of a batch are unrelated random offsets, so they overlap freely; os.pread drops the GIL where np.memmap slicing does not.')
    p.add_argument('--probe-timing', action='store_true', help='decompose each macro-step into data / compute / grad-sync / straggler-skew and print min/median/max across ranks. Adds one barrier per step, so do NOT use it for runs whose tok/s is being reported.')
    p.add_argument('--compile-step', choices=['off', 'default'], default='off', help="whole-step torch.compile of the LM forward+loss (mode='default'; max-autotune and CUDA graphs are closed negatives, do not revisit)")
    p.add_argument('--ckpt-every-blocks', type=int, default=0, help="activation-checkpoint every k-th block (4 -> '6/24' at 24 layers). 0 = off")
    p.add_argument('--ce-chunk', type=int, default=0, help='chunked+checkpointed CE row-chunk size (pin: 4096). 0 = off (full logits materialized)')
    p.add_argument('--precision', choices=['fp32', 'bf16'], default='bf16', help='bf16 = autocast forward in bfloat16 (paper recipe; needs SSM that supports bf16). fp32 = no autocast (required for the M1 kernel which is fp32-only).')
    return p.parse_args()

def setup_dist() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get('SLURM_PROCID', os.environ.get('RANK', 0)))
    world = int(os.environ.get('SLURM_NTASKS', os.environ.get('WORLD_SIZE', 1)))
    local = int(os.environ.get('SLURM_LOCALID', os.environ.get('LOCAL_RANK', 0)))
    if world > 1:
        os.environ.setdefault('MASTER_ADDR', os.environ.get('SLURM_LAUNCH_NODE_IPADDR', '127.0.0.1'))
        os.environ.setdefault('MASTER_PORT', '29500')
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world)
        os.environ['LOCAL_RANK'] = str(local)
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
    device = torch.device(f'cuda:{local}') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return (rank, world, local, device)

def is_rank0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0

def log(msg: str) -> None:
    if is_rank0():
        print(msg, flush=True)

@torch.no_grad()
def evaluate(model: torch.nn.Module, val: FineWebTokens, device: torch.device, micro_batch: int, n_batches: int, autocast_enabled: bool) -> float:
    model.eval()
    losses = []
    g = torch.Generator().manual_seed(0)
    for _ in range(n_batches):
        x, y = val.get_batch(micro_batch, device=device, generator=g)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=autocast_enabled):
            out = model(x, targets=y)
        losses.append(out['loss'].detach().float())
    model.train()
    loss = torch.stack(losses).mean()
    if dist.is_initialized():
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)
    return loss.item()

def main():
    args = parse_args()
    rank, world, local, device = setup_dist()
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    size = SIZES[args.size]
    import json as _json
    from experiments.lm.model.lm import NAMED_SSM as _NAMED_SSM
    if args.ssm in _NAMED_SSM:
        ssm_spec = _NAMED_SSM[args.ssm]
    elif args.ssm.lstrip().startswith('{'):
        ssm_spec = _json.loads(args.ssm)
    else:
        ssm_spec = args.ssm
    cfg = size.model_cfg(ssm=ssm_spec)
    if args.d_state > 0:
        from dataclasses import replace as _replace
        cfg = _replace(cfg, d_state=args.d_state)
    if args.mlp_dim > 0:
        from dataclasses import replace as _replace
        cfg = _replace(cfg, mlp_dim=args.mlp_dim)
    if args.n_layers > 0:
        from dataclasses import replace as _replace
        cfg = _replace(cfg, n_layers=args.n_layers)
    autocast_enabled = device.type == 'cuda' and args.precision == 'bf16'
    if is_rank0():
        log(f'[cfg] device={device} precision={args.precision} autocast_enabled={autocast_enabled}')
    tokens_per_microstep = world * args.micro_batch * args.seq_len
    grad_accum = max(1, args.batch_tokens // tokens_per_microstep)
    effective_batch_tokens = grad_accum * tokens_per_microstep
    if is_rank0():
        log(f"[cfg] preset={args.size} (width preset; depth/mlp below are the arm's) d_model={cfg.d_model} n_layers={cfg.n_layers} mlp={cfg.mlp_dim} peak_lr={size.peak_lr:.2e}")
        log(f'[cfg] world={world} micro_batch={args.micro_batch} grad_accum={grad_accum} seq_len={args.seq_len} batch_tokens={effective_batch_tokens} (target {args.batch_tokens})')
    train_ds = FineWebTokens(args.tokens_dir, seq_len=args.seq_len, split='train', val_shards=args.val_shards, read_threads=args.data_read_threads)
    val_ds = FineWebTokens(args.tokens_dir, seq_len=args.seq_len, split='val', val_shards=args.val_shards, read_threads=args.data_read_threads)
    plan = train_ds.plan_sequential(world_size=world, batch_size=args.micro_batch)
    micro_per_epoch_per_rank = plan['batches_per_rank_per_epoch']
    macro_per_epoch = micro_per_epoch_per_rank // grad_accum
    if macro_per_epoch < 1:
        raise RuntimeError(f'epoch too small for current shapes: micro_per_epoch_per_rank={micro_per_epoch_per_rank}, grad_accum={grad_accum}')
    total_macro_steps_target = args.max_epochs * macro_per_epoch
    if is_rank0():
        log(f'[data] train_tokens={train_ds.total_tokens:,}  val_tokens={val_ds.total_tokens:,}')
        log(f"[data] {plan['total_windows_padded']:,} windows/epoch (padded); {plan['dropped_windows_per_epoch']} windows/epoch dropped ({plan['dropped_windows_per_epoch'] * args.seq_len / train_ds.total_tokens * 100:.6f}% of dataset)")
        log(f'[data] {macro_per_epoch:,} macro-steps/epoch -> max {total_macro_steps_target:,} macro-steps total')
    model = MambaLM(cfg).to(device)
    _raw = model
    _raw.ckpt_every_blocks = args.ckpt_every_blocks
    _raw.ce_chunk = args.ce_chunk
    if args.compile_step == 'default':
        model = torch.compile(model)
    if world > 1:
        model = DDP(model, device_ids=[local], gradient_as_bucket_view=True)
    base = _raw
    n_params = base.num_params()
    n_nonemb = base.num_params(non_embedding=True) if hasattr(base, 'num_params') else n_params
    log(f'[model] params={n_params / 1000000.0:.1f}M total, {n_nonemb / 1000000.0:.1f}M non-embedding (tied embedding {(n_params - n_nonemb) / 1000000.0:.1f}M)')
    _peak_lr = float(args.peak_lr) if args.peak_lr > 0 else size.peak_lr
    _min_lr = float(args.min_lr) if args.min_lr > 0 else RECIPE['lr_min']
    optimizer = make_optimizer(base, lr=_peak_lr, weight_decay=RECIPE['weight_decay'], betas=RECIPE['betas'])
    if args.lr_horizon_tokens > 0:
        lr_horizon = args.lr_horizon_tokens
    else:
        lr_horizon = min(total_macro_steps_target * effective_batch_tokens, args.max_tokens)
    sched = LRSchedule(peak_lr=_peak_lr, total_tokens=lr_horizon, warmup_tokens=args.warmup_tokens, min_lr=_min_lr, constant=args.lr_constant, decay_from=args.decay_from_tokens, decay_to=args.decay_to_tokens)
    stop_tokens = args.stop_tokens if args.stop_tokens > 0 else min(total_macro_steps_target * effective_batch_tokens, args.max_tokens)
    sched_meta = dict(lr_horizon=lr_horizon, warmup_tokens=args.warmup_tokens, peak_lr=_peak_lr, min_lr=_min_lr, constant=bool(args.lr_constant), decay_from=int(args.decay_from_tokens), decay_to=int(args.decay_to_tokens))
    if is_rank0():
        if args.decay_from_tokens > 0:
            log(f'[lr] DECAY BRANCH: constant peak={_peak_lr:.2e} until {args.decay_from_tokens:,} tokens, then cosine to min={_min_lr:.1e} at {args.decay_to_tokens:,} tokens  | stop@{stop_tokens:,} tokens')
        elif args.lr_constant:
            log(f'[lr] CONSTANT peak (warmup-stable, no decay)  warmup={args.warmup_tokens:,}  peak={_peak_lr:.2e}  | stop@{stop_tokens:,} tokens')
        else:
            log(f'[lr] cosine horizon={lr_horizon:,} tokens  warmup={args.warmup_tokens:,}  peak={_peak_lr:.2e} -> min={_min_lr:.1e} ({100 * _min_lr / _peak_lr:.2f}%% of peak)  | stop@{stop_tokens:,} tokens')
    macro_step = 0
    nonfinite_steps = 0
    NONFINITE_TOLERANCE = 2
    tokens_seen = 0
    if args.resume:
        latest = ckpt_mod.find_latest(args.run_dir)
        if latest is not None:
            info = ckpt_mod.load(latest, model=model, optimizer=optimizer, map_location=device)
            macro_step = info['extra'].get('macro_step', info['step'])
            tokens_seen = info['tokens_seen']
            saved = info['extra'].get('sched_meta')
            if saved and args.lr_horizon_tokens == 0:
                was_constant = bool(saved.get('constant', False))
                constant = was_constant or bool(args.lr_constant)
                d_from = args.decay_from_tokens or int(saved.get('decay_from', 0))
                d_to = args.decay_to_tokens or int(saved.get('decay_to', 0))
                sched = LRSchedule(peak_lr=saved['peak_lr'], total_tokens=saved['lr_horizon'], warmup_tokens=saved['warmup_tokens'], min_lr=saved['min_lr'], constant=constant, decay_from=d_from, decay_to=d_to)
                sched_meta = dict(saved, constant=constant, decay_from=d_from, decay_to=d_to)
                if args.decay_from_tokens > 0:
                    log(f'[resume] DECAY BRANCH declared: cosine from {d_from:,} to min at {d_to:,} tokens (tokens_seen={tokens_seen:,}); recorded in sched_meta')
                elif d_from > 0:
                    log(f'[resume] restored decay branch {d_from:,} -> {d_to:,} tokens')
                if constant and (not was_constant):
                    log(f"[resume] --lr-constant OVERRIDE: the saved cosine (horizon={saved['lr_horizon']:,}) is replaced by a CONSTANT peak={saved['peak_lr']:.2e} after warmup from tokens_seen={tokens_seen:,} onwards")
                elif constant:
                    log(f"[resume] restored CONSTANT peak={saved['peak_lr']:.2e} after warmup={saved['warmup_tokens']:,}")
                else:
                    log(f"[resume] restored cosine horizon={saved['lr_horizon']:,} (LR continues from tokens_seen, no restart)")
            log(f'[resume] loaded {latest.name} @ macro_step={macro_step} tokens={tokens_seen:,} lr@resume={sched.lr_at(tokens_seen):.3e}')
    epoch = macro_step // macro_per_epoch
    macro_in_epoch = macro_step % macro_per_epoch
    start_micro_in_epoch = macro_in_epoch * grad_accum
    train_iter = train_ds.iter_batches_sequential(rank=rank, world_size=world, batch_size=args.micro_batch, base_seed=args.seed, start_epoch=epoch, start_batch_in_epoch=start_micro_in_epoch, device=device, prefetch=args.data_prefetch)
    if is_rank0():
        log(f'[data] loader prefetch={args.data_prefetch} read_threads={args.data_read_threads}')
    Path(args.run_dir).mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.run_dir) / 'metrics.jsonl'
    if is_rank0() and (not args.resume):
        metrics_path.write_text('')
    last_val_at = tokens_seen
    last_ckpt_at = tokens_seen
    last_log_t = time.time()
    last_log_tokens = tokens_seen
    while tokens_seen < stop_tokens and tokens_seen < args.max_tokens:
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        cur_epoch = None
        _t = _PhaseTimer(args.probe_timing)
        for micro in range(grad_accum):
            _t.mark()
            x, y, cur_epoch, _bi = next(train_iter)
            _t.add('data')
            is_last = isinstance(model, DDP) and micro == grad_accum - 1
            if is_last:
                _t.barrier_skew(world)
            no_sync_ctx = _null_ctx() if is_last else model.no_sync() if isinstance(model, DDP) else _null_ctx()
            with no_sync_ctx:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=autocast_enabled):
                    out = model(x, targets=y)
                loss = out['loss'] / grad_accum
                loss.backward()
            loss_accum += loss.detach().float().item()
            _t.add('bwd_sync' if is_last else 'bwd')
        if not isinstance(model, DDP):
            _t.barrier_skew(world)
        grad_norm = torch.nn.utils.clip_grad_norm_(base.parameters(), RECIPE['grad_clip']).item()
        if not math.isfinite(grad_norm):
            nonfinite_steps += 1
            optimizer.zero_grad(set_to_none=True)
            print(f'[nonfinite] grad_norm={grad_norm} at macro_step={macro_step} loss={loss_accum:.5f} tokens={tokens_seen:,} ({nonfinite_steps}/{NONFINITE_TOLERANCE}) — step SKIPPED', flush=True)
            if nonfinite_steps >= NONFINITE_TOLERANCE:
                raise RuntimeError(f'aborting: {nonfinite_steps} non-finite grad_norms (last at macro_step={macro_step}, tokens={tokens_seen:,})')
            macro_step += 1
            tokens_seen += effective_batch_tokens
            continue
        lr_now = sched.apply(optimizer, tokens_seen)
        optimizer.step()
        for _m in base.modules():
            if hasattr(_m, 'project_'):
                _m.project_()
        _t.add('optim')
        macro_step += 1
        tokens_seen += effective_batch_tokens
        _t.report(macro_step, args.log_every_steps, world, rank)
        if macro_step % args.log_every_steps == 0 and is_rank0():
            dt = time.time() - last_log_t
            tps = (tokens_seen - last_log_tokens) / max(dt, 1e-06)
            _peak_gib = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0
            log(f'[step {macro_step:>6}] ep={cur_epoch} tok={tokens_seen:>13,} loss={loss_accum:.4f} lr={lr_now:.2e} gn={grad_norm:.2f} tok/s={tps:,.0f} peak_mem={_peak_gib:.1f}GiB')
            with open(metrics_path, 'a') as f:
                f.write(f'{{"step":{macro_step},"epoch":{cur_epoch},"tokens":{tokens_seen},"loss":{loss_accum:.5f},"lr":{lr_now:.6e},"grad_norm":{grad_norm:.4f},"tok_s":{tps:.0f}}}\n')
            last_log_t = time.time()
            last_log_tokens = tokens_seen
        if tokens_seen - last_val_at >= args.val_every_tokens:
            val_loss = evaluate(model, val_ds, device, args.micro_batch, args.val_batches, autocast_enabled)
            if is_rank0():
                log(f'[val   {macro_step:>6}] tok={tokens_seen:>13,} val_loss={val_loss:.4f} val_ppl={math.exp(val_loss):.2f}')
                with open(metrics_path, 'a') as f:
                    f.write(f'{{"step":{macro_step},"tokens":{tokens_seen},"val_loss":{val_loss:.5f}}}\n')
            last_val_at = tokens_seen
        if tokens_seen - last_ckpt_at >= args.ckpt_every_tokens:
            saved = ckpt_mod.save(args.run_dir, step=macro_step, tokens_seen=tokens_seen, model=model, optimizer=optimizer, model_cfg=cfg, extra={'macro_step': macro_step, 'epoch': macro_step // macro_per_epoch, 'macro_per_epoch': macro_per_epoch, 'base_seed': args.seed, 'sched_meta': sched_meta})
            if saved is not None:
                log(f'[ckpt ] {saved.name}')
            last_ckpt_at = tokens_seen
        if args.smoke_steps and macro_step >= args.smoke_steps:
            break
    final = ckpt_mod.save(args.run_dir, step=macro_step, tokens_seen=tokens_seen, model=model, optimizer=optimizer, model_cfg=cfg, extra={'macro_step': macro_step, 'epoch': macro_step // macro_per_epoch, 'macro_per_epoch': macro_per_epoch, 'base_seed': args.seed, 'sched_meta': sched_meta, 'final': True})
    if final is not None:
        log(f'[done ] final ckpt {final.name} @ macro_step={macro_step} tokens={tokens_seen:,}')
    if dist.is_initialized():
        dist.destroy_process_group()

class _null_ctx:

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
_PHASES = ('data', 'bwd', 'bwd_sync', 'skew', 'optim')

class _PhaseTimer:
    __slots__ = ('on', '_t0', 'acc')

    def __init__(self, on: bool) -> None:
        self.on = bool(on)
        self._t0 = time.perf_counter() if self.on else 0.0
        self.acc = dict.fromkeys(_PHASES, 0.0) if self.on else None

    def mark(self) -> None:
        if self.on:
            self._t0 = time.perf_counter()

    def add(self, phase: str) -> None:
        if self.on:
            now = time.perf_counter()
            self.acc[phase] += now - self._t0
            self._t0 = now

    def barrier_skew(self, world: int) -> None:
        if self.on and world > 1 and dist.is_initialized():
            self.mark()
            dist.barrier()
            self.add('skew')

    def report(self, step: int, every: int, world: int, rank: int) -> None:
        if not self.on or step % every:
            return
        mine = torch.tensor([self.acc[p] for p in _PHASES], dtype=torch.float64, device='cuda')
        if world > 1 and dist.is_initialized():
            buf = [torch.zeros_like(mine) for _ in range(world)]
            dist.all_gather(buf, mine)
        else:
            buf = [mine]
        if rank != 0:
            return
        allr = torch.stack(buf)
        tot = allr.sum(1)
        parts = ' '.join((f'{p}={allr[:, i].min():.3f}/{allr[:, i].median():.3f}/{allr[:, i].max():.3f}' for i, p in enumerate(_PHASES)))
        slow = int(torch.argmax(tot))
        print(f'[probe {step:>5}] world={world} step={tot.max():.3f}s (min/med/max per phase) {parts} slowest_rank={slow}', flush=True)
if __name__ == '__main__':
    main()
