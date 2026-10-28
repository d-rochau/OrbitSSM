"""A small mixed-family language model end to end, on CPU (not a performance configuration).

1. build a `CausalLM` whose layers are two families (SSD-MIMO, HybridMamba) with a SwiGLU MLP;
2. one training step (next-token cross entropy, AdamW);
3. stateful streaming: the sequence fed in two pieces with ``state=...`` gives the full-sequence logits;
4. token-by-token decode with ``step`` gives the same logits;
5. save the recipe (``to_recipe()``, JSON) and the weights (``state_dict``), rebuild with
   ``CausalLM.from_recipe`` and reload: the rebuilt model computes the same logits bit for bit;
6. ``generate`` a few greedy tokens from the reloaded model.

Run: ``python examples/lm_roundtrip.py`` (``PYTHONPATH=src`` from a checkout).
"""
import json
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from orbit_ssm import CausalLM, HybridMambaConfig, SSDMimoConfig

VOCAB, D_MODEL, L = 64, 32, 24


def build():
    layers = [
        SSDMimoConfig(d_state=16, expand=1, d_ssm_in=1, d_ssm_out=1, n_groups=16),
        {"type": "hybrid_mamba", "d_state": 16},             # a dict entry names its family
    ]
    return CausalLM(vocab_size=VOCAB, layers=layers, d_model=D_MODEL, d_mlp=2 * D_MODEL, execution={"backend": "torch"})


def rel_err(x, ref):
    return ((x - ref).abs().max() / ref.abs().max()).item()


def main():
    torch.manual_seed(7)
    lm = build()
    ids = torch.randint(0, VOCAB, (2, L))

    # -- one training step
    opt = torch.optim.AdamW(lm.parameters(), lr=1e-3)
    logits = lm(ids[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, VOCAB), ids[:, 1:].reshape(-1))
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    assert torch.isfinite(loss)

    # The chunked scan, the two-piece stream and the token recurrence sum the same terms in different
    # orders: per layer at most ~L fp32 roundings on a length-L sum, times a small constant for the
    # norm / gate / MLP epilogue.  The bound scales with that, not a fixed number.
    tol = 8 * L * len(lm.ssm_layers) * torch.finfo(torch.float32).eps
    with torch.no_grad():
        full = lm(ids)

        # -- stateful streaming: two pieces, the state carried between them
        head, state = lm(ids[:, :10], return_state=True)
        tail, state = lm(ids[:, 10:], state=state, return_state=True)
        assert rel_err(torch.cat([head, tail], 1), full) <= tol

        # -- token-by-token decode from the documented initial state
        states = lm.init_state(ids.shape[0])
        steps = []
        for t in range(L):
            step_logits, states = lm.step(ids[:, t], states)
            steps.append(step_logits)
        assert rel_err(torch.stack(steps, 1), full) <= tol

    # -- recipe + weights round trip
    with tempfile.TemporaryDirectory() as tmp:
        recipe, weights = Path(tmp) / "recipe.json", Path(tmp) / "weights.pt"
        recipe.write_text(json.dumps(lm.to_recipe(), indent=1))
        torch.save(lm.state_dict(), weights)
        lm2 = CausalLM.from_recipe(json.loads(recipe.read_text()))
        lm2.load_state_dict(torch.load(weights, weights_only=True))
    with torch.no_grad():
        assert torch.equal(lm2(ids), full), "the reloaded model must compute the same logits"
        out = lm2.generate(ids[:, :4], max_new_tokens=6, temperature=0)
    assert out.shape == (2, 10)
    print(f"LM round trip passed: loss {loss.item():.3f}, stream/step within {tol:.1e}, "
          f"reload bit-identical, generated {tuple(out.shape)}")


if __name__ == "__main__":
    main()
