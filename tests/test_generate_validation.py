"""Pins: `CausalLM.generate` refuses malformed arguments by name (TypeError for a wrong type, ValueError for a
value out of range), prefills the prompt with `forward(..., return_state=True)`, and greedy decoding equals
the token-by-token `step` reference.
"""

import pytest
import torch
from orbit_ssm import CausalLM
from _tolerance import relative, route_bound
from test_models import _model_kwargs


pytestmark = pytest.mark.paper1


_CPU_FAMILIES = ("ssd_mimo", "hybrid_mamba")


def _lm(family):
    torch.manual_seed(0)
    kw = _model_kwargs(family)
    kw["execution"] = {**(kw.get("execution") or {}), "backend": "torch"}      # the CPU route of every family
    return CausalLM(**kw).eval()


@pytest.fixture(scope="module")
def lm():
    return _lm("hybrid_mamba")


@pytest.mark.parametrize("family", _CPU_FAMILIES)
def test_greedy_generate_equals_the_step_reference(family):
    """contract: prefilling with one `forward(return_state=True)` pass then stepping gives the same greedy
    tokens as feeding the prompt through `step` one token at a time (the decode oracle); the prompt is kept
    and `max_new_tokens=0` returns it."""
    model = _lm(family)
    prompt = torch.randint(0, model.vocab_size, (2, 5), generator=torch.Generator().manual_seed(1))
    got = model.generate(prompt, max_new_tokens=6, temperature=0)
    with torch.no_grad():
        states = model.init_state(2)
        for t in range(prompt.shape[1]):
            logits, states = model.step(prompt[:, t:t + 1], states)
        want = [prompt]
        for _ in range(6):
            tok = logits.argmax(-1, keepdim=True)
            want.append(tok)
            logits, states = model.step(tok, states)
    assert torch.equal(got, torch.cat(want, 1))
    assert torch.equal(model.generate(prompt, 0), prompt)


def test_sampled_generate_stays_in_the_vocabulary_and_honours_top_k(lm):
    """contract: sampled tokens lie in [0, vocab_size); top_k=1 is greedy."""
    prompt = torch.randint(0, lm.vocab_size, (3, 4), generator=torch.Generator().manual_seed(2))
    out = lm.generate(prompt, 5, temperature=0.7, top_k=5)
    assert out.shape == (3, 9) and int(out.max()) < lm.vocab_size and int(out.min()) >= 0
    assert torch.equal(lm.generate(prompt, 4, temperature=1.3, top_k=1), lm.generate(prompt, 4, temperature=0))
    assert lm.generate(prompt, 2, top_k=10 ** 6).shape == (3, 6)       # top_k > vocab is the full vocabulary


@pytest.mark.parametrize("bad,exc,match", [
    ([[1, 2]], TypeError, "input_ids must be a torch.Tensor"),
    (torch.zeros(2, 3, dtype=torch.int32), TypeError, "torch.long"),
    (torch.zeros(2, 3), TypeError, "torch.long"),
    (torch.zeros(3, dtype=torch.long), ValueError, "2-D"),
    (torch.zeros(1, 2, 3, dtype=torch.long), ValueError, "2-D"),
    (torch.zeros(2, 0, dtype=torch.long), ValueError, "at least one prompt token"),
    (torch.zeros(0, 3, dtype=torch.long), ValueError, "at least one prompt token"),
    (torch.tensor([[0, 41]]), ValueError, r"in \[0, 41\)"),
    (torch.tensor([[-1, 3]]), ValueError, r"in \[0, 41\)"),
])
def test_generate_refuses_bad_input_ids(lm, bad, exc, match):
    """contract: input ids must be a 2-D non-empty long tensor with values in [0, vocab_size)."""
    with pytest.raises(exc, match=match):
        lm.generate(bad, 1)


@pytest.mark.parametrize("kw,exc,match", [
    (dict(max_new_tokens=-1), ValueError, "max_new_tokens must be >= 0"),
    (dict(max_new_tokens=1.5), TypeError, "max_new_tokens must be an int"),
    (dict(max_new_tokens=True), TypeError, "max_new_tokens must be an int"),
    (dict(max_new_tokens=None), TypeError, "max_new_tokens must be an int"),
    (dict(temperature=-0.1), ValueError, "temperature must be finite and >= 0"),
    (dict(temperature=float("nan")), ValueError, "temperature must be finite"),
    (dict(temperature=float("inf")), ValueError, "temperature must be finite"),
    (dict(temperature="1"), TypeError, "temperature must be a number"),
    (dict(top_k=0), ValueError, "top_k must be >= 1"),
    (dict(top_k=-3), ValueError, "top_k must be >= 1"),
    (dict(top_k=2.0), TypeError, "top_k must be an int"),
    (dict(top_k=True), TypeError, "top_k must be an int"),
])
def test_generate_refuses_bad_options(lm, kw, exc, match):
    """contract: max_new_tokens is an int >= 0, temperature a finite number >= 0, top_k None or an int >= 1."""
    args = dict(max_new_tokens=1, temperature=1.0, top_k=None)
    args.update(kw)
    with pytest.raises(exc, match=match):
        lm.generate(torch.zeros(1, 2, dtype=torch.long), **args)


def test_prefill_with_return_state_continues_by_step():
    """contract: `forward(ids, return_state=True)` returns (logits, per-layer states) and stepping from those
    states equals running the longer sequence in one forward (the prefill path `generate` uses); fp64, two
    routes of different arithmetic (A7: u * sqrt(depth) * sqrt(arms), depth = width * layers * length)."""
    lm = _lm("hybrid_mamba").double()
    ids = torch.randint(0, lm.vocab_size, (2, 6), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        logits, states = lm(ids[:, :5], return_state=True)
        assert logits.shape == (2, 5, lm.vocab_size) and len(states) == len(lm.ssm_layers)
        nxt, _ = lm.step(ids[:, 5:6], states)
        assert relative(nxt, lm(ids)[:, -1]) < route_bound("fp64", depth=16 * 2 * 6)
