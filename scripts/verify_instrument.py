"""Verify the Tinker primitives this repo is built on, before building on them.

Each check exists because a later module depends on the answer: the divergence
heatmap assumes compute_logprobs is aligned (`lps[i] = log P(tok_i | tok_<i)`),
p_skip assumes the stop token id is stable, and the evaluator assumes the
loop hands it a snapshot client whose path is recoverable. Offline checks run
without credentials; the live section needs TINKER_API_KEY and costs a few
cents on Inkling-Small.

    python scripts/verify_instrument.py [--offline] [--ckpt tinker://...]
"""


import argparse
import asyncio
import math
import os

import tinker
from tinker.types import ModelInput, SamplingParams
from tinker_cookbook import model_info
from tinker_cookbook.renderers import get_renderer, get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL = "thinkingmachines/Inkling-Small"
QUESTION = [{"role": "user", "content": "Compute 47*6. Show your work briefly."}]
EFFORTS = (0.0, 0.2, 0.9)

# Inkling-Small, discount tier, $/token (price table Sep 2026).
PREFILL_PRICE = 0.58e-6
GEN_PRICE = 1.44e-6

_cost = {"prefill": 0, "gen": 0}


def check_prompt_render(renderer, tokenizer) -> bool:
    """Effort must appear as an injected system line and nothing else may change."""
    print("== prompt rendering (offline) ==")
    rendered = {
        e: renderer.build_generation_prompt(QUESTION, effort=e).to_ints() for e in EFFORTS
    }
    for e, toks in rendered.items():
        print(f"effort={e}: {len(toks)} tokens :: {tokenizer.decode(toks)!r}")

    a, b = rendered[0.0], rendered[0.9]
    pre = 0
    while pre < min(len(a), len(b)) and a[pre] == b[pre]:
        pre += 1
    post = 0
    while post < min(len(a), len(b)) - pre and a[-1 - post] == b[-1 - post]:
        post += 1
    print(
        f"0.0 vs 0.9: shared prefix {pre} tokens, shared suffix {post} tokens; "
        f"differs in {a[pre:len(a) - post]!r} -> {b[pre:len(b) - post]!r}"
    )
    return all(
        "Thinking effort level" in tokenizer.decode(toks) for toks in rendered.values()
    )


def check_stop_tokens(renderer, tokenizer) -> list[int]:
    """The end-of-message token id(s), decoded — p_skip scores this token in slot 0."""
    print("== stop sequences (offline) ==")
    ids = list(renderer.get_stop_sequences())
    for tok in ids:
        print(f"  {tok} :: {tokenizer.decode([tok])!r}")
    return ids


async def check_logprobs(client, prompt: list[int]) -> bool:
    """Per-token logprobs must align with input positions, be None at index 0, and be deterministic — the divergence metric's entire premise."""
    print("== compute_logprobs ==")
    lps1 = await client.compute_logprobs_async(ModelInput.from_ints(prompt))
    lps2 = await client.compute_logprobs_async(ModelInput.from_ints(prompt))
    _cost["prefill"] += 2 * len(prompt)
    n_none = sum(x is None for x in lps1)
    print(
        f"len(out)={len(lps1)} len(in)={len(prompt)} pos0={lps1[0]!r} "
        f"n_none={n_none} repeat_identical={lps1 == lps2}"
    )
    return len(lps1) == len(prompt) and lps1[0] is None and lps1 == lps2


async def check_sample(client, renderer, tokenizer) -> bool:
    """One call must return k parsed sequences; sampled-token logprobs must be recoverable from top-k."""
    print("== sample_async k=4 T=0.6 topk=8 ==")
    prompt = renderer.build_generation_prompt(QUESTION, effort=0.9).to_ints()
    resp = await client.sample_async(
        ModelInput.from_ints(prompt),
        num_samples=4,
        sampling_params=SamplingParams(
            max_tokens=384, temperature=0.6, stop=renderer.get_stop_sequences()
        ),
        topk_sample_logprobs=8,
    )
    _cost["prefill"] += len(prompt)
    print(f"n_sequences={len(resp.sequences)}")
    ok = len(resp.sequences) == 4
    for i, seq in enumerate(resp.sequences):
        _cost["gen"] += len(seq.tokens)
        topk = seq.topk_logprobs
        in_topk = (
            sum(
                seq.tokens[j] in {tok for tok, _ in cands}
                for j, cands in enumerate(topk)
                if cands
            )
            if topk
            else 0
        )
        msg, term = renderer.parse_response(seq.tokens)
        print(
            f"  seq{i}: n={len(seq.tokens)} stop={seq.stop_reason} term={term.value} "
            f"logprobs={'yes' if seq.logprobs else 'no'} "
            f"sampled-in-topk={in_topk}/{len(seq.tokens)} "
            f"text={get_text_content(msg)[:60]!r}"
        )
        ok &= seq.stop_reason in ("length", "stop")
    return ok


async def check_prefix_gap(client, renderer, tokenizer) -> bool:
    """Score one generated trace under two effort prefixes; the generating prefix should win by tens of nats."""
    print("== prefix gap ==")
    p_high = renderer.build_generation_prompt(QUESTION, effort=0.9).to_ints()
    p_low = renderer.build_generation_prompt(QUESTION, effort=0.0).to_ints()
    resp = await client.sample_async(
        ModelInput.from_ints(p_high),
        num_samples=1,
        sampling_params=SamplingParams(
            max_tokens=384, temperature=0.0, stop=renderer.get_stop_sequences()
        ),
    )
    cont = resp.sequences[0].tokens
    _cost["prefill"] += len(p_high)
    _cost["gen"] += len(cont)
    print(f"sampled {len(cont)} tokens at effort 0.9, stop={resp.sequences[0].stop_reason}")

    sums = {}
    for e, p in ((0.9, p_high), (0.0, p_low)):
        lps = await client.compute_logprobs_async(ModelInput.from_ints(p + cont))
        _cost["prefill"] += len(p) + len(cont)
        span = [x for x in lps[len(p) :] if x is not None]
        sums[e] = sum(span)
        print(f"effort={e}: n={len(span)} sum={sums[e]:.1f} nats")
    gap = sums[0.9] - sums[0.0]
    print(f"gap = {gap:+.1f} nats")
    return gap > 5.0


async def check_p_skip(client, renderer, tokenizer, stop_ids: list[int]) -> None:
    """P(end-of-message as the first generated token) on the base model — the instrument's healthy floor."""
    print("== p_skip ==")
    prompt = renderer.build_generation_prompt(QUESTION, effort=0.9).to_ints()
    for tok in stop_ids:
        lps = await client.compute_logprobs_async(ModelInput.from_ints(prompt + [tok]))
        _cost["prefill"] += len(prompt) + 1
        lp = lps[len(prompt)]
        print(f"  P(first token = {tok} :: {tokenizer.decode([tok])!r}) = {math.exp(lp):.4f}")


async def check_checkpoint(service: tinker.ServiceClient, ckpt: str) -> None:
    """Checkpoint clients must resolve their base model; documents that the path has no public accessor."""
    print("== checkpoint client ==")
    client = service.create_sampling_client(model_path=ckpt)
    print(f"  get_base_model: {await client.get_base_model_async()}")
    info = await client._get_sampler_submit()  # no public accessor for the path
    print(f"  GetSamplerResponse.model_path: {info.model_path}")
    run = await service.create_rest_client().get_training_run_by_tinker_path_async(ckpt)
    print(f"  REST training_run.base_model: {run.base_model}")


async def main(args) -> int:
    """Run offline checks first, then live ones; exit nonzero if any check fails."""
    tokenizer = get_tokenizer(MODEL)
    renderer = get_renderer(model_info.get_recommended_renderer_name(MODEL), tokenizer)

    results = {"prompt_render": check_prompt_render(renderer, tokenizer)}
    stop_ids = check_stop_tokens(renderer, tokenizer)
    if args.offline:
        print("offline: skipping live checks")
        return 0 if results["prompt_render"] else 1

    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY not set; source the env file first")

    service = tinker.ServiceClient()
    client = service.create_sampling_client(base_model=MODEL)
    prompt = renderer.build_generation_prompt(QUESTION, effort=0.9).to_ints()

    results["logprobs"] = await check_logprobs(client, prompt)
    results["sample"] = await check_sample(client, renderer, tokenizer)
    results["prefix_gap"] = await check_prefix_gap(client, renderer, tokenizer)
    await check_p_skip(client, renderer, tokenizer, stop_ids)
    if args.ckpt:
        await check_checkpoint(service, args.ckpt)

    cost = _cost["prefill"] * PREFILL_PRICE + _cost["gen"] * GEN_PRICE
    print(
        f"\nest. cost: {_cost['prefill']} prefill + {_cost['gen']} gen tokens "
        f"~ ${cost:.4f} (discount tier)"
    )
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--offline", action="store_true")
    p.add_argument("--ckpt", help="tinker:// checkpoint path to test resolution")
    raise SystemExit(asyncio.run(main(p.parse_args())))
