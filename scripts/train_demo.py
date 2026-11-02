"""Demo SFT run with RegressionEvaluator in the training loop.

`data` is offline. `data --distill` samples the model itself for targets:
at effort 0.9 the rendered assistant turn carries a thinking segment, so the
distilled file is the control for the human-written (no-thinking) targets.
`train` prints a token/cost forecast and requires --yes.
Training data renders at the renderer default effort 0.9 evaluate at 0.9.
Uses 1024 subset of gsm8k into an SFT (LoRA rank=32) training loop of inkling-small.
32 minibatch standard SGD
"""

import argparse
import asyncio
import json
import random
import re
from pathlib import Path

import chz
from tinker_cookbook import checkpoint_utils, cli_utils, model_info
from tinker_cookbook.eval.benchmarks._common import load_benchmark_dataset
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

from flipbook.evaluator import RegressionEvaluator
from flipbook.pricing import estimate_usd
from flipbook.store import Store

MODEL = "thinkingmachines/Inkling-Small"
SYSTEM = "Put your final answer in \\boxed{}."

# convert gsm8k into training conversations
def build_data(out: Path, n: int, seed: int) -> None:
    ds = load_benchmark_dataset("openai/gsm8k", name="main", split="train")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    rows = []
    for i in idx[:n]:
        r = ds[i]
        body, _, final = r["answer"].rpartition("####")
        final = final.strip().replace(",", "")
        # gsm8k solutions carry calculator annotations like <<3*4=12>> drop them
        body = re.sub(r"<<[^>]*>>", "", body).strip()
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": r["question"]},
                    {"role": "assistant", "content": f"{body}\n\nFinal answer: \\boxed{{{final}}}"},
                ]
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {len(rows)} conversations → {out}")


async def build_distilled_data(
    out: Path, n: int, seed: int, model: str, max_tokens: int,
    pool: int, yes: bool, client=None,
) -> int:
    """Same prompts as build_data, but the target is the model's own sampled
    response: parse_response puts the thinking segment into the message, and
    FromConversationFileBuilder renders it back bit-exactly."""
    import tinker
    from tinker.types import ModelInput, SamplingParams
    from tinker_cookbook.renderers import get_renderer, get_text_content
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    from flipbook.graders import grade
    from flipbook.runner import _sampling_client

    cost = estimate_usd(model, "sample", pool * max_tokens)
    print(f"forecast <= ${cost:.2f} rollouts "
          f"({pool} prompts x <= {max_tokens} gen tok, early-stop at {n} kept)")
    if not yes:
        print("dry run: pass --yes to sample")
        return 3

    ds = load_benchmark_dataset("openai/gsm8k", name="main", split="train")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    renderer_name = model_info.get_recommended_renderer_name(model)
    renderer = get_renderer(renderer_name, get_tokenizer(model))
    client = client or _sampling_client(tinker.ServiceClient(), None, model)
    sem = asyncio.Semaphore(16)

    async def attempt(i: int):
        r = ds[i]
        _, _, final = r["answer"].rpartition("####")
        gold = final.strip().replace(",", "")
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": r["question"]}]
        toks = renderer.build_generation_prompt(msgs, effort=0.9).to_ints()
        # HF from_list needs one schema: a text-part list renders identically
        # to a bare string, so every content goes in as a list
        stored = [{"role": m["role"],
                   "content": [{"type": "text", "text": m["content"]}]}
                  for m in msgs]
        async with sem:
            try:
                resp = await client.sample_async(
                    ModelInput.from_ints(toks), num_samples=1,
                    sampling_params=SamplingParams(
                        max_tokens=max_tokens, temperature=0.6,
                        stop=renderer.get_stop_sequences(), seed=0))
            except Exception:  # noqa: BLE001 — one bad draw must not lose the batch
                return {"i": i, "drop": "error", "gen_tokens": 0, "row": None}
        seq = resp.sequences[0]
        msg, term = renderer.parse_response(seq.tokens)
        text = get_text_content(msg)
        correct = term.is_stop_sequence and grade("gsm8k", text, gold).verdict == 1.0
        drop = "truncated" if not term.is_stop_sequence else None if correct else "wrong"
        return {
            "i": i, "drop": drop, "gen_tokens": len(seq.tokens),
            "row": {"messages": [*stored, msg]} if correct else None,
        }

    kept, dropped, attempted, gen_tok = [], {"wrong": 0, "truncated": 0, "error": 0}, 0, []
    # bounded batches in prompt order so early-stop does not waste draws
    for lo in range(0, min(pool, len(idx)), 64):
        for a in await asyncio.gather(*[attempt(i) for i in idx[lo:lo + 64]]):
            attempted += 1
            if a["row"] is not None and len(kept) < n:
                kept.append(a["row"])
                gen_tok.append(a["gen_tokens"])
            elif a["drop"]:
                dropped[a["drop"]] += 1
        if len(kept) >= n:
            break
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in kept) + "\n")
    mean_tok = round(sum(gen_tok) / len(gen_tok)) if gen_tok else 0
    print(f"kept {len(kept)}/{attempted} attempted · mean {mean_tok} gen tok "
          f"· dropped {dropped['wrong']} wrong, {dropped['truncated']} truncated, "
          f"{dropped['error']} errored → {out}")
    return 0

# construct the config for training run 
def build_config(args):
    renderer_name = model_info.get_recommended_renderer_name(args.model)
    # knobs on the config
    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=args.model,
        renderer_name=renderer_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    # chz blueprint
    builder = FromConversationFileBuilder(common_config=common, file_path=str(args.data))
    store = Store(args.store)
    # held by name: the cookbook SFT loop saves sampler_weights/final without
    # an eval, so the recipe calls this instance once more after train.main
    evaluator = RegressionEvaluator(
        manifest=args.manifest,
        baseline=args.baseline or args.model,
        store=store,
        log_dir=args.log_dir,
        eval_every=args.eval_every,
        efforts=(0.9,),
        k=args.k,
        temperature=0.6,
        max_tokens=32768,
        diverge=args.diverge,
        # recorded into each run's provenance so the GUI can show
        # which knobs produced a checkpoint
        train_config={
            "lr": args.lr, "lr_schedule": args.lr_schedule,
            "epochs": args.epochs, "batch_size": args.batch_size,
            "lora_rank": args.lora_rank, "max_length": args.max_length,
            "max_steps": args.max_steps,
        },
        effort_pair=(0.2, 0.9) if args.effort_pair else None,
        study=args.study or args.log_dir.name,
    )
    return chz.Blueprint(train.Config).apply(
        {
            "log_path": str(args.log_dir),
            "model_name": args.model,
            "recipe_name": "flipbook_demo_sft",
            "renderer_name": renderer_name,
            "dataset_builder": builder,
            "learning_rate": args.lr,
            "lr_schedule": args.lr_schedule,
            "num_epochs": args.epochs,
            "max_steps": args.max_steps,
            "lora_rank": args.lora_rank,
            "save_every": args.save_every,
            "eval_every": 0,
            # fires at step 0 (base weights) then every 8 steps
            "infrequent_eval_every": args.eval_every,
            "infrequent_evaluator_builders": [lambda: evaluator],
        }
    ).make(), evaluator


def count_tokens(config) -> tuple[int, int]:
    train_ds, _ = config.dataset_builder()
    n_batches = len(train_ds)
    total = sum(
        datum.model_input.length
        for b in range(n_batches)
        for datum in train_ds.get_batch(b)
    )
    return total * config.num_epochs, n_batches * config.num_epochs

# dry or true run of train loop
def run_train(args) -> int:
    config, evaluator = build_config(args)
    tokens, steps = count_tokens(config)
    cost = estimate_usd(args.model, "train", tokens)
    n_evals = steps // args.eval_every + 1
    print(f"{tokens:,} training tokens over {steps} steps, "
          f"{steps // args.save_every + 1} checkpoints")
    print(f"forecast ~${cost:.2f} training + {n_evals} in-loop eval points "
          f"(30 rows x k={args.k} each, priced separately by flipbook)")
    if not args.yes:
        print("dry run: pass --yes to train")
        return 3  # distinct from success so the CLI wrapper skips the "inspect" hint
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="raise")
    asyncio.run(train.main(config))
    # the cookbook loop saves sampler_weights/final and never evals it: score
    # it here so the study arc ends at the trained weights. The evaluator's
    # already-evaluated short-circuit makes this a no-op when the loop's own
    # last eval lands on the same step
    last = checkpoint_utils.get_last_checkpoint(
        str(args.log_dir), required_key="sampler_path")
    if last is not None:
        import tinker

        client = tinker.ServiceClient().create_sampling_client(
            model_path=last.sampler_path)
        asyncio.run(evaluator(client, step=steps))
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--out", type=Path, default=None)
    d.add_argument("--n", type=int, default=1024)
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--distill", action="store_true",
                   help="targets are the model's own sampled responses "
                        "(thinking segment preserved) rather than human solutions")
    d.add_argument("--model", default=MODEL)
    d.add_argument("--max-tokens", type=int, default=4096)
    d.add_argument("--pool", type=int, default=1200,
                   help="prompts to try, in the same shuffled order as --n")
    d.add_argument("--yes", action="store_true")
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, default=Path("data/gsm8k_sft.jsonl"))
    t.add_argument("--log-dir", type=Path, default=Path("runs_sft/demo_eval"))
    t.add_argument("--model", default=MODEL,
                   help="base model to fine-tune, e.g. thinkingmachines/Inkling")
    t.add_argument("--manifest", required=True)
    t.add_argument("--baseline", default=None,
                   help="run ref for in-loop eval comparison; defaults to --model as a fresh eval")
    t.add_argument("--store", default="flipbook_store")
    t.add_argument("--study", default=None)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--lr-schedule", choices=["linear", "cosine", "constant"], default="linear")
    t.add_argument("--lora-rank", type=int, default=32)
    t.add_argument("--max-length", type=int, default=2048)
    t.add_argument("--max-steps", type=int, default=None)
    t.add_argument("--save-every", type=int, default=8)
    t.add_argument("--eval-every", type=int, default=8)
    t.add_argument("--k", type=int, default=4)
    t.add_argument("--diverge", action=argparse.BooleanOptionalAction, default=True)
    t.add_argument("--effort-pair", action=argparse.BooleanOptionalAction, default=True)
    t.add_argument("--yes", action="store_true")
    args = p.parse_args()
    if args.cmd == "data":
        if args.distill:
            out = args.out or Path("data/gsm8k_sft_thinking.jsonl")
            return asyncio.run(build_distilled_data(
                out, args.n, args.seed, args.model, args.max_tokens,
                args.pool, args.yes))
        return build_data(args.out or Path("data/gsm8k_sft.jsonl"), args.n, args.seed) or 0
    return run_train(args)


if __name__ == "__main__":
    raise SystemExit(main())
