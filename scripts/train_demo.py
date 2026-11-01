"""Demo SFT run with RegressionEvaluator in the training loop.

`data` is offline. `train` prints a token/cost forecast and requires --yes.
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
from tinker_cookbook import cli_utils, model_info
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

# construct the config for training run 
def build_config(args):
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    # knobs on the config
    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=MODEL,
        renderer_name=renderer_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    # chz blueprint
    builder = FromConversationFileBuilder(common_config=common, file_path=str(args.data))
    store = Store(args.store)
    return chz.Blueprint(train.Config).apply(
        {
            "log_path": str(args.log_dir),
            "model_name": MODEL,
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
            "infrequent_evaluator_builders": [
                lambda: RegressionEvaluator(
                    manifest=args.manifest,
                    baseline=args.baseline,
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
            ],
        }
    ).make()


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
    config = build_config(args)
    tokens, steps = count_tokens(config)
    cost = estimate_usd(MODEL, "train", tokens)
    n_evals = steps // args.eval_every + 1
    print(f"{tokens:,} training tokens over {steps} steps, "
          f"{steps // args.save_every + 1} checkpoints")
    print(f"forecast ~${cost:.2f} training + {n_evals} in-loop eval points "
          f"(30 rows x k={args.k} each, priced separately by flipbook)")
    if not args.yes:
        print("dry run — pass --yes to train")
        return 0
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="raise")
    asyncio.run(train.main(config))
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--out", type=Path, default=Path("data/gsm8k_sft.jsonl"))
    d.add_argument("--n", type=int, default=1024)
    d.add_argument("--seed", type=int, default=7)
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, default=Path("data/gsm8k_sft.jsonl"))
    t.add_argument("--log-dir", type=Path, default=Path("runs_sft/demo_eval"))
    t.add_argument("--manifest", required=True)
    t.add_argument("--baseline", default=MODEL)
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
        build_data(args.out, args.n, args.seed)
        return 0
    return run_train(args)


if __name__ == "__main__":
    raise SystemExit(main())
