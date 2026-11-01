"""RL run with RegressionEvaluator in the training loop.

Mirrors train_demo.py (SFT) so the two recipes read side by side: same
evaluator, same study layout, same forecast-then---yes gate. The training
signal is the cookbook math env's verifiable reward (correct boxed answer,
small format penalty), so the question flipbook asks is "does held-out eval
move with reward, and what breaks on the way?"

Cost is dominated by rollouts (groups_per_batch x group_size x max_tokens per
step), not by the optimizer, so the forecast is a sampling estimate.
"""

import argparse
import asyncio
from pathlib import Path

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.math_rl import arithmetic_env, math_env
from tinker_cookbook.rl.train import Config
from tinker_cookbook.rl.train import main as rl_main

from flipbook.evaluator import RegressionEvaluator
from flipbook.import_metrics import import_metrics
from flipbook.pricing import estimate_usd
from flipbook.store import Store

MODEL = "thinkingmachines/Inkling-Small"


def build_dataset(args, renderer_name: str):
    if args.env == "arithmetic":
        return arithmetic_env.ArithmeticDatasetBuilder(
            batch_size=args.groups_per_batch,
            model_name_for_tokenizer=args.model,
            renderer_name=renderer_name,
            n_batches=args.steps,
            include_fewshot=True,
            group_size=args.group_size,
        )
    return math_env.get_math_dataset_builder(
        dataset_name=args.env,
        batch_size=args.groups_per_batch,
        model_name_for_tokenizer=args.model,
        renderer_name=renderer_name,
        group_size=args.group_size,
        seed=args.seed,
    )


def build_config(args) -> Config:
    renderer_name = model_info.get_recommended_renderer_name(args.model)
    store = Store(args.store)
    study = args.study or args.log_dir.name
    return Config(
        learning_rate=args.lr,
        dataset_builder=build_dataset(args, renderer_name),
        model_name=args.model,
        recipe_name="flipbook_demo_rl",
        renderer_name=renderer_name,
        lora_rank=args.lora_rank,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        log_path=str(args.log_dir),
        kl_penalty_coef=args.kl_coef,
        # WHY keep this on: it adds a paid compute_logprobs pass per step, but
        # kl_pre_post_v1/v2 is the per-update policy drift the study page charts
        compute_post_kl=True,
        eval_every=args.eval_every,
        save_every=args.save_every,
        max_steps=args.steps,
        evaluator_builders=[
            lambda: RegressionEvaluator(
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
                train_config={
                    "recipe": "rl", "env": args.env, "lr": args.lr,
                    "group_size": args.group_size, "groups_per_batch": args.groups_per_batch,
                    "rollout_max_tokens": args.max_tokens, "kl_coef": args.kl_coef,
                    "lora_rank": args.lora_rank, "steps": args.steps,
                },
                effort_pair=(0.2, 0.9) if args.effort_pair else None,
                study=study,
            )
        ],
    )


def forecast(args) -> tuple[int, float]:
    """Upper-bound rollout tokens: every rollout runs to max_tokens. Real
    usage is lower (Inkling stops early on easy problems), so the printed
    number is a ceiling, not an estimate."""
    rollouts = args.steps * args.groups_per_batch * args.group_size
    gen = rollouts * args.max_tokens
    return gen, estimate_usd(args.model, "sample", gen)


def run_train(args) -> int:
    gen, cost = forecast(args)
    n_evals = args.steps // args.eval_every + 1
    print(f"{args.steps} steps x {args.groups_per_batch} groups x {args.group_size} rollouts "
          f"x <= {args.max_tokens} tok = <= {gen:,} rollout tokens")
    print(f"forecast <= ${cost:.2f} rollouts + {n_evals} in-loop eval points "
          f"(manifest rows x k={args.k} each, priced by flipbook)")
    if not args.yes:
        print("dry run: pass --yes to train")
        return 3  # distinct from success so the CLI wrapper skips the "inspect" hint
    config = build_config(args)
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="raise")
    asyncio.run(rl_main(config))
    # reward/kl/episode-length land next to pass@1 on the study page
    n = import_metrics(Store(args.store), args.log_dir, args.study or args.log_dir.name)
    print(f"imported {n} training metrics into study {args.study or args.log_dir.name!r}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--env", choices=["arithmetic", "gsm8k", "math", "polaris", "deepmath"],
                   default="arithmetic", help="cookbook math env providing prompts + verifiable reward")
    p.add_argument("--log-dir", type=Path, default=Path("runs_rl/demo_eval"))
    p.add_argument("--model", default=MODEL)
    p.add_argument("--manifest", required=True, help="held-out eval set the evaluator scores")
    p.add_argument("--baseline", default=None,
                   help="run ref for in-loop comparison; defaults to --model as a fresh eval")
    p.add_argument("--store", default="flipbook_store")
    p.add_argument("--study", default=None)
    p.add_argument("--steps", type=int, default=10, help="training iterations (batches)")
    p.add_argument("--groups-per-batch", type=int, default=8)
    p.add_argument("--group-size", type=int, default=4, help="rollouts per prompt")
    p.add_argument("--max-tokens", type=int, default=4096, help="rollout cap; thinking counts")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--kl-coef", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--eval-every", type=int, default=5,
                   help="must divide --save-every so eval'd weights are durable")
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--diverge", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--effort-pair", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--yes", action="store_true")
    args = p.parse_args()
    if args.save_every % args.eval_every:
        raise SystemExit("--eval-every must divide --save-every (evals on unsaved steps "
                         "produce ephemeral weights that divergence cannot rescore)")
    return run_train(args)


if __name__ == "__main__":
    raise SystemExit(main())
