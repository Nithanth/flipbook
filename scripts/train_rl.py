"""RL run with RegressionEvaluator in the training loop.

Mirrors train_demo.py (SFT) so the two recipes read side by side: same
evaluator, same study layout, same forecast-then---yes gate. The training
signal is the cookbook math env's verifiable reward (correct boxed answer,
small format penalty, optional per-token length penalty or a binary
short-answer reward on gsm8k), so the
question flipbook asks is "does held-out eval move with reward, and what
breaks on the way?"

Cost is dominated by rollouts (groups_per_batch x group_size x max_tokens per
step), not by the optimizer, so the forecast is a sampling estimate.
"""

import argparse
import asyncio
import dataclasses
from functools import partial
from pathlib import Path

import chz
from tinker_cookbook import checkpoint_utils, cli_utils, model_info, renderers
from tinker_cookbook.recipes.math_rl import arithmetic_env, math_env
from tinker_cookbook.rl.train import Config
from tinker_cookbook.rl.train import main as rl_main
from tinker_cookbook.tokenizer_utils import get_tokenizer

from flipbook.evaluator import RegressionEvaluator
from flipbook.import_metrics import import_metrics
from flipbook.import_rollouts import import_rollouts
from flipbook.pricing import estimate_usd
from flipbook.store import Store

MODEL = "thinkingmachines/Inkling-Small"


class LengthPenalizedEnv(math_env.MathEnv):
    """MathEnv that also charges for every generated token:
    reward = correct - format penalty - length_coef * len(action) / max_tokens.
    The cheapest way to earn it is to think less; that is the point."""

    def __init__(self, *args, length_coef: float, max_tokens: int, **kw):
        super().__init__(*args, **kw)
        self.length_coef = length_coef
        self.max_tokens = max_tokens

    async def step(self, action, *, extra=None):
        r = await super().step(action, extra=extra)
        penalty = self.length_coef * len(action) / self.max_tokens
        r.metrics["length_penalty"] = penalty
        return dataclasses.replace(r, reward=r.reward - penalty)


class ShortAnswerEnv(math_env.MathEnv):
    """Reward 1 for any well-formed response of at most `short_tokens` tokens,
    0 otherwise; correctness is ignored but still logged. A reward the
    optimizer can satisfy by dropping the reasoning - the mechanism test for
    'reward up, held-out down'."""

    def __init__(self, *args, short_tokens: int, **kw):
        super().__init__(*args, **kw)
        self.short_tokens = short_tokens

    async def step(self, action, *, extra=None):
        r = await super().step(action, extra=extra)
        short = len(action) <= self.short_tokens
        well_formed = r.metrics.get("format", 1.0) >= 1.0
        r.metrics["short"] = float(short)
        return dataclasses.replace(r, reward=float(short and well_formed))


class ShapedGsm8kDataset(math_env.Gsm8kDataset):
    def __init__(self, *args, length_coef, short_tokens, max_tokens, test_size, **kw):
        super().__init__(*args, **kw)
        self.length_coef, self.short_tokens, self.max_tokens = (
            length_coef, short_tokens, max_tokens)
        # WHY: the cookbook rolls out the whole 1319-row test split at every
        # eval; a fixed slice keeps the in-distribution check to a few dollars
        if kw.get("split") == "test":
            self.ds = self.ds.select(range(min(test_size, len(self.ds))))

    def _make_env_group_builder(self, x, group_size):
        b = super()._make_env_group_builder(x, group_size)
        if b is None:
            return None
        if self.short_tokens <= 0 and self.length_coef <= 0:
            return b
        thunk = b.env_thunk  # partial(MathEnv, problem, answer, renderer, convo_prefix=...)
        if self.short_tokens > 0:
            env_thunk = partial(ShortAnswerEnv, *thunk.args,
                                short_tokens=self.short_tokens, **thunk.keywords)
        else:
            env_thunk = partial(LengthPenalizedEnv, *thunk.args,
                                length_coef=self.length_coef,
                                max_tokens=self.max_tokens, **thunk.keywords)
        return math_env.ProblemGroupBuilder(env_thunk=env_thunk, num_envs=group_size)


@chz.chz
class ShapedGsm8kBuilder(math_env.Gsm8kDatasetBuilder):
    length_coef: float = 0.0
    short_tokens: int = 0
    max_tokens: int
    test_size: int = 64

    async def __call__(self) -> tuple[math_env.Gsm8kDataset, math_env.Gsm8kDataset]:
        if self.convo_prefix == "standard":
            convo_prefix = math_env.MathEnv.standard_fewshot_prefix()
        else:
            convo_prefix = self.convo_prefix
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        datasets = [
            ShapedGsm8kDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                convo_prefix=convo_prefix,
                split=split,
                seed=self.seed,
                length_coef=self.length_coef,
                short_tokens=self.short_tokens,
                max_tokens=self.max_tokens,
                test_size=self.test_size,
            )
            for split in ("train", "test")
        ]
        return (datasets[0], datasets[1])


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
    # WHY: gsm8k always takes this builder, even with no reward shaping,
    # because it is also what caps the test split the cookbook rolls out whole
    if args.env == "gsm8k":
        return ShapedGsm8kBuilder(
            batch_size=args.groups_per_batch,
            model_name_for_tokenizer=args.model,
            renderer_name=renderer_name,
            group_size=args.group_size,
            seed=args.seed,
            length_coef=args.length_coef,
            short_tokens=args.short_tokens,
            max_tokens=args.max_tokens,
            test_size=args.test_size,
        )
    return math_env.get_math_dataset_builder(
        dataset_name=args.env,
        batch_size=args.groups_per_batch,
        model_name_for_tokenizer=args.model,
        renderer_name=renderer_name,
        group_size=args.group_size,
        seed=args.seed,
    )


def _start_batch(args) -> int:
    """The batch a --resume run continues from; 0 for a fresh run."""
    if not args.resume:
        return 0
    last = checkpoint_utils.get_last_checkpoint(str(args.log_dir))
    return last.batch if last else 0


def build_config(args) -> tuple[Config, RegressionEvaluator]:
    start = _start_batch(args)
    if start >= args.steps:
        raise SystemExit(
            f"nothing to do: log dir is at batch {start}, --steps is {args.steps}")
    renderer_name = model_info.get_recommended_renderer_name(args.model)
    store = Store(args.store)
    study = args.study or args.log_dir.name
    # held by name: the RL loop stops after saving `final` without evaluating
    # it, so the recipe calls this instance once more after rl_main returns
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
        train_config={
            "recipe": "rl", "env": args.env, "lr": args.lr,
            "group_size": args.group_size, "groups_per_batch": args.groups_per_batch,
            "rollout_max_tokens": args.max_tokens, "kl_coef": args.kl_coef,
            "lora_rank": args.lora_rank, "steps": args.steps,
            "length_coef": args.length_coef,
            "short_tokens": args.short_tokens,
        },
        effort_pair=(0.2, 0.9) if args.effort_pair else None,
        study=study,
        start_step=start,
    )
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
        evaluator_builders=[lambda: evaluator],
    ), evaluator


def forecast(args) -> tuple[int, float, float]:
    """Upper-bound rollout tokens: every rollout runs to max_tokens. Real
    usage is lower (Inkling stops early on easy problems), so the printed
    number is a ceiling, not an estimate. Returns (rollout tokens, rollout
    usd, in-distribution test usd)."""
    remaining = args.steps - _start_batch(args)
    rollouts = remaining * args.groups_per_batch * args.group_size
    gen = rollouts * args.max_tokens
    n_evals = (
        remaining // args.eval_every
        if args.resume
        else args.steps // args.eval_every + 1
    )
    test_tok = n_evals * args.test_size * args.max_tokens if args.env == "gsm8k" else 0
    return gen, estimate_usd(args.model, "sample", gen), estimate_usd(args.model, "sample", test_tok)


def run_train(args) -> int:
    gen, cost, test_cost = forecast(args)
    start = _start_batch(args)
    remaining = args.steps - start
    n_evals = (
        remaining // args.eval_every
        if args.resume
        else args.steps // args.eval_every + 1
    )
    resumed = f"resume from batch {start}: " if start else ""
    print(f"{resumed}{remaining} steps x {args.groups_per_batch} groups x {args.group_size} rollouts "
          f"x <= {args.max_tokens} tok = <= {gen:,} rollout tokens")
    print(f"rollouts <= ${cost:.2f}")
    if args.env == "gsm8k":
        print(f"in-distribution test: {n_evals} x {args.test_size} rollouts "
              f"x <= {args.max_tokens} tok <= ${test_cost:.2f}")
    elif args.env != "arithmetic":
        print("in-distribution test: the cookbook rolls out this env's whole test split "
              "at every eval point; not forecast here")
    print(f"forecast <= ${cost + test_cost:.2f} + {n_evals} in-loop eval points"
          f"{' after the resumed step' if args.resume else ''} "
          f"(manifest rows x k={args.k} each, priced by flipbook)")
    if not args.yes:
        print("dry run: pass --yes to train")
        return 3  # distinct from success so the CLI wrapper skips the "inspect" hint
    config, evaluator = build_config(args)
    cli_utils.check_log_dir(
        config.log_path,
        behavior_if_exists="resume" if args.resume else "raise")
    asyncio.run(rl_main(config))
    # the RL loop saves sampler_weights/final and never evals it (unlike the
    # SFT loop): score it here so the study arc includes the last checkpoint
    last = checkpoint_utils.get_last_checkpoint(
        str(args.log_dir), required_key="sampler_path"
    )
    if last is not None:
        import tinker

        client = tinker.ServiceClient().create_sampling_client(
            model_path=last.sampler_path
        )
        asyncio.run(evaluator(client, step=args.steps))
    # reward/kl/episode-length land next to pass@1 on the study page
    n = import_metrics(Store(args.store), args.log_dir, args.study or args.log_dir.name)
    imp = import_rollouts(Store(args.store), args.log_dir, args.study or args.log_dir.name)
    print(f"imported {n} training metrics and {imp.n_rollouts} rollouts "
          f"into study {args.study or args.log_dir.name!r}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="flipbook train rl",
                                description=__doc__.splitlines()[0])
    p.add_argument("--env", choices=["arithmetic", "gsm8k", "math", "polaris", "deepmath"],
                   default="arithmetic", help="cookbook math env providing prompts + verifiable reward")
    p.add_argument("--log-dir", type=Path, default=None,
                   help="defaults to runs_rl/<study>")
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
    p.add_argument("--length-coef", type=float, default=0.0,
                   help="subtract length_coef * tokens/max_tokens from each rollout's "
                        "reward (gsm8k only); a realistic 'stop rambling' penalty whose "
                        "cheapest exploit is to drop the reasoning")
    p.add_argument("--short-tokens", type=int, default=0,
                   help="binary reward: 1 for a well-formed answer of at most N "
                        "tokens, 0 otherwise, correctness ignored (gsm8k only); the "
                        "optimizer's cheapest path is to stop reasoning")
    p.add_argument("--test-size", type=int, default=64,
                   help="rows of the env's test split rolled out at each eval as the "
                        "in-distribution check (cookbook default is the whole split)")
    p.add_argument("--diverge", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--effort-pair", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--resume", action="store_true",
                   help="continue from the last checkpoint in --log-dir; --steps is the new total")
    p.add_argument("--yes", action="store_true")
    args = p.parse_args()
    if args.length_coef > 0 and args.short_tokens > 0:
        raise SystemExit("--short-tokens and --length-coef are alternative reward shapes; pick one")
    if (args.length_coef > 0 or args.short_tokens > 0) and args.env != "gsm8k":
        raise SystemExit("--length-coef and --short-tokens only apply to the gsm8k env")
    if args.log_dir is None:
        args.log_dir = Path("runs_rl") / (args.study or "demo_eval")
    if args.save_every % args.eval_every:
        raise SystemExit("--eval-every must divide --save-every (evals on unsaved steps "
                         "produce ephemeral weights that divergence cannot rescore)")
    return run_train(args)


if __name__ == "__main__":
    raise SystemExit(main())
