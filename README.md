# flipbook

Regression analysis for [Tinker](https://thinkingmachines.ai/tinker/) fine-tuning runs.
You freeze an eval set, sample each checkpoint against it, and flipbook tells
you which examples got better, which got worse, what it cost, and where in the
output the model's behavior changed. A checkpoint can improve on its objective every
step while breaking on everything else, and nothing in the loop will tell you. Per-example
evals can diagnose this in flight.

## Try it without a key

```bash
git clone https://github.com/Nithanth/flipbook && cd flipbook
uv sync
uv run flipbook demo --synthetic
uv run flipbook serve
```

`demo --synthetic` writes a fabricated training study into a local store.
`serve` opens the GUI at http://127.0.0.1:8484. The data is invented and
labeled as such; it is there so every screen and command has something to show.

Terminal equivalents of the GUI:

```bash
uv run flipbook status                                      # what is in the store
uv run flipbook report synthetic                            # pass@1 arc + CI-gated verdict
uv run flipbook compare synthetic/baseline synthetic/step24 # paired stats and per-example flips
uv run flipbook show synthetic/step24 synthetic-1           # one question, every sample
```

## Use it on your own runs

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/), a Tinker API key.
Node is only needed to build the GUI (`cd web && npm install && npm run build`);
without it `serve` still exposes the JSON API.

```bash
export TINKER_API_KEY=...        # or put it in ~/.secrets/tinker.env
uv run flipbook doctor           # checks key, store, GUI build
```

Everything below is `uv run flipbook <command>`. Any command that samples a
model prints a cost forecast first.

**1. Freeze an eval set.** Rows, gold answers, and grader ids are hashed into a
manifest so every run is scored on identical inputs.

```bash
flipbook freeze --benchmark aime2026:30 --name aime30            # from a cookbook benchmark
flipbook freeze --from-jsonl tasks.jsonl --benchmark mine --name mine   # from your own file
```

A JSONL row is `{"question": ..., "gold": ..., "grader_id": ...}` (or
`"messages": [...]` instead of `"question"`). Built-in graders cover exact,
numeric, boxed, regex, and the math benchmarks; `module.path:func` loads your
own, including LLM-as-judge or fractional scores. `flipbook graders` lists them.

**2. Evaluate a model.**

```bash
flipbook eval --model thinkingmachines/Inkling-Small --manifest aime30 \
    --effort 0.9 --k 2 --study myrun --label baseline
```

**3. Train with evals in the loop.** `RegressionEvaluator` is a cookbook
`SamplingClientEvaluator`: add it to any training config's `evaluator_builders`
and every evaluated checkpoint lands in the study, compared against the
baseline, with divergence scored on the baseline's traces.

```python
from flipbook.evaluator import RegressionEvaluator
from flipbook.store import Store

evaluator = RegressionEvaluator(
    manifest="aime30", baseline="myrun/baseline", store=Store("flipbook_store"),
    study="myrun", eval_every=10, k=2,
)
config = Config(..., eval_every=10, save_every=10, evaluator_builders=[lambda: evaluator])
```

Works in the SFT and RL loops alike; `eval_every` should divide `save_every`
so evaluated weights are the saved ones. The RL loop saves its final
checkpoint without evaluating it, so call `await evaluator(client, step=N)`
on it afterward (the RL recipe shows how). For a run that already finished,
`flipbook track <log dir>` evaluates its saved checkpoints after the fact.
Two complete recipes ship as examples (`flipbook train sft`, `flipbook train
rl`); both print a cost forecast and stop unless you pass `--yes`.

```bash
flipbook train sft train --study myrun --manifest aime30     # SFT; evals each checkpoint into the study
flipbook train rl --env gsm8k --study myrl --manifest aime30 # RL with a verifiable reward
flipbook eval myrun/final          # refs resolve checkpoint paths and inherit the study's settings
```

**4. Look at what changed.**

```bash
flipbook status                         # studies, cost, which steps still lack divergence or effort data
flipbook report myrun                   # the arc and the verdict
flipbook compare myrun/baseline myrun/final
flipbook diverge --base myrun/baseline --ckpt myrun/final   # per-token logprob shift on the baseline's own traces
flipbook effort --run myrun/final --pair 0.2,0.9            # does the effort setting still change the output
flipbook serve
```

`flipbook guide` prints this workflow in the terminal. Tab completion:
`eval "$(register-python-argcomplete flipbook)"`.

## What the numbers mean

- **pass@1**: mean verdict per question. Works for binary and fractional graders.
- **pass@n**: probability at least one of n samples is fully correct (unbiased estimator).
- **score@n**: expected best score over n samples; shown when verdicts are not binary.
- **delta, CI**: paired per-question change against the baseline with a bootstrap 95% interval. Deltas inside the interval are noise.
- **flips**: questions that went right to wrong (regressions) or wrong to right (gains).
- **failure kinds**: empty, truncation, parse, wrong_answer, partial. Tells you *how* a sample failed, not just that it did.
- **divergence**: summed per-token logprob difference of a checkpoint against the baseline, scored on the baseline's own sampled traces. Measures how far the policy moved on those inputs.
- **p_skip**: probability the model ends its turn immediately.
- **effort gap**: logprob difference between two effort settings. Near zero means the setting no longer does anything.

## Design

- The GUI is read-only. The CLI owns all sampling and all writes.
- The store is a directory of Parquet and JSON files (`./flipbook_store`,
  or `--store`, or `FLIPBOOK_STORE`). Copy it anywhere and `serve` it; the
  server keeps no state.
- Run identity is a hash of the eval config, so re-running an eval resumes
  from what's already on disk instead of paying twice.
- No math assumptions: the manifest picks the task, the grader picks the
  score. `examples/` has a non-math setup.

## What it catches

The two failure shapes it was built for share one property: the training
metrics look healthy the whole time, and the only evidence is on the
held-out eval.

- **A fine-tune that stops reasoning.** Targets with no reasoning segment
  teach a thinking model to skip it. Loss falls while pass@1 collapses;
  the first symptoms are empty responses, rising `p_skip`, and divergence
  on the baseline's own traces.
- **A reward the policy can satisfy cheaply.** Pay for short or well-formed
  outputs and the reward saturates while held-out responses thin out.
  Length and divergence move before held-out accuracy does.

`flipbook report <study>` prints the pass@1 arc across a study's
checkpoints, then a verdict - regressed, improved, or within noise -
chosen by whether the paired 95% CI excludes zero, with the metrics that
moved first listed under it.

## Development

```bash
uv sync
uv run pytest -q
uv run ruff check src tests
cd web && npm install && npm run build
```
