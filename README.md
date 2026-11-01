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
uv run flipbook report synthetic                            # one-paragraph summary of a study
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

**3. Train with evals in the loop.** `flipbook train` runs an included recipe
(SFT in `scripts/train_demo.py`, RL in `scripts/train_rl.py`) and evaluates
every saved checkpoint into the same study. Both print a cost forecast and
stop unless you pass `--yes`. For your own cookbook training loop, point
`flipbook track <log dir>` at it afterward, or add `RegressionEvaluator` from
`flipbook.evaluator` to your config's evaluator builders.

```bash
flipbook train sft data                                    # writes data/gsm8k_sft.jsonl
flipbook train sft train --study myrun --manifest aime30   # SFT; evals each checkpoint into the study
flipbook train rl --env gsm8k --study myrl --manifest aime30   # RL with verifiable math reward
flipbook eval myrun/final          # refs resolve checkpoint paths and inherit the study's settings
```

`flipbook train sft train --help` and `flipbook train rl --help` list each
recipe's knobs.

**4. Look at what changed.**

```bash
flipbook status                         # studies, cost, which steps still lack divergence or effort data
flipbook report myrun                   # the summary
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

- The CLI is the only thing that spends money or writes to the store. The GUI is read-only.
- The store is a directory of Parquet and JSON files (`./flipbook_store` by default, or `--store`, or `FLIPBOOK_STORE`). The server holds no state; copy the directory anywhere and serve it.
- Run identity is a hash of the full config. Re-running the same eval resumes from the samples already on disk, including after a killed process.
- Nothing is specific to math. The manifest decides the task and the grader decides the score; `examples/language_tasks.jsonl` and `examples/language_graders.py` show a non-math setup.

## An example of what it catches

Fine-tuning Inkling-Small on 1,024 GSM8K solutions (LoRA rank 32) dropped
its pass@1 on AIME 2026 from 0.93 to 0.02 by step 24, while train loss fell
steadily the whole way. The per-example evals showed why: the training targets
contained no reasoning segment, and the model learned to stop producing one.
At step 24, 42 of 60 responses were empty. The same recipe on the larger
Inkling model did the same thing. `flipbook report` on either study tells
that story from the stored samples.

## Development

```bash
uv sync
uv run pytest -q
uv run ruff check src tests
cd web && npm install && npm run build
```
