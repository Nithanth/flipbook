# flipbook

Regression analysis for [Tinker](https://thinkingmachines.ai/tinker/) fine-tuning runs.
Evaluate checkpoints on the same questions, compare scores and costs, and
inspect the examples that improved or regressed.

## Try it without a key

```bash
git clone https://github.com/Nithanth/flipbook && cd flipbook
uv sync
uv run flipbook demo --synthetic
uv run flipbook serve
```

`demo --synthetic` creates a study with clearly labeled synthetic data.
`serve` opens the GUI at http://127.0.0.1:8484.

To inspect it in the terminal:

```bash
uv run flipbook status                                    # what is in the store
uv run flipbook report synthetic                          # scores, verdict, and cost
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

Prefix the commands below with `uv run`. Sampling commands print a cost forecast first.

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

**3. Evaluate during training.** Add `RegressionEvaluator` to your cookbook
loop to compare checkpoints against a baseline and save the results as a study.

```python
from flipbook.evaluator import RegressionEvaluator
from flipbook.store import Store

evaluator = RegressionEvaluator(
    manifest="aime30", baseline="myrun/baseline", store=Store("flipbook_store"),
    study="myrun", eval_every=10, k=2,
)
config = Config(..., eval_every=10, save_every=10, evaluator_builders=[lambda: evaluator])
```

Use `evaluator_builders` for RL or `infrequent_evaluator_builders` for SFT.
Set matching save and eval intervals so evaluated checkpoints are saved.
If your loop skips the final evaluation, call `await evaluator(client, step=N)`
on the final checkpoint; the example recipes show how.

For a finished run, use `flipbook track <log dir> --manifest aime30 --study myrun`.
The included SFT and RL recipes print a forecast and stop unless you pass `--yes`.

```bash
flipbook train sft train --study myrun --manifest aime30     # SFT; evals each checkpoint into the study
flipbook train rl --env gsm8k --study myrl --manifest aime30 # RL with a verifiable reward
flipbook eval myrun/final          # refs resolve checkpoint paths and inherit the study's settings
```

**4. Look at what changed.**

```bash
flipbook status                         # studies, cost, which steps still lack divergence or effort data
flipbook report myrun                   # scores, verdict, and cost
flipbook compare myrun/baseline myrun/final
flipbook diverge --base myrun/baseline --ckpt myrun/final   # per-token logprob shift on the baseline's own traces
flipbook effort --run myrun/final --pair 0.2,0.9           # compare trace likelihood at two effort settings
flipbook serve
```

`flipbook guide` prints this workflow in the terminal. Tab completion:
`eval "$(register-python-argcomplete flipbook)"`.

## What the numbers mean

- **pass@1**: mean verdict per question. Works for binary and fractional graders.
- **pass@n**: probability at least one of n samples is fully correct (unbiased estimator).
- **score@n**: expected best score over n samples; shown when verdicts are not binary.
- **delta, CI**: mean per-question score change against the baseline, with a bootstrap 95% confidence interval. If it includes zero, the result is inconclusive.
- **flips**: questions that went right to wrong (regressions) or wrong to right (gains).
- **failure kinds**: labels such as empty, truncation, parse, wrong_answer, and partial.
- **divergence**: summed logprob difference on the baseline's sampled traces. Positive means the checkpoint assigns them higher probability; negative means lower.
- **p_skip**: probability the model ends its turn immediately.
- **effort gap**: logprob difference between two effort settings on the same traces. Near zero means little sensitivity on those traces.

## Design

- Tinker-integrated: evals accept model names or `tinker://` checkpoint
  paths; `RegressionEvaluator` plugs into a cookbook config's
  `evaluator_builders`; `track` reads cookbook log dirs. Costs come from
  sampled token counts times the model's price.
- The GUI is read-only. Sampling runs through the CLI or Python library.
- Data lives in Parquet and JSON files under `./flipbook_store`. Override
  the location with `--store` or `FLIPBOOK_STORE`.
- Eval configs are hashed into run IDs. Repeating an eval reuses stored samples.
- Custom tasks and graders are supported; see the language examples in `examples/`.

## What it catches

- **Lower loss, worse answers.** SFT loss can fall while held-out accuracy
  drops. Compare the failed examples, response lengths, and `p_skip` to
  check for empty answers or shortened reasoning.
- **Higher reward, worse answers.** An RL policy can earn more reward
  without improving on held-out tasks. Compare reward against eval scores,
  then inspect the examples that regressed.

`flipbook report <study>` prints checkpoint scores, a comparison verdict,
supporting metrics, and recorded eval cost. Paired comparisons use a 95%
confidence interval to distinguish improvement or regression from an
inconclusive result.

## Development

```bash
uv sync
uv run pytest -q
uv run ruff check src tests
cd web && npm install && npm run build
```
