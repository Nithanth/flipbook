# flipbook

> Trajectory-level regression analysis for Tinker fine-tuning: which examples
> flipped, *where in the reasoning* the checkpoints diverge, and whether model
> still responds to effort.

## Eval data

`flipbook freeze --benchmark name:n --name M` freezes a benchmark subset into
a manifest. For custom tasks, `--from-jsonl file.jsonl --benchmark NAME`
freezes one `{"question"|"messages", "gold", "grader_id"}` object per line
(`"system"` prepends a system message to the question form).

`flipbook graders` lists registered graders (`gsm8k`, `math500`, `aime`,
`exact`, `number`, `boxed`, `contains`); `regex:<pattern>` and
`module.path:func` also resolve. `flipbook grade TEXT --grader ID --gold G`
dry-runs a grader on one text.

`flipbook compare` reports pass@n (unbiased estimator) for every n up to the
smaller run's k, paired deltas with bootstrap CIs, and per-question flips.

`flipbook eval --config eval.toml` reads the same settings from TOML
(file = defaults, explicit flags always override — see
`examples/eval_baseline.toml`).

`flipbook diverge --base A --ckpt B` rescores the baseline's own sampled
traces under the checkpoint: per-token `lp_ckpt − lp_base` in nats, plus the
model's probability of emitting nothing (`p_skip`). This is how you see the
internals move *before* accuracy does. `flipbook effort` measures whether the
effort dial still modulates the model.

`flipbook guide` prints the full workflow. Tab completion via
`eval "$(register-python-argcomplete flipbook)"` completes run labels,
manifest names, studies, grader ids, and row ids.

## Inspecting

```bash
flipbook serve        # read-only GUI at localhost:8484, opens the browser
```

Everything lives in the store directory — the server owns no state, so a
store can be moved, copied, or served anywhere. Four screens: **study** (one
training run's checkpoints vs train loss — the "optimizer happy, eval broken"
chart, with a plain-language read of what moved first), **runs** (per-run
table with config provenance), **compare** (paired stats, CIs, flips),
**divergence** (per-token disagreement heatmap over the baseline's traces,
with a branch probe that asks "what would the checkpoint have said here?").

During training, `flipbook track <cookbook log dir> --manifest M --study S`
evaluates every checkpoint as it lands and groups the runs into a study —
that's what the study screen draws. The GUI is strictly read-only: the CLI is
the control panel (it forecasts spend before any sampling), the GUI is for
inspection.

`flipbook serve` loads `~/.secrets/tinker.env` if present — needed only for
the one paid GUI action (the branch probe). Nothing else in the UI spends
money.

## Development

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check src tests
uv run scripts/verify_instrument.py   # --offline skips live API checks
```
