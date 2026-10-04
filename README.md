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
smaller run's k.

## Development

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check src tests
uv run scripts/verify_instrument.py   # --offline skips live API checks
```
