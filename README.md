# flipbook

> Trajectory-level regression analysis for Tinker fine-tuning: which examples
> flipped, *where in the reasoning* the checkpoints diverge, and whether model
> still responds to effort.

## Development

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check src tests
uv run scripts/verify_instrument.py   # --offline skips live API checks
```
