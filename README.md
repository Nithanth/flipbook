# flipbook

> Trajectory-level regression analysis for Tinker fine-tuning: which examples
> flipped, *where in the reasoning* the checkpoints diverge, and whether model
> still responds to effort.

## Development

```bash
uv venv .venv --python 3.12
uv pip install -e '.[dev]'
source .venv/bin/activate
pytest -q && ruff check src tests
```
