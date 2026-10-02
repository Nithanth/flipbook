"""`flipbook` CLI spine

"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from flipbook.budget import budget
from flipbook.config import RunConfig
from flipbook.lint import lint
from flipbook.manifest import Manifest
from flipbook.runner import LintFailed, evaluate, resolve_model
from flipbook.store import Store


def _store(args) -> Store:
    return Store(args.store)


def _add_store(p: argparse.ArgumentParser) -> None:
    p.add_argument("--store", default="./flipbook_store")


def _add_eval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--effort", type=float, default=None)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--renderer", default=None)
    p.add_argument("--base-model", default=None, help="offline override for tinker:// resolution")
    p.add_argument("--label", default=None)
    p.add_argument("--study", default=None)


async def _resolve(args) -> tuple[str, str]:
    """→ (base_model, renderer). Creates a service client only for tinker://."""
    sc = None
    if args.model.startswith("tinker://") and args.base_model is None:
        import tinker

        sc = tinker.ServiceClient()
    r = await resolve_model(
        args.model, renderer=args.renderer, base_model=args.base_model, service_client=sc
    )
    return r.base_model, r.renderer


def _build_cfg(args, store: Store, base_model: str, renderer: str) -> RunConfig:
    doc = store.manifest_doc(args.manifest)
    if doc is None:
        raise SystemExit(f"no manifest {args.manifest!r} in {args.store}")
    return RunConfig(
        model=args.model,
        manifest_hash=doc["manifest_hash"],
        effort=args.effort,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        k=args.k,
        seed=args.seed,
        renderer=renderer,
        study=args.study,
        label=args.label,
    )


def _print_findings(findings) -> int:
    for f in findings:
        print(f"{f.level.upper()} {f.code}: {f.message}")
    return 2 if any(f.level == "error" for f in findings) else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="flipbook")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("freeze", help="freeze a benchmark subset into a manifest")
    p.add_argument("--benchmark", action="append", required=True, help="name:n, repeatable")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", required=True)
    _add_store(p)

    p = sub.add_parser("eval", help="run a config over a manifest")
    _add_eval_args(p)
    p.add_argument("--forecast", action="store_true")
    p.add_argument("--concurrency", type=int, default=8)
    _add_store(p)

    p = sub.add_parser("lint", help="check a config or stored run")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run")
    src.add_argument("--config", action="store_true")
    _add_eval_args(p)
    _add_store(p)

    p = sub.add_parser("budget", help="token/cost distribution for a run")
    p.add_argument("run")
    _add_store(p)

    args = ap.parse_args(argv)

    if args.cmd == "freeze":
        benches = {b: int(n) for b, n in (x.rsplit(":", 1) for x in args.benchmark)}
        m = Manifest.freeze(benches, seed=args.seed, name=args.name)
        _store(args).put_manifest(m.to_doc(), m.rows)
        print(f"froze {m.name}: {len(m.rows)} rows, hash {m.manifest_hash[:16]}")
        return 0

    if args.cmd == "eval":
        store = _store(args)
        base_model, renderer = asyncio.run(_resolve(args))
        cfg = _build_cfg(args, store, base_model, renderer)
        findings = lint(cfg, base_model)
        rc = _print_findings(findings)
        if rc:
            return rc
        try:
            summ = evaluate(cfg, store, forecast=args.forecast, concurrency=args.concurrency)
        except LintFailed as e:
            return _print_findings(e.findings)
        if args.forecast:
            f = summ.forecast
            print(f"forecast: {f.cells} cells · {f.prompt_tokens:,} prompt tok · "
                  f"~{f.est_gen_tokens:,} gen tok (prior: {f.prior})")
            if f.usd_discount is not None:
                print(f"~${f.usd_discount:.3f} discount · ~${f.usd_list:.3f} list")
            else:
                print("cost: unknown (no price table)")
            print("dry run — no sampling performed")
            return 0
        print(f"run {summ.run_id}: {summ.n_new_samples} new samples, "
              f"{summ.n_errors} errors, pass1={summ.pass1}, "
              f"~${summ.est_cost_usd or 0:.4f}")
        return 0

    if args.cmd == "lint":
        store = _store(args)
        if args.run:
            f = store.path / "runs" / f"{args.run}.json"
            if not f.exists():
                raise SystemExit(f"no run {args.run!r}")
            rec = json.loads(f.read_text())
            cfg = RunConfig(
                model=rec["checkpoint_path"] or rec["model_id"],
                manifest_hash=rec["manifest_hash"], effort=rec["effort"],
                temperature=rec["temperature"], max_tokens=rec["max_tokens"],
                k=rec["k"], seed=rec["seed"], renderer=rec["renderer"],
                study=rec.get("study"), label=rec.get("label"),
            )
            findings = lint(cfg, rec["model_id"])
        else:
            base_model, renderer = asyncio.run(_resolve(args))
            cfg = _build_cfg(args, store, base_model, renderer)
            findings = lint(cfg, base_model)
        return _print_findings(findings) if findings else 0

    if args.cmd == "budget":
        b = budget(_store(args), args.run)
        print(json.dumps(b.__dict__, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
