"""`flipbook` CLI spine
"""


import argparse
import asyncio
import json
import sys

from flipbook.budget import budget
from flipbook.config import RunConfig
from flipbook.diverge import diverge
from flipbook.effort import effort_gap
from flipbook.lint import lint
from flipbook.manifest import Manifest
from flipbook.runner import LintFailed, evaluate, resolve_model
from flipbook.stats import compare, gate
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
    p = sub.add_parser("compare", help="paired stats between two runs")
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("--gate", action="store_true", help="exit 2 on a significant regression")
    p.add_argument("--max-regression", type=float, default=0.02)
    p.add_argument("--max-new-truncation-rate", type=float, default=0.0)
    p.add_argument("--json", action="store_true")
    p.add_argument("--markdown", action="store_true")
    _add_store(p)
    p = sub.add_parser("diverge", help="per-token logprob gap of ckpt vs base on base's traces")
    p.add_argument("--base", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--rows", choices=["flips", "all"], default="all")
    p.add_argument("--forecast", action="store_true")
    _add_store(p)
    p = sub.add_parser("effort", help="effort-prefix gap for one run's traces")
    p.add_argument("--run", required=True)
    p.add_argument("--pair", required=True, help="e_low,e_high")
    p.add_argument("--forecast", action="store_true")
    _add_store(p)
    p = sub.add_parser("track", help="evaluate every checkpoint in a cookbook log dir")
    p.add_argument("log_dir")
    p.add_argument("--manifest", required=True)
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--every", type=int)
    sel.add_argument("--last", type=int)
    p.add_argument("--include-base", action="store_true")
    p.add_argument("--base-model", default=None, help="offline override for tinker:// resolution")
    p.add_argument("--diverge", action="store_true")
    p.add_argument("--effort-pair", default=None, help="e_low,e_high")
    p.add_argument("--effort", type=float, default=0.9)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--study", default=None)
    p.add_argument("--forecast", action="store_true")
    p.add_argument("--concurrency", type=int, default=8)
    _add_store(p)
    p = sub.add_parser("import-evalstore", help="import a legacy eval bundle into the store")
    p.add_argument("path")
    _add_store(p)
    p = sub.add_parser("import-metrics", help="import metrics.jsonl from a cookbook log dir")
    p.add_argument("log_dir")
    p.add_argument("--study", required=True)
    _add_store(p)
    p = sub.add_parser("serve", help="read-only API + GUI over the store")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8484)
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
    if args.cmd == "compare":
        pair = compare(_store(args), args.run_a, args.run_b)
        if args.json:
            print(json.dumps(pair.to_dict(), indent=2))
        elif args.markdown:
            a = pair.agreement
            print(f"| | {args.run_a} | {args.run_b} |\n|---|---|---|")
            print(f"| acc | {pair.acc_a:.3f} | {pair.acc_b:.3f} |")
            print(f"| delta | — | {pair.delta:+.3f} "
                  f"[{pair.delta_ci[0]:+.3f}, {pair.delta_ci[1]:+.3f}] |")
            print(f"| truncation | {pair.truncation_rate_a:.2f} | {pair.truncation_rate_b:.2f} |")
            print(f"| cost | ${pair.cost_a:.3f} | ${pair.cost_b:.3f} |")
            print(f"| agreement | both {a['both_right']}/{a['both_wrong']} "
                  f"| a_only {a['a_only']} b_only {a['b_only']} |")
            print(f"\n{pair.n_pairs} paired rows, {len(pair.excluded)} excluded")
            for f in pair.flips:
                print(f"- {'**HARD** ' if f['hard'] else ''}{f['kind']}: "
                      f"`{f['row_id']}` {f['p_a']:.2f} → {f['p_b']:.2f}")
        else:
            a = pair.agreement
            print(f"{pair.run_a} acc={pair.acc_a:.3f}  vs  {pair.run_b} acc={pair.acc_b:.3f}")
            print(f"delta {pair.delta:+.3f}  CI [{pair.delta_ci[0]:+.3f}, {pair.delta_ci[1]:+.3f}]"
                  f"  on {pair.n_pairs} paired rows ({len(pair.excluded)} excluded)")
            print(f"agreement: both {a['both_right']}/{a['both_wrong']} "
                  f"a_only {a['a_only']} b_only {a['b_only']}")
            print(f"flips: {len(pair.flips)}  "
                  f"truncation {pair.truncation_rate_a:.2f} → {pair.truncation_rate_b:.2f}  "
                  f"cost ${pair.cost_a:.3f} vs ${pair.cost_b:.3f}")
            for f in pair.flips:
                tag = "HARD " if f["hard"] else ""
                print(f"  {tag}{f['kind']}: {f['row_id']}  {f['p_a']:.2f} → {f['p_b']:.2f}")
            for e in pair.excluded:
                print(f"  excluded {e['row_id']}: {e['reason']}")
        if args.gate:
            ok, reasons = gate(pair, args.max_regression, args.max_new_truncation_rate)
            for r in reasons:
                print(f"GATE FAIL: {r}", file=sys.stderr)
            return 0 if ok else 2
        return 0
    if args.cmd == "diverge":
        store = _store(args)
        rows = None
        if args.rows == "flips":
            pair = compare(store, args.base, args.ckpt)
            rows = [f["row_id"] for f in pair.flips]
        s = diverge(store, args.base, args.ckpt, rows, forecast=args.forecast)
        if args.forecast:
            print(f"forecast: {s.n_rows} rows · ~{s.forecast['prefill_tokens']:,} prefill tok "
                  f"· ~${s.est_cost_usd or 0:.3f} discount · ~${s.forecast['usd_list'] or 0:.3f} list")
            return 0
        print(f"{s.base_run_id} → {s.ckpt_run_id}: {s.n_rows} rows · "
              f"mean gap {s.mean_sum_nats:+.1f} nats · diverged {s.n_diverged} · "
              f"p_skip {s.mean_p_skip_base:.3f} → {s.mean_p_skip_ckpt:.3f} · ~${s.est_cost_usd or 0:.4f}")
        return 0
    if args.cmd == "effort":
        e_low, e_high = (float(x) for x in args.pair.split(","))
        s = effort_gap(_store(args), args.run, e_low, e_high, forecast=args.forecast)
        if args.forecast:
            print(f"forecast: {s.n_rows} traces · ~{s.forecast['prefill_tokens']:,} prefill tok "
                  f"· ~${s.est_cost_usd or 0:.3f} discount · ~${s.forecast['usd_list'] or 0:.3f} list")
            return 0
        print(f"{s.run_id} effort {s.e_low}→{s.e_high}: {s.n_rows} traces · "
              f"mean gap {s.mean_gap_nats:+.1f} nats · ~${s.est_cost_usd or 0:.4f}")
        return 0
    if args.cmd == "track":
        from flipbook.track import track
        pair = tuple(float(x) for x in args.effort_pair.split(",")) if args.effort_pair else None
        plan = track(
            _store(args), args.log_dir, args.manifest,
            every=args.every, last=args.last, include_base=args.include_base,
            base_model=args.base_model, diverge=args.diverge, effort_pair=pair,
            effort=args.effort, k=args.k, temperature=args.temperature,
            max_tokens=args.max_tokens, seed=args.seed, study=args.study,
            forecast=args.forecast, concurrency=args.concurrency,
        )
        if args.forecast:
            cost = f"~${plan.est_cost_usd:.3f}" if plan.est_cost_usd is not None else "cost unknown"
            print(f"forecast: {len(plan.checkpoints)} checkpoints"
                  f"{' + base' if plan.base_run_id else ''} · {plan.total_cells} cells · {cost}")
            return 0
        print(f"tracked {len(plan.checkpoints)} checkpoints into study {args.study or args.log_dir}")
        return 0
    if args.cmd == "import-evalstore":
        from flipbook.import_evalstore import import_evalstore
        rid = import_evalstore(_store(args), args.path)
        print(f"imported {args.path} as run {rid}")
        return 0
    if args.cmd == "import-metrics":
        from flipbook.import_metrics import import_metrics
        n = import_metrics(_store(args), args.log_dir, args.study)
        print(f"imported {n} metric rows for study {args.study}")
        return 0
    if args.cmd == "serve":
        from flipbook.api import serve
        print(f"serving {args.store} at http://{args.host}:{args.port}")
        serve(args.store, host=args.host, port=args.port)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
