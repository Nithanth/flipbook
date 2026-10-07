"""`flipbook` CLI spine
"""


import argparse
import asyncio
import json
import sys
import textwrap

from flipbook.budget import budget
from flipbook.config import RunConfig
from flipbook.diverge import diverge
from flipbook.effort import effort_gap
from flipbook.graders import strip_control_tokens
from flipbook.lint import lint
from flipbook.manifest import Manifest, question_text
from flipbook.runner import LintFailed, evaluate, resolve_model
from flipbook.stats import compare, gate
from flipbook.store import Store


def _store(args) -> Store:
    return Store(args.store)


def _run_rec(store: Store, ref: str) -> dict:
    """CLI surface for Store.resolve_run — a bad ref is a usage error."""
    try:
        return store.resolve_run(ref)
    except LookupError as e:
        raise SystemExit(str(e)) from e


def _run_id(store: Store, ref: str) -> str:
    return _run_rec(store, ref)["run_id"]


def _add_store(p: argparse.ArgumentParser) -> None:
    p.add_argument("--store", default="./flipbook_store")


def _add_eval_args(p: argparse.ArgumentParser, required: bool = True) -> None:
    # lint --run reads model/manifest off the stored run record instead
    p.add_argument("--model", required=required)
    p.add_argument("--manifest", required=required)
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


def _grader_ids(store: Store, name_or_hash: str) -> list[str] | None:
    """grader_ids on a manifest's rows, or None when they can't be read —
    lint just skips the check rather than failing on a store detail."""
    doc = store.manifest_doc(name_or_hash)
    if doc is None:
        return None
    f = store.path / "manifest_rows" / f"{doc['manifest_hash']}.parquet"
    if not f.exists():
        return None
    return [r["grader_id"] for r in store.manifest_rows(doc["manifest_hash"])]


def _print_findings(findings) -> int:
    for f in findings:
        print(f"{f.level.upper()} {f.code}: {f.message}")
    return 2 if any(f.level == "error" for f in findings) else 0


_RUN_REF_EPILOG = "run refs: run_id, unique id prefix, 'study/label', or a unique label"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="flipbook")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser(
        "freeze", help="freeze a benchmark subset into a manifest",
        epilog="--from-jsonl rows: {'question'|'messages', 'gold', 'grader_id'} — "
               "graders: builtins, 'regex:<pat>', 'module.path:fn', or 'path/file.py:fn'",
    )
    p.add_argument("--benchmark", action="append", required=True,
                   help="name:n, repeatable; with --from-jsonl, a single bare name")
    p.add_argument("--from-jsonl", default=None,
                   help="freeze rows from a jsonl file instead of a benchmark")
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
    _add_eval_args(p, required=False)
    _add_store(p)
    p = sub.add_parser("graders", help="list registered grader ids")
    p = sub.add_parser("grade", help="dry-run a grader on one text")
    p.add_argument("text")
    p.add_argument("--grader", required=True)
    p.add_argument("--gold", required=True)
    p = sub.add_parser("runs", help="list runs in the store")
    p.add_argument("--study")
    p.add_argument("--json", action="store_true")
    _add_store(p)
    p = sub.add_parser("budget", help="token/cost distribution for a run", epilog=_RUN_REF_EPILOG)
    p.add_argument("run")
    _add_store(p)
    p = sub.add_parser("show", help="inspect one row's samples in a run", epilog=_RUN_REF_EPILOG)
    p.add_argument("run", help="run id, prefix, study/label, or label")
    p.add_argument("row", help="row id or unique prefix (of id or hash suffix)")
    p.add_argument("--sample", type=int, default=None, help="only this sample index")
    p.add_argument("--full", action="store_true", help="print full text, not the tail")
    p.add_argument("--thinking", action="store_true", help="decode thinking from token_ids")
    p.add_argument("--json", action="store_true")
    _add_store(p)
    p = sub.add_parser("compare", help="paired stats between two runs", epilog=_RUN_REF_EPILOG)
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("--gate", action="store_true", help="exit 2 on a significant regression")
    p.add_argument("--max-regression", type=float, default=0.02)
    p.add_argument("--max-new-truncation-rate", type=float, default=0.0)
    p.add_argument("--json", action="store_true")
    p.add_argument("--markdown", action="store_true")
    _add_store(p)
    p = sub.add_parser("diverge", help="per-token logprob gap of ckpt vs base on base's traces", epilog=_RUN_REF_EPILOG)
    p.add_argument("--base", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--rows", choices=["flips", "all"], default="all")
    p.add_argument("--forecast", action="store_true")
    _add_store(p)
    p = sub.add_parser("effort", help="effort-prefix gap for one run's traces", epilog=_RUN_REF_EPILOG)
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
        if args.from_jsonl:
            if len(args.benchmark) != 1 or ":" in args.benchmark[0]:
                raise SystemExit("--from-jsonl takes a single bare --benchmark NAME (no :n)")
            m = Manifest.freeze_jsonl(
                args.from_jsonl, benchmark=args.benchmark[0], seed=args.seed, name=args.name
            )
        else:
            benches = {b: int(n) for b, n in (x.rsplit(":", 1) for x in args.benchmark)}
            m = Manifest.freeze(benches, seed=args.seed, name=args.name)
        _store(args).put_manifest(m.to_doc(), m.rows)
        print(f"froze {m.name}: {len(m.rows)} rows, hash {m.manifest_hash[:16]}")
        return 0
    if args.cmd == "eval":
        store = _store(args)
        base_model, renderer = asyncio.run(_resolve(args))
        cfg = _build_cfg(args, store, base_model, renderer)
        findings = lint(cfg, base_model, grader_ids=_grader_ids(store, args.manifest))
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
              f"{summ.n_errors} errors, pass@1={summ.pass1}, "
              f"~${summ.est_cost_usd or 0:.4f}")
        return 0
    if args.cmd == "lint":
        store = _store(args)
        if args.run:
            try:
                rec = store.resolve_run(args.run)
            except LookupError as e:
                raise SystemExit(str(e)) from e
            cfg = RunConfig(
                model=rec["checkpoint_path"] or rec["model_id"],
                manifest_hash=rec["manifest_hash"], effort=rec["effort"],
                temperature=rec["temperature"], max_tokens=rec["max_tokens"],
                k=rec["k"], seed=rec["seed"], renderer=rec["renderer"],
                study=rec.get("study"), label=rec.get("label"),
            )
            findings = lint(cfg, rec["model_id"],
                            grader_ids=_grader_ids(store, rec["manifest_hash"]))
        else:
            if not args.model or not args.manifest:
                raise SystemExit("lint --config needs --model and --manifest")
            base_model, renderer = asyncio.run(_resolve(args))
            cfg = _build_cfg(args, store, base_model, renderer)
            findings = lint(cfg, base_model, grader_ids=_grader_ids(store, args.manifest))
        return _print_findings(findings) if findings else 0
    if args.cmd == "graders":
        from flipbook.graders import GRADERS
        for gid, (_, desc) in GRADERS.items():
            print(f"{gid:<10} {desc}")
        print("regex:<pattern>  first capture group (or whole match) vs gold, normalized")
        print("module.path:func custom fn(text, gold) -> Grade | bool | float")
        return 0
    if args.cmd == "grade":
        from flipbook.graders import UnknownGraderError, grade
        try:
            g = grade(args.grader, args.text, args.gold)
        except UnknownGraderError as e:
            print(e, file=sys.stderr)
            return 2
        print(f"verdict={g.verdict} extracted={g.extracted!r} note={g.note}")
        return 0
    if args.cmd == "runs":
        store = _store(args)
        if args.json:
            print(json.dumps(store.runs(args.study), indent=2, default=str))
            return 0
        def step(r):
            return (r.get("train_step")
                    or (r.get("provenance") or {}).get("train_step_measured") or -1)
        # manifest_hash -> name, so the table shows which eval each run targets
        mnames = {
            d["manifest_hash"]: d["name"]
            for d in (
                json.loads(f.read_text())
                for f in sorted((store.path / "manifests").glob("*.json"))
            )
        }
        print(f"{'run':<12}  {'label':<12} {'study':<12} {'manifest':<14} "
              f"{'effort':<6} {'k':<3} {'model'}")
        for r in sorted(store.runs(args.study),
                        key=lambda x: (x.get("study") or "", step(x))):
            label = r.get("label") or "-"
            print(f"{r['run_id'][:12]}  {label:<12} {(r.get('study') or '-'):<12} "
                  f"{mnames.get(r.get('manifest_hash'), (r.get('manifest_hash') or '-')[:12]):<14} "
                  f"{r.get('effort')!s:<6} {r.get('k')!s:<3} {r.get('model_id') or ''}")
        return 0
    if args.cmd == "show":
        store = _store(args)
        rec = _run_rec(store, args.run)
        try:
            row = store.resolve_row(rec["manifest_hash"], args.row)
        except LookupError as e:
            raise SystemExit(str(e)) from e
        tbl = store.samples(rec["run_id"])
        samples = sorted(
            (r for r in tbl.to_pylist() if r["row_id"] == row["row_id"]),
            key=lambda r: r["sample_idx"],
        )
        if args.json:
            print(json.dumps(
                {
                    "run_id": rec["run_id"],
                    "study": rec.get("study"),
                    "label": rec.get("label"),
                    "row": row,
                    "samples": samples,
                },
                indent=2, default=str,
            ))
            return 0
        print(f"run {rec['run_id']}  ({rec.get('study') or '-'}/{rec.get('label') or '-'})")
        print(textwrap.fill("Q: " + question_text(row["messages"]), 100))
        print(f"gold: {row['gold']!r}   grader: {row['grader_id']}")
        from flipbook.decode import thinking

        if not samples:
            print("  (no samples for this row)")
        for s in samples:
            if args.sample is not None and s["sample_idx"] != args.sample:
                continue
            v = s["verdict"]
            badge = "✓" if v == 1.0 else ("✗" if v == 0.0 else (f"{v:.2f}" if v is not None else "err"))
            bits = [f"s{s['sample_idx']} {badge}", f"{s['gen_tokens'] or 0} tok",
                    f"stop={s['stop_reason'] or '-'}"]
            if s["extracted"] is not None:
                bits.append(f"extracted={s['extracted'][:40]!r}")
            if s["failure_kind"]:
                bits.append(f"failure={s['failure_kind']}")
            if s["grade_note"]:
                bits.append(f"note={s['grade_note'][:60]!r}")
            if s["error"]:
                bits.append(f"error={s['error'][:60]!r}")
            print("  " + "  ·  ".join(bits))
            if args.thinking:
                think = thinking(rec.get("model_id"), s.get("token_ids"), s.get("text"))
                if think:
                    print(textwrap.indent(think, "    thinking: "))
            text = strip_control_tokens(s["text"] or "").strip()
            if args.full:
                if text:
                    print(textwrap.indent(text, "    "))
            elif len(text) > 400:
                print(textwrap.indent("…" + text[-400:], "    "))
            elif text:
                print(textwrap.indent(text, "    "))
        return 0
    if args.cmd == "budget":
        store = _store(args)
        b = budget(store, _run_id(store, args.run))
        print(json.dumps(b.__dict__, indent=2))
        return 0
    if args.cmd == "compare":
        store = _store(args)
        pair = compare(
            store, _run_id(store, args.run_a), _run_id(store, args.run_b)
        )
        compat = pair.comparability
        if compat.get("blocks"):
            for msg in compat["blocks"]:
                print(f"error: {msg}", file=sys.stderr)
            return 1
        for msg in compat.get("warnings", []):
            print(f"warning: {msg}", file=sys.stderr)
        kind_by_row = {
            rid: k for k, rids in pair.failure_rows_b.items() for rid in rids
        }
        if args.json:
            print(json.dumps(pair.to_dict(), indent=2))
        elif args.markdown:
            a = pair.agreement
            print(f"| | {pair.run_a} | {pair.run_b} |\n|---|---|---|")
            print(f"| acc | {pair.acc_a:.3f} | {pair.acc_b:.3f} |")
            print(f"| delta | — | {pair.delta:+.3f} "
                  f"[{pair.delta_ci[0]:+.3f}, {pair.delta_ci[1]:+.3f}] |")
            print(f"| truncation | {pair.truncation_rate_a:.2f} | {pair.truncation_rate_b:.2f} |")
            print(f"| cost | ${pair.cost_a:.3f} | ${pair.cost_b:.3f} |")
            print(f"| agreement | both {a['both_right']}/{a['both_wrong']} "
                  f"| a_only {a['a_only']} b_only {a['b_only']} |")
            print(f"\n{pair.n_pairs} paired rows, {len(pair.excluded)} excluded")
            for f in pair.flips:
                q = f" — {f['q']}" if f.get("q") else ""
                print(f"- {'**HARD** ' if f['hard'] else ''}{f['kind']}: "
                      f"`{f['row_id']}` {f['p_a']:.2f} → {f['p_b']:.2f}{q}")
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
            def fk(d):
                return ", ".join(f"{k}×{v}" for k, v in sorted(d.items())) or "none"
            print(f"failure kinds: a: {fk(pair.failures['a'])}   b: {fk(pair.failures['b'])}")
            for n in sorted(pair.passn, key=int):
                e = pair.passn[n]
                print(f"  pass@{n} {e['a']:.3f} → {e['b']:.3f}")
            for f in pair.flips:
                tag = "HARD " if f["hard"] else ""
                kind = kind_by_row.get(f["row_id"])
                tail = f"  b:{kind}" if kind else ""
                q = f"  {f['q']!r}" if f.get("q") else ""
                print(f"  {tag}{f['kind']}: {f['row_id']}  {f['p_a']:.2f} → {f['p_b']:.2f}{tail}{q}")
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
        base, ckpt = _run_id(store, args.base), _run_id(store, args.ckpt)
        rows = None
        if args.rows == "flips":
            pair = compare(store, base, ckpt)
            rows = [f["row_id"] for f in pair.flips]
        s = diverge(store, base, ckpt, rows, forecast=args.forecast)
        if args.forecast:
            print(f"forecast: {s.n_rows} rows · ~{s.forecast['prefill_tokens']:,} prefill tok "
                  f"· ~${s.est_cost_usd or 0:.3f} discount · ~${s.forecast['usd_list'] or 0:.3f} list")
            return 0
        print(f"{s.base_run_id} → {s.ckpt_run_id}: {s.n_rows} rows · "
              f"mean gap {s.mean_sum_nats:+.1f} nats · diverged {s.n_diverged} · "
              f"p_skip {s.mean_p_skip_base:.3f} → {s.mean_p_skip_ckpt:.3f} · ~${s.est_cost_usd or 0:.4f}")
        return 0
    if args.cmd == "effort":
        store = _store(args)
        e_low, e_high = (float(x) for x in args.pair.split(","))
        s = effort_gap(store, _run_id(store, args.run), e_low, e_high, forecast=args.forecast)
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
