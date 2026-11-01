"""`flipbook` CLI spine
"""


import argparse
import asyncio
import functools
import json
import os
import sys
import textwrap
from pathlib import Path

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
    p.add_argument("--store", default=os.environ.get("FLIPBOOK_STORE", "./flipbook_store"),
                   help="store dir; env FLIPBOOK_STORE")


def _completion_store(ns) -> Store:
    return Store(getattr(ns, "store", None)
                 or os.environ.get("FLIPBOOK_STORE", "./flipbook_store"))


def _complete_run_ref(prefix, parsed_args, **kw):
    """Every form resolve_run accepts: ids, labels, study/label."""
    try:
        refs = []
        for r in _completion_store(parsed_args).runs():
            refs.append(r["run_id"])
            if r.get("label"):
                refs.append(r["label"])
                if r.get("study"):
                    refs.append(f"{r['study']}/{r['label']}")
        return refs
    except Exception:  # noqa: BLE001 - a completer must never break the shell
        return []


def _complete_manifest(prefix, parsed_args, **kw):
    try:
        return [d["name"] for d in _completion_store(parsed_args).manifests()]
    except Exception:  # noqa: BLE001 - a completer must never break the shell
        return []


def _complete_study(prefix, parsed_args, **kw):
    try:
        return sorted({r.get("study") for r in _completion_store(parsed_args).runs()} - {None})
    except Exception:  # noqa: BLE001 - a completer must never break the shell
        return []


def _complete_row(prefix, parsed_args, **kw):
    """Row ids for the run already typed; empty until it resolves."""
    try:
        store = _completion_store(parsed_args)
        rec = store.resolve_run(parsed_args.run)
        return [r["row_id"] for r in store.manifest_rows(rec["manifest_hash"])]
    except Exception:  # noqa: BLE001 - a completer must never break the shell
        return []


def _complete_grader(prefix, parsed_args, **kw):
    from flipbook.graders import GRADERS
    return sorted(GRADERS)


def _add_eval_args(p: argparse.ArgumentParser, required: bool = True) -> None:
    # lint --run reads model/manifest off the stored run record instead
    p.add_argument("--model", required=required,
                   help="'thinkingmachines/Inkling-Small', or 'tinker://<run>/sampler_weights/<ckpt>'")
    p.add_argument("--manifest", required=required, help="manifest name or hash"
                   ).completer = _complete_manifest
    p.add_argument("--effort", type=float, default=None,
                   help="reasoning effort 0..1 (required by Inkling)")
    p.add_argument("--k", type=int, default=4, help="samples per question")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--renderer", default=None, help="renderer override, e.g. tml_v0")
    p.add_argument("--base-model", default=None, help="offline override for tinker:// resolution")
    p.add_argument("--label", default=None, help="short name for this run, e.g. 'baseline' or 'step56'")
    p.add_argument("--study", default=None, help="grouping name - runs in a study chart together")


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


# keys a --config TOML may set (== the eval flag dests)
_EVAL_FILE_KEYS = {
    "model", "manifest", "effort", "k", "temperature", "max_tokens",
    "seed", "renderer", "base_model", "label", "study",
}


def _load_eval_file(path: str) -> dict:
    import tomllib

    p = Path(path)
    if not p.exists():
        raise SystemExit(f"no config file at {path}")
    try:
        cfg = tomllib.loads(p.read_text())
    except tomllib.TOMLDecodeError as e:
        raise SystemExit(f"{path}: bad TOML: {e}") from e
    bad = set(cfg) - _EVAL_FILE_KEYS
    if bad:
        raise SystemExit(f"{path}: unknown keys {sorted(bad)} - allowed: {sorted(_EVAL_FILE_KEYS)}")
    return cfg


def _merge_eval_config(args, argv: list[str]) -> None:
    """File values are defaults; an explicit --flag always wins."""
    if args.config:
        cfg = _load_eval_file(args.config)
        for k, v in cfg.items():
            flag = "--" + k.replace("_", "-")
            if not any(a == flag or a.startswith(flag + "=") for a in argv):
                setattr(args, k, v)
    if not args.model or not args.manifest:
        raise SystemExit("eval needs --model and --manifest (flags or config keys)")


def _write_eval_config(args, path: str) -> None:
    lines = [
        "# flipbook eval preset - `flipbook eval --config <this file>`",
    ]
    for k in sorted(_EVAL_FILE_KEYS):
        v = getattr(args, k, None)
        if v is None:
            continue
        lines.append(f"{k} = {json.dumps(v) if isinstance(v, str) else v}")
    Path(path).write_text("\n".join(lines) + "\n")


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


class _Fmt(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


_GUIDE = """\
workflow
  1. freeze    build a manifest: frozen questions + golds + grader_ids
  2. eval      sample a model over it (a spend forecast prints first)
  3. compare   paired stats and flips between two runs
  4. diverge   per-token logprob gap of a checkpoint on base traces
  5. serve     read-only API + browser UI over the store

walkthrough
  flipbook freeze --benchmark aime2026:30 --name aime30
  flipbook eval --model thinkingmachines/Inkling-Small \\
      --manifest aime30 --effort 0.5 --label baseline
  flipbook eval --model tinker://<run>/sampler_weights/final \\
      --manifest aime30 --effort 0.5 --study sft --label step56
  flipbook compare baseline sft/step56
  flipbook show sft/step56 <row_id>
  flipbook diverge --base baseline --ckpt sft/step56
  flipbook serve      # opens the GUI at localhost:8484

inspecting
  the GUI is a read-only view over the store directory - nothing lives
  in the server. kill it, move the store, serve it again: same results.
  the one paid action (branch probe) needs TINKER_API_KEY; serve loads
  ~/.secrets/tinker.env if it exists.

during training
  flipbook track <cookbook run dir> --manifest aime30 --study sft
  evaluates every checkpoint under a cookbook log dir and groups the runs
  into a study - that is what powers the study chart and 'the read'.

run refs
  anywhere a run is taken, any of these works:
    run_id          013ccbd2f51d...
    prefix          013ccb   (if unique)
    study/label     sft/step56
    label           baseline (if unique)

store
  ./flipbook_store by default; override per command with --store or
  set FLIPBOOK_STORE. everything is local parquet + json - inspectable
  with duckdb or pandas. 'flipbook manifests' lists what's frozen.

scripting
  manifests, runs, show, and compare take --json for machine-readable
  output. eval prints a spend forecast before the first paid call;
  --forecast prints it without running.

tab completion (argcomplete)
  eval "$(register-python-argcomplete flipbook)"   # bash, or zsh with bashcompinit
  run labels, manifest names, studies, grader ids, and row ids all
  complete on <TAB>.

migration helpers (rarely needed): import-evalstore, import-metrics
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="flipbook",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="the short version:\n  freeze -> eval -> compare -> serve\n\n"
               "'flipbook guide' prints a full walkthrough.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="cmd")
    Sub = functools.partial(sub.add_parser, formatter_class=_Fmt)
    sub.add_parser("guide", help="print a walkthrough of the whole workflow")
    p = Sub(
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
    p = Sub(
        "eval", help="run a config over a manifest",
        epilog="example: flipbook eval --model thinkingmachines/Inkling-Small "
               "--manifest aime30 --effort 0.5 --label baseline\n"
               "--config FILE reads the same settings from TOML (file = "
               "defaults, flags always override).\n"
               "a spend forecast prints before the first paid call; "
               "--forecast prints it and exits.",
    )
    # required=False on model/manifest: --config can supply them; validated post-merge
    _add_eval_args(p, required=False)
    p.add_argument("--config", default=None, metavar="FILE",
                   help="TOML of eval settings; explicit flags override")
    p.add_argument("--save-config", default=None, metavar="FILE",
                   help="write the resolved settings to TOML and exit")
    p.add_argument("--forecast", action="store_true")
    p.add_argument("--concurrency", type=int, default=8)
    _add_store(p)
    p = Sub(
        "demo",
        help="one-command tour: freeze a small manifest, eval the base model, then `flipbook serve`",
    )
    p.add_argument("--model", default="thinkingmachines/Inkling-Small")
    p.add_argument("--benchmark", default="math500:8", help="NAME[:n] to freeze as 'demo'")
    p.add_argument("--effort", type=float, default=0.9)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--forecast", action="store_true", help="print the spend estimate and stop")
    p.add_argument("--concurrency", type=int, default=8)
    _add_store(p)
    p = Sub("lint", help="check a config or stored run")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run").completer = _complete_run_ref
    src.add_argument("--config", action="store_true")
    _add_eval_args(p, required=False)
    _add_store(p)
    p = Sub("graders", help="list registered grader ids")
    p = Sub("grade", help="dry-run a grader on one text")
    p.add_argument("text")
    p.add_argument("--grader", required=True).completer = _complete_grader
    p.add_argument("--gold", required=True)
    p = Sub("manifests", help="list manifests in the store")
    p.add_argument("--json", action="store_true")
    _add_store(p)
    p = Sub(
        "status",
        help="store at a glance: studies, coverage gaps, and the commands that fill them",
    )
    _add_store(p)

    p = Sub("runs", aliases=["evals"], help="list eval runs in the store")
    p.add_argument("--study").completer = _complete_study
    p.add_argument("--json", action="store_true")
    _add_store(p)
    p = Sub("budget", help="token/cost distribution for a run", epilog=_RUN_REF_EPILOG)
    p.add_argument("run").completer = _complete_run_ref
    _add_store(p)
    p = Sub("show", help="inspect one row's samples in a run", epilog=_RUN_REF_EPILOG)
    p.add_argument("run", help="run id, prefix, study/label, or label").completer = _complete_run_ref
    p.add_argument("row", help="row id or unique prefix (of id or hash suffix)").completer = _complete_row
    p.add_argument("--sample", type=int, default=None, help="only this sample index")
    p.add_argument("--full", action="store_true", help="print full text, not the tail")
    p.add_argument("--thinking", action="store_true", help="decode thinking from token_ids")
    p.add_argument("--json", action="store_true")
    _add_store(p)
    p = Sub("compare", help="paired stats between two runs", epilog=_RUN_REF_EPILOG)
    p.add_argument("run_a").completer = _complete_run_ref
    p.add_argument("run_b").completer = _complete_run_ref
    p.add_argument("--gate", action="store_true", help="exit 2 on a significant regression")
    p.add_argument("--max-regression", type=float, default=0.02)
    p.add_argument("--max-new-truncation-rate", type=float, default=0.0)
    p.add_argument("--json", action="store_true")
    p.add_argument("--markdown", action="store_true")
    _add_store(p)
    p = Sub("diverge", help="per-token logprob gap of ckpt vs base on base's traces", epilog=_RUN_REF_EPILOG)
    p.add_argument("--base", required=True).completer = _complete_run_ref
    p.add_argument("--ckpt", required=True).completer = _complete_run_ref
    p.add_argument("--rows", choices=["flips", "all"], default="all")
    p.add_argument("--forecast", action="store_true")
    _add_store(p)
    p = Sub("effort", help="effort-prefix gap for one run's traces", epilog=_RUN_REF_EPILOG)
    p.add_argument("--run", required=True).completer = _complete_run_ref
    p.add_argument("--pair", required=True, help="e_low,e_high")
    p.add_argument("--forecast", action="store_true")
    _add_store(p)
    p = Sub("track", help="evaluate every checkpoint in a cookbook log dir")
    p.add_argument("log_dir")
    p.add_argument("--manifest", required=True).completer = _complete_manifest
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
    p = Sub("import-evalstore", help="import a legacy eval bundle into the store")
    p.add_argument("path")
    _add_store(p)
    p = Sub("import-metrics", help="import metrics.jsonl from a cookbook log dir")
    p.add_argument("log_dir")
    p.add_argument("--study", required=True)
    _add_store(p)
    p = Sub("serve", help="read-only API + GUI over the store")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8484)
    p.add_argument("--no-open", action="store_true", help="don't open the browser")
    p.add_argument(
        "--env-file", default="~/.secrets/tinker.env",
        help="dotenv loaded if present (TINKER_API_KEY for the branch probe)",
    )
    _add_store(p)
    # keep migration helpers out of the top-level list; they still run
    sub._choices_actions = [
        a for a in sub._choices_actions
        if a.dest not in ("import-evalstore", "import-metrics")
    ]
    if argv is None:
        argv = sys.argv[1:]
    # completion runs with argv empty - the typed line lives in COMP_LINE,
    # so autocomplete must come before the bare-invocation help
    import argcomplete  # lazy: only the completion path pays the import
    argcomplete.autocomplete(ap)
    if not argv:
        ap.print_help()
        return 2
    args = ap.parse_args(argv)
    if args.cmd == "guide":
        print(_GUIDE)
        return 0
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
    if args.cmd == "demo":
        store = _store(args)
        if store.manifest_doc("demo") is None:
            bench, n = args.benchmark.rsplit(":", 1)
            m = Manifest.freeze({bench: int(n)}, seed=0, name="demo")
            store.put_manifest(m.to_doc(), m.rows)
            print(f"froze demo: {len(m.rows)} rows from {bench}")
        # reuse the eval pipeline verbatim - demo is just defaults around it
        args = argparse.Namespace(
            cmd="demo",
            model=args.model, manifest="demo", effort=args.effort, k=args.k,
            temperature=0.7, max_tokens=32768, seed=0, renderer=None,
            base_model=None, label="demo", study="demo",
            forecast=args.forecast, concurrency=args.concurrency,
            config=None, save_config=None, store=args.store,
        )
    if args.cmd in ("eval", "demo"):
        if args.cmd == "eval":
            _merge_eval_config(args, argv if argv is not None else sys.argv[1:])
        if args.save_config:
            _write_eval_config(args, args.save_config)
            print(f"wrote {args.save_config} - run it with `flipbook eval --config {args.save_config}`")
            return 0
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
        if args.cmd == "demo":
            print("inspect it: flipbook serve  ·  list runs: flipbook evals")
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
    if args.cmd == "manifests":
        store = _store(args)
        docs = store.manifests()
        if args.json:
            print(json.dumps(docs, indent=2, default=str))
            return 0
        print(f"{'name':<16}  {'hash':<12} {'rows':>5}  {'created'}")
        for d in docs:
            print(f"{d['name']:<16}  {d['manifest_hash'][:12]}  "
                  f"{d.get('n_rows', len(d.get('rows', []))):>5}  "
                  f"{str(d.get('created_at', '-'))[:10]}")
        return 0
    if args.cmd == "status":
        store = _store(args)
        runs = store.runs()
        mnames = {d["manifest_hash"]: d["name"] for d in store.manifests()}
        print(f"{len(mnames)} manifests · {len(runs)} eval runs in {store.path}")
        studies: dict[str, list[dict]] = {}
        for r in runs:
            studies.setdefault(r.get("study") or "-", []).append(r)
        for name, rs in sorted(studies.items()):
            ckpts = sorted(
                (
                    r
                    for r in rs
                    if (r.get("provenance") or {}).get("train_step_measured") is not None
                ),
                key=lambda r: r["provenance"]["train_step_measured"],
            )
            base = next(
                (r for r in rs if (r.get("provenance") or {}).get("train_step_measured") is None),
                None,
            )
            mans = {mnames.get(r.get("manifest_hash"), str(r.get("manifest_hash"))[:12]) for r in rs}
            print(f"\n{name} - {len(rs)} runs on {', '.join(sorted(mans))}")
            if not ckpts:
                continue
            steps = [r["provenance"]["train_step_measured"] for r in ckpts]
            print(f"  checkpoints  {len(ckpts)} (steps {steps[0]}..{steps[-1]})")
            divdir, effdir = store.path / "divergence", store.path / "effort"
            have_div = {r["run_id"] for r in ckpts
                        if any(divdir.glob(f"*__{r['run_id']}.parquet"))}
            have_eff = {r["run_id"] for r in ckpts
                        if any(effdir.glob(f"{r['run_id']}__*.parquet"))}
            # reuse the pair this study already uses, e.g. run__0.2_0.9.parquet
            pair = next(
                (f.stem.split("__")[1] for r in ckpts
                 for f in effdir.glob(f"{r['run_id']}__*.parquet")),
                "LOW,HIGH",
            ).replace("_", ",")
            for kind, have, hint in (
                ("divergence", have_div, "diverge --base {b} --ckpt {c}"),
                ("effort gap", have_eff, "effort --run {c} --pair " + pair),
            ):
                miss = [r for r in ckpts if r["run_id"] not in have]
                if not miss:
                    print(f"  {kind:<12} {len(have)}/{len(ckpts)} checkpoints")
                    continue
                msteps = ", ".join(str(r["provenance"]["train_step_measured"]) for r in miss)
                print(f"  {kind:<12} {len(have)}/{len(ckpts)} checkpoints  missing: step {msteps}")
                c = f"{name}/{miss[0].get('label') or miss[0]['run_id']}"
                b = f"{name}/{base.get('label') or base['run_id']}" if base else "BASE"
                print(f"    fill: flipbook {hint.format(b=b, c=c)}")
        return 0
    if args.cmd in ("runs", "evals"):
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
        from flipbook.api import load_env_file, serve
        loaded = load_env_file(Path(args.env_file).expanduser())
        if loaded:
            print(f"loaded {', '.join(loaded)} from {args.env_file}", flush=True)
        print(f"serving {args.store} at http://{args.host}:{args.port}", flush=True)
        serve(args.store, host=args.host, port=args.port, open_browser=not args.no_open)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
