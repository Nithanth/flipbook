"""`flipbook` CLI spine
"""


import argparse
import asyncio
import functools
import json
import os
import re
import sys
import textwrap
from pathlib import Path

import pyarrow.parquet as pq

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
from flipbook.synthetic import MANIFEST_NAME, seed_synthetic


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


def _model_names() -> list[str]:
    """the cookbook's known registry, flattened to org/name - completion and
    `flipbook models` both read it, so they can never disagree"""
    from tinker_cookbook import model_info

    orgs = {
        "meta-llama": model_info.get_llama_info, "Qwen": model_info.get_qwen_info,
        "deepseek-ai": model_info.get_deepseek_info, "openai": model_info.get_gpt_oss_info,
        "moonshotai": model_info.get_moonshot_info, "zai-org": model_info.get_zai_info,
        "nvidia": model_info.get_nvidia_info,
        "thinkingmachines": model_info.get_thinkingmachines_info,
    }
    return [f"{org}/{k}" for org, fn in orgs.items() for k in fn()]


def _complete_grader(prefix, parsed_args, **kw):
    from flipbook.graders import GRADERS
    return sorted(GRADERS)


def _add_eval_args(p: argparse.ArgumentParser, required: bool = True) -> None:
    # lint --run reads model/manifest off the stored run record instead;
    # eval accepts a positional ref so both are optional there (a run ref
    # supplies model + manifest + sampling config on its own)
    p.add_argument("--model", required=required,
                   help="'thinkingmachines/Inkling-Small', or 'tinker://<run>/sampler_weights/<ckpt>'; "
                        "registered names: flipbook models"
                   ).completer = lambda prefix, **kw: [n for n in _model_names() if n.startswith(prefix)]
    p.add_argument("--manifest", required=required, help="manifest name or hash"
                   ).completer = _complete_manifest
    p.add_argument("--effort", type=float, default=None,
                   help="reasoning effort 0..1 (required by Inkling)")
    p.add_argument("--k", type=int, default=4, help="samples per question")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="sampling temperature - keep it identical across a study's runs")
    p.add_argument("--max-tokens", type=int, default=32768,
                   help="generation cap; thinking tokens count toward it")
    p.add_argument("--seed", type=int, default=0, help="sampling seed")
    p.add_argument("--renderer", default=None, help="renderer override, e.g. tml_v0")
    p.add_argument("--base-model", default=None, help="offline override for tinker:// resolution")
    p.add_argument("--label", default=None, help="short name for this run, e.g. 'baseline' or 'step56'")
    p.add_argument("--study", default=None, help="grouping name - runs in a study chart together")


async def _resolve(args) -> tuple[str, str]:
    """→ (base_model, renderer). Creates a service client only for tinker://."""
    from tinker_cookbook.exceptions import ConfigurationError
    sc = None
    if args.model.startswith("tinker://") and args.base_model is None:
        import tinker
        sc = tinker.ServiceClient()
    try:
        r = await resolve_model(
            args.model, renderer=args.renderer, base_model=args.base_model, service_client=sc
        )
    except (ConfigurationError, KeyError, ValueError) as e:
        import difflib
        near = difflib.get_close_matches(args.model, _model_names(), n=3, cutoff=0.5)
        hint = f" - did you mean {near[0]}?" if near else " - `flipbook models` lists the registry"
        raise SystemExit(f"unknown model {args.model!r}{hint}") from e
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


def _flagged(argv: list[str], dest: str) -> bool:
    flag = "--" + dest.replace("_", "-")
    return any(a == flag or a.startswith(flag + "=") for a in argv)


def _merge_eval_config(args, argv: list[str]) -> None:
    """File values are defaults; an explicit --flag always wins."""
    if args.config:
        cfg = _load_eval_file(args.config)
        for k, v in cfg.items():
            if not _flagged(argv, k):
                setattr(args, k, v)


def _resolve_eval_ref(args, store: Store, argv: list[str]) -> None:
    """A positional ref points at weights through the store, not a URI: a run
    ref reuses that run's checkpoint (or base model) and inherits its config
    for flags not explicitly set; 'study/stepN' and 'study/final' synthesize
    the checkpoint path off the study's recorded sampler paths."""
    if getattr(args, "ref", None) is None:
        return
    ref = args.ref
    rec = None
    try:
        rec = store.resolve_run(ref)
    except LookupError as e:
        if "ambiguous" in str(e).lower():
            raise SystemExit(str(e)) from e
    if rec is None:
        m = re.fullmatch(r"([^/]+)/(step\d+|final)", ref)
        ck = m and next(
            (r for r in store.runs()
             if r.get("study") == m.group(1)
             and "sampler_weights" in (r.get("checkpoint_path") or "")),
            None,
        )
        if ck:
            step = m.group(2)
            suffix = f"{int(step[4:]):06d}" if step.startswith("step") else step
            rec = dict(ck)
            rec["checkpoint_path"] = re.sub(
                r"/sampler_weights/[^/]+$", f"/sampler_weights/{suffix}",
                ck["checkpoint_path"],
            )
    if rec is None:
        args.model = args.model or ref
        return
    args.model = args.model or rec["checkpoint_path"] or rec["model_id"]
    if args.label is None:
        m = re.fullmatch(r"[^/]+/(step\d+|final)", ref)
        args.label = m.group(1) if m else None
    inherit = {
        "manifest": rec.get("manifest_hash"), "effort": rec.get("effort"),
        "temperature": rec.get("temperature"), "max_tokens": rec.get("max_tokens"),
        "k": rec.get("k"), "seed": rec.get("seed"), "renderer": rec.get("renderer"),
        "study": rec.get("study"),
    }
    for k, v in inherit.items():
        if v is not None and not _flagged(argv, k):
            setattr(args, k, v)


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

_TINKER_ENV = Path.home() / ".secrets" / "tinker.env"


def _load_tinker_env() -> None:
    """Autoload the tinker dotenv for paid commands; an already-set
    TINKER_API_KEY always wins over the file."""
    if os.environ.get("TINKER_API_KEY"):
        return
    from flipbook.api import load_env_file  # lazy: api pulls in fastapi

    load_env_file(_TINKER_ENV)


def _doctor(args) -> int:
    """First-run checklist: what works, what's missing, and the fix."""
    have_key = bool(os.environ.get("TINKER_API_KEY"))
    rows = Path(args.store) / "runs"
    n_runs = len(list(rows.glob("*.json"))) if rows.exists() else 0
    dist = Path(__file__).resolve().parent.parent.parent / "web" / "dist"
    checks = [
        ("TINKER_API_KEY", have_key or _TINKER_ENV.exists(),
         "in environment" if have_key
         else f"autoloaded from {_TINKER_ENV} by paid commands" if _TINKER_ENV.exists()
         else f"missing - put it in {_TINKER_ENV}"),
        ("store", n_runs > 0,
         f"{args.store} · {n_runs} runs" if n_runs
         else "empty - `flipbook demo --synthetic` seeds it, no key needed"),
        ("gui build", dist.exists(),
         str(dist) if dist.exists()
         else "missing - `cd web && npm install && npm run build` (`serve` still gives the api)"),
    ]
    print("flipbook doctor")
    for name, good, note in checks:
        print(f"  {'ok ' if good else '!! '} {name:<15} {note}")
    print("  ·  tab completion    eval \"$(register-python-argcomplete flipbook)\"")
    return 0 if all(g for _, g, _ in checks) else 1


def _run_cost(store: Store, r: dict) -> float:
    """est_cost_usd from provenance when present, else summed from samples."""
    prov = r.get("provenance") or {}
    if prov.get("est_cost_usd") is not None:
        return prov["est_cost_usd"]
    col = store.samples(r["run_id"]).column("est_cost_usd")
    return sum(v or 0 for v in col.to_pylist())


def _report(store: Store, study: str, as_json: bool = False) -> int:
    """'The read' as a terminal summary: the pass@1 arc, whether divergence
    moved first, response-length collapse, and total cost."""
    runs = [r for r in store.study_runs(study)]
    # a run still sampling is not a point
    runs = [r for r in runs if r.get("state") != "running"]
    if not runs:
        raise SystemExit(f"no runs in study {study!r} - studies: "
                         f"{sorted({r.get('study') for r in store.runs()})}")
    def step(r):
        return (r.get("train_step")
                or (r.get("provenance") or {}).get("train_step_measured"))

    # checkpoints name their baseline in provenance; it may live in another
    # study (shared base). a step-less run with a checkpoint_path is an
    # unmeasured final, not a baseline - prefer it only as a last resort
    by_id = {r["run_id"]: r for r in store.runs()}
    refs = {(r.get("provenance") or {}).get("baseline_run_id") for r in runs} - {None}
    base = (next((by_id[i] for i in sorted(refs) if i in by_id), None)
            or next((r for r in runs
                     if step(r) is None and not r.get("checkpoint_path")), None)
            or next((r for r in runs if step(r) is None), None))
    ckpts = sorted((r for r in runs if step(r) is not None), key=step)
    if base is None or not ckpts:
        raise SystemExit(
            f"study {study!r} has no baseline->checkpoint arc to narrate "
            f"({len(runs)} runs)")

    _p1 = {}
    def pass1(r):
        """provenance pass1 when present, else computed from samples (runs
        recorded before the lifecycle fields existed)."""
        rid = r["run_id"]
        if rid not in _p1:
            p = (r.get("provenance") or {}).get("pass1")
            if p is None:
                s = store.samples(rid).to_pylist()
                ok = [x for x in s if x["verdict"] is not None]
                p = sum(x["verdict"] for x in ok) / len(ok) if ok else 0.0
            _p1[rid] = p
        return _p1[rid]

    bullets: list[str] = []
    arc = [f"{pass1(base):.2f}"] + [f"{pass1(r):.2f} (step {step(r)})" for r in ckpts]

    tm = store.training_metrics(study).to_pylist()
    def _mseries(key):
        return sorted((m["step"], m["value"]) for m in tm if m["key"] == key)

    reward = _mseries("env/all/reward/total")
    rtok = _mseries("env/all/ac_tokens_per_turn")
    if reward:
        tok = (f" · rollout tokens/turn {rtok[0][1]:.0f} -> {rtok[-1][1]:.0f}"
               if rtok else "")
        reward_line = f"{reward[0][1]:.2f} -> {reward[-1][1]:.2f}{tok}"
    else:
        reward_line = None

    # mean divergence + p_skip per checkpoint from its parquet, if computed
    divdir = store.path / "divergence"
    div = {}
    for r in ckpts:
        fs = list(divdir.glob(f"*__{r['run_id']}.parquet"))
        if fs:
            t = pq.read_table(fs[0]).to_pylist()
            div[r["run_id"]] = {
                "nats": sum(d["sum_nats"] for d in t) / len(t),
                "p_skip": sum(d["p_skip_ckpt"] for d in t) / len(t),
            }

    last, first_ck = ckpts[-1], ckpts[0]
    trough = min(ckpts, key=lambda r: pass1(r))
    # a moved point estimate only counts when its paired CI excludes zero;
    # unpaired runs get the old point-estimate rule
    pair = compare(store, base["run_id"], last["run_id"])
    if pair.n_pairs:
        collapsed = pair.delta_ci[1] < 0
        improved = pair.delta_ci[0] > 0
    else:
        collapsed = pass1(base) - pass1(last) > 0.05
        improved = pass1(last) - pass1(first_ck) > 0.05
    reward_rose = bool(reward) and reward[-1][1] - reward[0][1] > 0.1
    verdict = ("reward up, held-out eval down" if collapsed and reward_rose
               else "the fine-tune regressed this eval" if collapsed
               else "the fine-tune improved this eval" if improved
               else "no eval-visible change")
    if reward:
        rf, rl = reward[0][1], reward[-1][1]
        if rf >= 0.95:
            bullets.append(f"reward saturated from the first batch ({rf:.2f}): "
                  "this env gives the policy no gradient signal")
        elif reward_rose and collapsed:
            bullets.append(f"rollout reward rose {rf:.2f} -> {rl:.2f} while held-out "
                  "pass@1 fell: the policy is fitting the reward, not the capability")
        elif reward_rose and improved:
            bullets.append(f"reward and held-out pass@1 rose together ({rf:.2f} -> {rl:.2f}, "
                  f"{pass1(base):.2f} -> {pass1(last):.2f})")
        elif reward_rose:
            bullets.append(f"rollout reward rose {rf:.2f} -> {rl:.2f}; held-out pass@1 "
                  "unchanged: no transfer to the eval either way")
        else:
            bullets.append(f"rollout reward flat ({rf:.2f} -> {rl:.2f})")
    if rtok and abs(rtok[-1][1] - rtok[0][1]) / rtok[0][1] > 0.2:
        bullets.append(f"rollouts {'shortened' if rtok[-1][1] < rtok[0][1] else 'lengthened'} "
              f"{rtok[0][1]:.0f} -> {rtok[-1][1]:.0f} tokens per turn on the "
              "training distribution")
    if not collapsed and not improved and pair.n_pairs and abs(pair.delta) >= 0.05:
        bullets.append(f"pass@1 {pass1(base):.2f} -> {pass1(last):.2f} by step {step(last)}, "
              f"but the CI [{pair.delta_ci[0]:+.2f}, {pair.delta_ci[1]:+.2f}] includes zero: "
              f"within noise at n={pair.n_pairs}")
    if collapsed:
        bullets.append(f"pass@1 {pass1(base):.2f} -> {pass1(trough):.2f} by step {step(trough)}"
              + (f", ending {pass1(last):.2f} (step {step(last)})" if last is not trough else ""))
    if collapsed:
        early = [r for r in ckpts if r["run_id"] in div and div[r["run_id"]]["nats"] < -50]
        cliff = next((r for r in ckpts if pass1(r) < pass1(base) - 0.05), None)
        if early and cliff and step(early[0]) <= step(cliff):
            d = div[early[0]["run_id"]]
            bullets.append(f"divergence moved early: {d['nats']:.0f} nats, p_skip {d['p_skip']:.0%} "
                  f"at step {step(early[0])}")
    gen = {}
    for r in (first_ck, last):
        s = store.samples(r["run_id"]).to_pylist()
        gen[r["run_id"]] = sum(x["gen_tokens"] for x in s) / len(s) if s else 0
    if (gen[first_ck["run_id"]] and gen[last["run_id"]]
            and gen[last["run_id"]] / gen[first_ck["run_id"]] < 0.4):
        bullets.append(f"responses shrank {gen[first_ck['run_id']]:.0f} -> {gen[last['run_id']]:.0f} tokens "
              f"({gen[first_ck['run_id']] / gen[last['run_id']]:.0f}x shorter) - it stopped reasoning, "
              "not just got answers wrong")
    nll = [m for m in tm if m["key"] == "train/nll"]
    if len(nll) >= 2 and collapsed:
        bullets.append(f"train/nll kept falling {nll[0]['value']:.2f} -> {nll[-1]['value']:.2f} "
              "while eval fell - the loss never saw it")
    spend = sum(_run_cost(store, r) for r in runs)
    def _ref(r):
        # shared baselines live under their own study, not this one
        return f"{r['study']}/{r['label']}" if r.get("study") and r.get("label") else r["run_id"]
    compare_ref = f"{_ref(base)} {_ref(last)}"
    if as_json:
        # the same read as a payload: verdict + evidence, for a notifier or a bot
        print(json.dumps({
            "study": study, "verdict": verdict, "bullets": bullets,
            "pass1": [{"step": 0, "value": pass1(base), "run_id": base["run_id"]}]
                     + [{"step": step(r), "value": pass1(r), "run_id": r["run_id"]} for r in ckpts],
            "delta": pair.delta if pair.n_pairs else None,
            "delta_ci": list(pair.delta_ci) if pair.n_pairs else None,
            "reward": reward_line, "cost_usd": round(spend, 2), "n_runs": len(runs),
            "compare": f"flipbook compare {compare_ref}",
            "inspect": f"#/study?name={study}",
        }, indent=2))
        return 0
    print(f"report: {study}")
    print(f"  pass@1: {' -> '.join(arc)}")
    if reward_line:
        print(f"  reward: {reward_line}")
    print(f"\n  the read: {verdict}")
    for t in bullets:
        print(f"    {t}")
    print(f"\n  cost: ~${spend:.2f} across {len(runs)} runs")
    print(f"  compare: flipbook compare {compare_ref}")
    print(f"  inspect: flipbook serve  ->  #/study?name={study}")
    return 0


class _Fmt(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


_GUIDE = """\
workflow
  1. freeze    build a manifest: frozen questions + golds + grader_ids
  2. eval      sample a model over it (a spend forecast prints first);
               `flipbook eval sft/final` resolves a study checkpoint ref
  +. train     `flipbook train sft|rl` = example recipes with in-loop
               evals; for your own loop add `RegressionEvaluator`
               (flipbook.evaluator) to evaluator_builders, or
               `flipbook track <log dir>` a finished run
  3. compare   paired stats and flips between two runs
  4. diverge   per-token logprob gap of a checkpoint on base traces
  5. serve     read-only API + browser UI over the store

no key? no spend?
  flipbook demo --synthetic seeds a fabricated collapse arc into the
  store - every command below works on it immediately.

walkthrough
  flipbook freeze --benchmark aime2026:30 --name aime30
  flipbook eval --model thinkingmachines/Inkling-Small \\
      --manifest aime30 --effort 0.5 --label baseline
  flipbook eval sft/final          # terse form: ref resolves the checkpoint
                                   # path and inherits manifest/effort/k/...
  flipbook compare sft/baseline sft/step56
  flipbook show sft/step56 <row_id>
  flipbook diverge --base sft/baseline --ckpt sft/step56
  flipbook serve      # opens the GUI at localhost:8484

inspecting
  the GUI is a read-only view over the store directory - nothing lives
  in the server. kill it, move the store, serve it again: same results.
  anything that samples needs TINKER_API_KEY; paid commands autoload
  ~/.secrets/tinker.env when the env var isn't already set.
  `flipbook doctor` checks the first-run setup end to end.

during training
  flipbook track <cookbook run dir> --manifest aime30 --study sft
  evaluates every checkpoint under a cookbook log dir and groups the runs
  into a study - that is what powers the study chart and 'the read'.
  `flipbook import-rollouts <log dir> --study s` pulls the raw rollout
  metrics and transcript sample into the store (offline, no sampling).

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
    # required=False on model/manifest: --config or a positional ref supplies
    # them; validated post-merge
    _add_eval_args(p, required=False)
    p.add_argument("ref", nargs="?",
                   help="model name, tinker:// path, or run ref (run_id prefix, "
                        "'study/label', 'study/stepN', 'study/final'); a run ref "
                        "inherits its config for flags not explicitly set"
                   ).completer = _complete_run_ref
    p.add_argument("--config", default=None, metavar="FILE",
                   help="TOML of eval settings; explicit flags override")
    p.add_argument("--save-config", default=None, metavar="FILE",
                   help="write the resolved settings to TOML and exit")
    p.add_argument("--forecast", action="store_true",
                   help="print the spend estimate and exit without sampling")
    p.add_argument("--concurrency", type=int, default=8,
                   help="max parallel sampling calls")
    _add_store(p)
    p = Sub(
        "demo",
        help="one-command tour: freeze a small manifest, eval the base model, then `flipbook serve`",
    )
    p.add_argument("--synthetic", action="store_true",
                   help="seed a fabricated collapse story instead - no key, no spend")
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
    p = Sub("models", help="list models the tinker registry knows")
    p.add_argument("--json", action="store_true")
    p = Sub("manifests", help="list manifests in the store")
    p.add_argument("--json", action="store_true")
    _add_store(p)
    p = Sub(
        "status",
        help="store at a glance: studies, coverage gaps, and the commands that fill them",
    )
    _add_store(p)
    p = Sub("report", help="'the read' for a study as a terminal summary")
    p.add_argument("study").completer = _complete_study
    p.add_argument("--json", action="store_true", help="verdict + bullets as a payload")
    _add_store(p)

    p = Sub("runs", aliases=["evals"], help="list eval runs in the store")
    p.add_argument("--study").completer = _complete_study
    p.add_argument("--json", action="store_true")
    _add_store(p)
    p = Sub("budget", help="token/cost distribution for a run", epilog=_RUN_REF_EPILOG)
    p.add_argument("run").completer = _complete_run_ref
    p.add_argument("--json", action="store_true")
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
    p = Sub("import-rollouts",
            help="import rollout metrics + transcripts from a cookbook RL log dir")
    p.add_argument("log_dir")
    p.add_argument("--study", required=True)
    _add_store(p)
    p = Sub("train",
            help="run a training recipe with in-loop checkpoint evals. "
                 "`train sft data` builds the jsonl, `train sft train --study S --manifest M` "
                 "runs SFT (scripts/train_demo.py); `train rl --env gsm8k --study S --manifest M` "
                 "runs RL (scripts/train_rl.py). `--help` after either lists knobs")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p = Sub("doctor", help="first-run checklist: secrets, store, gui build, completion")
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
    if args.cmd in ("eval", "demo", "track", "diverge", "effort"):
        _load_tinker_env()
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
        if args.synthetic:
            study = seed_synthetic(store)
            print(f"seeded '{study}': 4 runs on {MANIFEST_NAME} - a fabricated SFT")
            print("collapse (baseline ~0.92 -> step24 ~0.02 -> final ~0.05) with")
            print("samples, -1800-nat divergence traces, and a p_skip spike.")
            print("\nnumbers are invented - they demo the workflow, not a real model.")
            print("\nnext:")
            print("  flipbook status                                   # coverage gaps")
            print(f"  flipbook report {study:<33} # the read")
            print(f"  flipbook compare {study}/baseline {study}/step24   # the cliff")
            print("  flipbook serve                                    # inspect in the GUI")
            return 0
        print("demo: freeze a manifest -> eval the base model -> compare -> serve")
        if store.manifest_doc("demo") is None:
            bench, n = args.benchmark.rsplit(":", 1)
            m = Manifest.freeze({bench: int(n)}, seed=0, name="demo")
            store.put_manifest(m.to_doc(), m.rows)
            print(f"1/2 froze 'demo': {len(m.rows)} rows from {bench}")
        else:
            print("1/2 'demo' manifest already frozen")
        print(f"2/2 evaluating {args.model} (effort {args.effort}, k={args.k}) ...")
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
        store = _store(args)
        if args.cmd == "eval":
            real_argv = argv if argv is not None else sys.argv[1:]
            _merge_eval_config(args, real_argv)
            _resolve_eval_ref(args, store, real_argv)
            if not args.model or not args.manifest:
                raise SystemExit(
                    "eval needs weights and a manifest - pass a ref "
                    "(`flipbook eval sft_ink/step64`), --model + --manifest, or --config"
                )
        if args.save_config:
            _write_eval_config(args, args.save_config)
            print(f"wrote {args.save_config} - run it with `flipbook eval --config {args.save_config}`")
            return 0
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
            # one eval can't show a regression - suggest a second point at a
            # different effort so `compare` has a pair (rerun inherits config)
            print("\nto compare two points, eval the same config at a different effort:")
            print("  flipbook eval demo/demo --effort 0.1 --label effort0.1")
            print("  flipbook compare demo/demo demo/effort0.1")
            print("\ninspect everything: flipbook serve")
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
    if args.cmd == "models":
        from tinker_cookbook import model_info
        out = []
        for n in _model_names():
            a = model_info.get_model_attributes(n)
            rends = model_info.get_recommended_renderer_names(n)
            out.append({"model": n, "renderer": rends[0] if rends else None,
                        "size": a.size_str})
        if args.json:
            print(json.dumps(out, indent=2))
        else:
            for r in out:
                print(f"{r['model']:<50} {r['renderer'] or '-':<14} {r['size'] or ''}")
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
    if args.cmd == "report":
        return _report(_store(args), args.study, as_json=args.json)
    if args.cmd == "status":
        store = _store(args)
        runs = store.runs()
        mnames = {d["manifest_hash"]: d["name"] for d in store.manifests()}
        spend = sum(_run_cost(store, r) for r in runs)
        print(f"{len(mnames)} manifests · {len(runs)} eval runs in {store.path}"
              + (f" · ~${spend:.2f} total" if spend else ""))
        studies: dict[str, list[dict]] = {}
        for r in runs:
            studies.setdefault(r.get("study") or "-", []).append(r)
        def step(r):
            # a checkpoint's step: the ref-resolved train_step when recorded,
            # else the step the in-loop evaluator measured
            return (r.get("train_step") if r.get("train_step") is not None
                    else (r.get("provenance") or {}).get("train_step_measured"))
        for name, rs in sorted(studies.items()):
            if name != "-":
                rs = store.study_runs(name)
            ckpts = sorted((r for r in rs if step(r) is not None), key=step)
            # the divergence base is the untrained model: a run stamped with
            # neither an in-loop measured step nor a ref-resolved train_step
            base = next((r for r in rs if step(r) is None), None)
            mans = {mnames.get(r.get("manifest_hash"), str(r.get("manifest_hash"))[:12]) for r in rs}
            live = sum(1 for r in rs if r.get("state") == "running")
            sspend = sum(_run_cost(store, r) for r in rs)
            print(f"\n{name} - {len(rs)} runs on {', '.join(sorted(mans))}"
                  + (f" · ~${sspend:.2f}" if sspend else "")
                  + (f"  ({live} still sampling)" if live else ""))
            # a shared label makes 'study/label' refs ambiguous
            label_counts: dict[str, int] = {}
            for r in rs:
                if r.get("label"):
                    label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
            for lab in sorted(l for l, n in label_counts.items() if n > 1):
                print(f"  warning: duplicate label {lab!r} - use run_id refs")
            if not ckpts:
                continue
            steps = [step(r) for r in ckpts]
            print(f"  checkpoints  {len(ckpts)} (steps {steps[0]}..{steps[-1]})")
            divdir, effdir = store.path / "divergence", store.path / "effort"
            have_div = {r["run_id"] for r in ckpts
                        if any(divdir.glob(f"*__{r['run_id']}.parquet"))}
            have_eff = {r["run_id"] for r in ckpts
                        if any(effdir.glob(f"{r['run_id']}__*.parquet"))}
            # reuse the pair this study already uses, e.g. run__0.2_0.9.parquet;
            # the fallback is the RegressionEvaluator default pair
            pair = next(
                (f.stem.split("__")[1] for r in ckpts
                 for f in effdir.glob(f"{r['run_id']}__*.parquet")),
                "0.2_0.9",
            ).replace("_", ",")
            for kind, have, hint in (
                ("divergence", have_div, "diverge --base {b} --ckpt {c}"),
                ("effort gap", have_eff, "effort --run {c} --pair " + pair),
            ):
                miss = [
                    r for r in ckpts
                    if r["run_id"] not in have
                    # step-0 weights ARE the baseline: diverging them is a no-op
                    and not (kind == "divergence" and step(r) == 0)
                ]
                if not miss:
                    print(f"  {kind:<12} {len(have)}/{len(ckpts)} checkpoints")
                    continue
                msteps = ", ".join(str(step(r)) for r in miss)
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
        except LookupError:
            ids = [r["row_id"] for r in store.manifest_rows(rec["manifest_hash"])]
            raise SystemExit(f"no row matches {args.row!r} - rows in this "
                             f"manifest: {', '.join(ids[:8])}"
                             + (f" ... (+{len(ids)-8})" if len(ids) > 8 else ""))
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
        print(f"{b.n_samples} samples · {b.n_truncated} truncated · "
              f"gen tok p50 {b.gen_p50:.0f} / p95 {b.gen_p95:.0f} / max {b.gen_max} · "
              f"prompt {b.total_prompt_tokens:,} tok · gen {b.total_gen_tokens:,} tok"
              + (f" · ~${b.est_cost_usd:.2f}" if b.est_cost_usd is not None else ""))
        if args.json:
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
            ra, rb = store.resolve_run(pair.run_a), store.resolve_run(pair.run_b)
            ref = lambda r: f"{r.get('study')}/{r.get('label')}" if r.get("study") and r.get("label") else r["run_id"]
            print(f"{ref(ra)} ({pair.run_a}) acc={pair.acc_a:.3f}  vs  "
                  f"{ref(rb)} ({pair.run_b}) acc={pair.acc_b:.3f}")
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
            for n in sorted(pair.scoren, key=int):
                e = pair.scoren[n]
                print(f"  score@{n} {e['a']:.3f} → {e['b']:.3f}")
            for f in pair.flips:
                tag = "HARD " if f["hard"] else ""
                kind = kind_by_row.get(f["row_id"])
                tail = f"  b:{kind}" if kind else ""
                q = f"  {f['q']!r}" if f.get("q") else ""
                print(f"  {tag}{f['kind']}: {f['row_id']}  {f['p_a']:.2f} → {f['p_b']:.2f}{tail}{q}")
            for e in pair.excluded:
                print(f"  excluded {e['row_id']}: {e['reason']}")
            if not args.gate:
                print(f"\ninspect: flipbook serve  ->  #/compare?a={pair.run_a}&b={pair.run_b}")
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
    if args.cmd == "import-rollouts":
        from flipbook.import_rollouts import import_rollouts
        imp = import_rollouts(_store(args), args.log_dir, args.study)
        print(f"imported {imp.n_rollouts:,} rollouts ({imp.iterations} iterations · "
              f"splits: {', '.join(imp.splits)}) and {imp.n_texts:,} transcripts "
              f"into study {args.study}")
        cov = ", ".join(
            f"{sp} {lo:.0%}-{hi:.0%} of rollouts"
            for sp, (lo, hi) in sorted(imp.text_coverage.items()))
        print(f"transcripts: {cov} · a logged sample, not linked to rollout ids")
        return 0
    if args.cmd == "train":
        # `train rl ...` -> train_rl.py; anything else -> train_demo.py (sft)
        rest = list(args.rest)
        recipe = "train_rl.py" if rest[:1] == ["rl"] else "train_demo.py"
        if rest[:1] in (["rl"], ["sft"]):
            rest = rest[1:]
        script = Path(__file__).resolve().parent.parent.parent / "scripts" / recipe
        if not script.exists():
            raise SystemExit(f"training recipe not found at scripts/{recipe} - run from the repo root")
        import subprocess
        rc = subprocess.call([sys.executable, str(script), *rest])
        if rc == 0:
            print("\ninspect the trajectory: flipbook serve  ->  #/study")
        raise SystemExit(0 if rc == 3 else rc)  # 3 = recipe dry run, not a failure
    if args.cmd == "doctor":
        return _doctor(args)
    if args.cmd == "serve":
        from flipbook.api import load_env_file, serve
        loaded = load_env_file(Path(args.env_file).expanduser())
        if loaded:
            print(f"loaded {', '.join(loaded)} from {args.env_file}", flush=True)
        # probe first - uvicorn swallows bind errors into a bare ERROR log
        import socket
        probe = socket.socket()
        try:
            probe.bind((args.host, args.port))
        except OSError:
            probe.close()
            raise SystemExit(
                f"port {args.port} is busy - already serving at "
                f"http://{args.host}:{args.port} (or pass --port {args.port + 1})"
            ) from None
        probe.close()
        print(f"serving {args.store} at http://{args.host}:{args.port}", flush=True)
        serve(args.store, host=args.host, port=args.port, open_browser=not args.no_open)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
