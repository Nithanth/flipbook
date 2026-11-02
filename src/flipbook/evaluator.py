"""RegressionEvaluator: SamplingClientEvaluator the training loop calls.
Contract (tinker_cookbook.supervised.train.run_evals): called with a fresh
weight-snapshot client immediately before optim step `step`, so it measures
post-(step-1) weights. The baseline is evaluated once and reused via the
store.
"""


import json
import logging
import re
from pathlib import Path
from typing import Any

from tinker_cookbook.eval.evaluators import SamplingClientEvaluator

from flipbook.config import RunConfig, run_id
from flipbook.diverge import diverge_async
from flipbook.effort import effort_gap_async
from flipbook.manifest import Manifest
from flipbook.runner import evaluate_async
from flipbook.stats import compare
from flipbook.store import Store, _write_json

log = logging.getLogger(__name__)


class RegressionEvaluator(SamplingClientEvaluator):
    """Drops into ``infrequent_evaluator_builders``"""
    def __init__(
        self,
        *,
        manifest: str,
        baseline: str,
        store: Store,
        log_dir: str | Path | None = None,
        eval_every: int = 8,
        efforts: tuple[float, ...] = (0.9,),
        k: int = 4,
        temperature: float = 0.6,
        max_tokens: int = 32768,
        diverge: bool = True,
        effort_pair: tuple[float, float] | None = (0.2, 0.9),
        study: str | None = None,
        seed: int = 0,
        concurrency: int = 8,
        train_config: dict | None = None,
    ):
        self.manifest_name = manifest
        self.baseline = baseline
        self.store = store
        self.log_dir = Path(log_dir) if log_dir else None
        self.eval_every = eval_every
        self.efforts = efforts
        self.k = k
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.diverge = diverge
        self.effort_pair = effort_pair
        self.study = study
        self.seed = seed
        self.concurrency = concurrency
        self.train_config = train_config
        self.calls = 0
        self._baseline_rid: str | None = None
        self._resolved = None  # Resolved(base_model, renderer) — once per lifetime
    async def __call__(
        self, sampling_client: Any, step: int | None = None
    ) -> dict[str, float]:
        name = f"flipbook/{self.manifest_name}"
        try:
            return await self._run(sampling_client, name, step)
        except Exception:  # the loop must never see eval failures
            log.exception("flipbook eval failed")
            return {f"{name}/error": 1.0}
    async def _run(
        self, sampling_client: Any, name: str, step: int | None = None
    ) -> dict[str, float]:
        # no public accessor for the weights path; the submit response
        # carries it (verified in the instrument check)
        model_path = (await sampling_client._get_sampler_submit()).model_path
        if self._resolved is None:
            # resolve once: renderer is part of RunConfig identity, so the
            # same name `flipbook eval` would produce — runs dedupe across
            # in-loop and cli evaluation. ServiceClient only when the path
            # needs REST (it resolves auth eagerly).
            import tinker

            from flipbook.runner import resolve_model
            sc = tinker.ServiceClient() if model_path.startswith("tinker://") else None
            self._resolved = await resolve_model(model_path, service_client=sc)
        manifest = Manifest.load(self.store, self.manifest_name)
        if step is None:
            step = self.calls * self.eval_every
        self.calls += 1
        self._check_step_alignment(step, model_path)
        # step-0 weights ARE the base model: billing this eval as the baseline
        # (same config a `flipbook eval --model <base> --label baseline` run
        # produces) stops _ensure_baseline from sampling it a second time.
        # The run keeps train_step None so the API/GUI read it as baseline.
        as_baseline = (
            step == 0
            and self._canonical_path(model_path, step) == self._resolved.base_model
            and not self._baseline_rid
            and self._baseline_is_model_name()
        )
        run_ids = {}
        for eff in self.efforts:
            cfg = RunConfig(
                model=self._resolved.base_model if as_baseline else model_path,
                manifest_hash=manifest.manifest_hash,
                effort=eff, temperature=self.temperature,
                max_tokens=self.max_tokens, k=self.k, seed=self.seed,
                renderer=self._resolved.renderer, study=self.study,
                label="baseline" if as_baseline and eff == self.efforts[0] else f"step{step}",
            )
            await evaluate_async(
                cfg, self.store, concurrency=self.concurrency,
                log=lambda m: log.info(m), sampling_client=sampling_client,
            )
            rid = run_id(cfg)
            run_ids[eff] = rid
            if as_baseline:
                if eff == self.efforts[0]:
                    self._baseline_rid = rid
            else:
                self._patch_provenance(rid, step, model_path)
        rid = run_ids[self.efforts[0]]
        baseline_rid = await self._ensure_baseline(manifest)
        pair = compare(self.store, baseline_rid, rid)
        metrics = {
            f"{name}/pass1": pair.acc_b,
            f"{name}/delta_vs_base": pair.delta,
            f"{name}/delta_ci_lo": pair.delta_ci[0],
            f"{name}/delta_ci_hi": pair.delta_ci[1],
            f"{name}/regressions": float(sum(f["kind"] == "regression" for f in pair.flips)),
            f"{name}/gains": float(sum(f["kind"] == "gain" for f in pair.flips)),
            f"{name}/truncation_rate": pair.truncation_rate_b,
            f"{name}/mean_gen_tokens": pair.tokens["b"]["mean"],
            f"{name}/cost_usd": pair.cost_b,
        }
        for n in range(2, self.k + 1):
            if str(n) in pair.passn:
                metrics[f"{name}/pass{n}"] = pair.passn[str(n)]["b"]
        if self.diverge and pair.n_pairs:
            flips = [f["row_id"] for f in pair.flips]
            ds = await diverge_async(
                self.store, baseline_rid, rid, flips or None,
                concurrency=self.concurrency, log=lambda m: log.info(m),
            )
            if ds.n_rows:
                metrics[f"{name}/divergence_mean_nats"] = ds.mean_sum_nats
                metrics[f"{name}/p_skip"] = ds.mean_p_skip_ckpt
        if self.effort_pair:
            e_lo, e_hi = self.effort_pair
            if self.efforts[0] == e_hi:
                es_ckpt = await effort_gap_async(
                    self.store, rid, e_lo, e_hi,
                    concurrency=self.concurrency, log=lambda m: log.info(m),
                )
                es_base = await effort_gap_async(
                    self.store, baseline_rid, e_lo, e_hi,
                    concurrency=self.concurrency, log=lambda m: log.info(m),
                )
                if es_base.mean_gap_nats:
                    metrics[f"{name}/effort_gap_ratio"] = (
                        es_ckpt.mean_gap_nats / es_base.mean_gap_nats
                    )
            if e_lo in run_ids:
                tok_lo = compare(self.store, baseline_rid, run_ids[e_lo]).tokens["b"]["mean"]
                if tok_lo:
                    metrics[f"{name}/effort_len_ratio"] = pair.tokens["b"]["mean"] / tok_lo
        # eval metrics belong in the store the moment they exist - the study
        # page charts these live; cookbook-side loss/epoch still arrive via
        # import-metrics
        if self.study:
            self.store.put_training_metrics(
                self.study,
                [{"study": self.study, "step": step, "key": k, "value": v}
                 for k, v in metrics.items()],
            )
        return metrics
    async def _ensure_baseline(self, manifest: Manifest) -> str:
        """Baseline = a stored run_id, or a model name evaluated once lazily —
        after that the store's resume dedupe means it is never re-sampled."""
        if self._baseline_rid:
            return self._baseline_rid
        try:
            self._baseline_rid = self.store.resolve_run(self.baseline)["run_id"]
            return self._baseline_rid
        except LookupError:
            pass
        # a model name lands here: reuse an existing baseline eval of that
        # model on this manifest+study before paying for another one
        for r in self.store.runs(self.study):
            if (r.get("model_id") == self.baseline
                    and r.get("manifest_hash") == manifest.manifest_hash
                    and r.get("label") == "baseline"):
                log.info("baseline %r is not a run ref - reusing %s (same model+manifest)",
                         self.baseline, r["run_id"])
                self._baseline_rid = r["run_id"]
                return self._baseline_rid
        cfg = RunConfig(
            model=self._resolved.base_model,
            manifest_hash=manifest.manifest_hash,
            effort=self.efforts[0], temperature=self.temperature,
            max_tokens=self.max_tokens, k=self.k, seed=self.seed,
            renderer=self._resolved.renderer, study=self.study, label="baseline",
        )
        await evaluate_async(cfg, self.store, concurrency=self.concurrency,
                             log=lambda m: log.info(m))
        self._baseline_rid = run_id(cfg)
        return self._baseline_rid
    def _baseline_is_model_name(self) -> bool:
        try:
            self.store.resolve_run(self.baseline)
            return False
        except LookupError:
            return True
    def _canonical_path(self, model_path: str, step: int) -> str:
        # ephemeral eval samplers (sampler_weights/ephemeral_N) don't survive
        # server-side; the durable artifact is the numbered checkpoint. Step 0
        # pre-dates the first save, so its weights are the base model itself.
        # Requires eval steps to coincide with save steps (both are multiples
        # of the same cadence in practice).
        if "ephemeral_" not in model_path:
            return model_path
        if step == 0:
            return self._resolved.base_model
        return re.sub(r"/sampler_weights/[^/]+$", f"/sampler_weights/{step:06d}", model_path)

    def _patch_provenance(self, rid: str, step: int, model_path: str) -> None:
        """The run record knows its weights path; add the measured train step."""
        f = self.store.path / "runs" / f"{rid}.json"
        rec = json.loads(f.read_text())
        canon = self._canonical_path(model_path, step)
        rec["provenance"]["train_step_measured"] = step
        rec["provenance"]["model_path"] = canon
        rec["checkpoint_path"] = canon
        rec["train_step"] = step
        if self.train_config:
            rec["provenance"]["train_config"] = self.train_config
        _write_json(rec, f)
    def _check_step_alignment(self, step: int, model_path: str) -> None:
        """Warn if calls*eval_every disagrees with checkpoints.jsonl."""
        if not self.log_dir:
            return
        cks = self.log_dir / "checkpoints.jsonl"
        if not cks.exists():
            return
        recs = [json.loads(line) for line in cks.read_text().splitlines() if line.strip()]
        match = next((r for r in recs if r.get("batch") == step), None)
        if match and match.get("sampler_path") and match["sampler_path"] != model_path:
            log.warning(
                "step alignment: calls*eval_every=%d but checkpoints.jsonl path %s != sampled %s",
                step, match["sampler_path"], model_path,
            )
