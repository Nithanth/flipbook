import { useEffect, useState } from "react";
import { api, type Manifest } from "../api";

const quote = (s: string) => "'" + s.replaceAll("'", "'\\''") + "'";
type Mode = "Saved checkpoints" | "SFT recipe" | "RL recipe";

export default function Connect() {
  const [mode, setMode] = useState<Mode>("Saved checkpoints");
  const [study, setStudy] = useState("");
  const [manifest, setManifest] = useState("");
  const [manifests, setManifests] = useState<Manifest[]>([]);
  const [error, setError] = useState("");
  const [logDir, setLogDir] = useState("");
  const [baseline, setBaseline] = useState("");
  const [store, setStore] = useState("./flipbook_store");
  useEffect(() => {
    api.manifests().then(setManifests).catch(() => setError("Could not load frozen evaluations. Enter the manifest name from your store."));
  }, []);
  const saved = mode === "Saved checkpoints";
  const ready = study.trim() && manifest.trim() && store.trim() && (saved ? logDir.trim() : baseline.trim());
  const track = `uv run flipbook track --manifest=${quote(manifest)} --study=${quote(study)} --store=${quote(store)} --include-base`;
  const python = JSON.stringify;
  const hook = mode === "SFT recipe" ? "infrequent_eval" : "eval";
  return (
    <section className="connect">
      <h1>Connect a Tinker run</h1>
      <p>Compare checkpoints on the same frozen evaluation. Where are the weights coming from?</p>
      <div className="row" role="group" aria-label="Run source">
        {(["Saved checkpoints", "SFT recipe", "RL recipe"] as const).map(m =>
          <button key={m} aria-pressed={m === mode} onClick={() => setMode(m)}>{m}</button>)}
      </div>
      <div className="connect-fields">
        <label>Study name<input placeholder="my-experiment" value={study} onChange={e => setStudy(e.target.value)} /></label>
        <label>Frozen evaluation<input list="connect-manifests" placeholder="Manifest name" value={manifest} onChange={e => setManifest(e.target.value)} /></label>
        <datalist id="connect-manifests">{manifests.map(m => <option key={m.manifest_hash} value={m.name}>{m.n_rows} questions</option>)}</datalist>
        {saved ? <label>Cookbook log directory<input placeholder="/path/to/run" value={logDir} onChange={e => setLogDir(e.target.value)} /></label>
          : <label>Baseline model or run<input placeholder="Model ID or study/baseline" value={baseline} onChange={e => setBaseline(e.target.value)} /></label>}
        <label>Flipbook store<input value={store} onChange={e => setStore(e.target.value)} /></label>
      </div>
      {error && <p role="alert">{error}</p>}
      <p className="sub">Use the store served by this dashboard. To create a frozen evaluation, run <code>uv run flipbook freeze --help</code>.</p>
      {saved ? <>
        <h2>Evaluate saved checkpoints</h2>
        <p>Use the cookbook directory containing checkpoints.jsonl with saved sampler paths. Run the forecast first, then start the paid evaluation in your Tinker environment.</p>
        {ready && <pre>{`${track} --forecast -- ${quote(logDir)}\n\n${track} -- ${quote(logDir)}`}</pre>}
      </> : <>
        <h2>Attach the checkpoint evaluator</h2>
        <p>Before starting your recipe, add this evaluator and merge the settings into your cookbook config. It samples the baseline and checkpoints using your Tinker account.</p>
        {ready && <pre>{`from flipbook.evaluator import RegressionEvaluator
from flipbook.store import Store

evaluator = RegressionEvaluator(
    study=${python(study)}, manifest=${python(manifest)},
    baseline=${python(baseline)}, store=Store(${python(store)}),
    eval_every=8, diverge=False, effort_pair=None,
)

# Apply these settings to your recipe config.
# Append the builder if your recipe already has evaluators.
${hook}_every = 8
save_every = 8
${mode === "SFT recipe" ? "infrequent_evaluator_builders" : "evaluator_builders"} = [lambda: evaluator]`}</pre>}
      </>}
      {ready ? <p>After the first results are written, <a href={`#/study?name=${encodeURIComponent(study)}`}>open this study</a>.</p>
        : <p className="sub">Fill in the fields above to generate the {saved ? "commands" : "recipe settings"}.</p>}
    </section>
  );
}
