"""Summarize training rollouts for inspection.

Episodes, not steps, are the averaging unit; logged transcripts stay unlinked.
"""

import json
import math
from collections import defaultdict


def rollout_summary(rows: list[dict], texts: list[dict]) -> dict:
    episodes = defaultdict(list)
    for r in rows:
        episodes[(r["split"], r["iteration"], r["group_idx"], r["traj_idx"])].append(r)
    iterations = defaultdict(list)
    for (split, iteration, _, _), steps in episodes.items():
        last = max(steps, key=lambda r: r["step_idx"])
        # Use final-step metrics rather than summing flags across the episode.
        components = {
            k: float(v) for k, v in json.loads(last["metrics"] or "{}").items()
            if isinstance(v, (int, float)) and math.isfinite(v)
        }
        iterations[(split, iteration)].append({
            "tokens": sum(s["ac_len"] or 0 for s in steps),
            "reward": last["total_reward"],
            "components": components,
        })
    text_counts = defaultdict(int)
    fields = defaultdict(lambda: defaultdict(set))
    for t in texts:
        text_counts[(t["split"], t["iteration"])] += 1
        for k, v in json.loads(t["reward_fields"] or "{}").items():
            fields[t["split"]][k].add(v)
    series = defaultdict(list)
    coverage = defaultdict(lambda: {"rollouts": 0, "texts": 0})
    for (split, iteration), eps in sorted(iterations.items()):
        values = defaultdict(list)
        for ep in eps:
            for k, v in ep["components"].items():
                values[k].append(v)
        rewards = [e["reward"] for e in eps
                   if e["reward"] is not None and math.isfinite(e["reward"])]
        n_texts = text_counts[(split, iteration)]
        series[split].append({
            "iteration": iteration, "n": len(eps), "n_texts": n_texts,
            "tokens": sum(e["tokens"] for e in eps) / len(eps),
            "reward": sum(rewards) / len(rewards) if rewards else None,
            "components": {k: sum(v) / len(v) for k, v in sorted(values.items())},
            "component_n": {k: len(v) for k, v in sorted(values.items())},
        })
        coverage[split]["rollouts"] += len(eps)
        coverage[split]["texts"] += n_texts
    return {
        "splits": sorted(series), "series": dict(series),
        "events": {split: _events(points) for split, points in series.items()},
        "coverage": dict(coverage),
        "text_fields": {
            split: sorted(k for k, vs in fields[split].items() if vs <= {"True", "False"})
            for split in series
        },
    }


def _events(points: list[dict]) -> list[dict]:
    """Threshold hints on trailing three-iteration means, not statistical tests."""
    events = []
    keys = sorted({k for p in points for k in p["components"]})
    for k in keys:
        values = [p["components"].get(k) for p in points]
        if any(v is None or not 0 <= v <= 1 for v in values) or len(values) < 3:
            continue
        smooth = [sum(values[i - 2:i + 1]) / 3 for i in range(2, len(values))]
        peak = smooth[0]
        below = smooth[0] < 0.98
        emitted = set()
        for i, value in enumerate(smooth):
            peak = max(peak, value)
            kind, start = None, None
            if value >= 0.98 and below and "saturated" not in emitted:
                kind, start = "saturated", smooth[0]
            elif value <= peak - 0.10 + 1e-12 and "falling" not in emitted:
                kind, start = "falling", peak
            if kind:
                events.append({"component": k, "kind": kind,
                               "iteration": points[i + 2]["iteration"],
                               "from": start, "to": value})
                emitted.add(kind)
            below |= value < 0.98
    return sorted(events, key=lambda e: (e["iteration"], e["component"], e["kind"]))


def rollout_texts(
    texts: list[dict], split: str, iteration: int,
    field: str | None, value: str | None, sort: str, limit: int,
) -> dict:
    available = sorted({t["iteration"] for t in texts if t["split"] == split})
    rows = []
    for t in texts:
        if t["split"] != split or t["iteration"] != iteration:
            continue
        rewards = json.loads(t["reward_fields"] or "{}")
        if field is not None and rewards.get(field) != value:
            continue
        prompt = json.loads(t["prompt"] or "[]")
        content = next((m.get("content", "") for m in reversed(prompt)
                        if m.get("role") == "user"), "")
        question = content if isinstance(content, str) else "".join(
            p.get("text", "") for p in content if p.get("type") == "text")
        thinking, answer = t["thinking"] or "", t["answer"] or ""
        rows.append({
            "iteration": iteration, "group_pos": t["group_pos"], "traj_pos": t["traj_pos"],
            "reward_fields": rewards, "question": question,
            "thinking": thinking, "answer": answer, "chars": len(thinking) + len(answer),
        })
    rows.sort(key=lambda t: (t["group_pos"], t["traj_pos"]))
    if sort in ("short", "long"):
        rows.sort(key=lambda t: t["chars"], reverse=sort == "long")
    return {"iterations": available, "total": len(rows), "rows": rows[:limit]}
