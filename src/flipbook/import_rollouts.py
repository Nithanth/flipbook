"""Import a cookbook RL log dir's per-rollout records into the store.

The rollout_summaries jsonl carries rewards and lengths per trajectory step;
the logtree carries the sampled text but no rollout ids (only positions in a
rendered tree), so transcripts are stored positionally and never linked to
summary rows.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from flipbook.store import Store


@dataclass
class RolloutImport:
    n_rollouts: int
    n_steps: int
    n_texts: int
    iterations: int
    splits: list[str]
    text_coverage: dict[str, tuple[float, float]]


def _children(node: dict) -> list[dict]:
    return [c for c in node.get("children", []) if isinstance(c, dict)]


def _text(node) -> str:
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        return "".join(_text(c) for c in node.get("children", []))
    return ""


def _h_text(sec: dict, tag: str) -> str:
    for c in _children(sec):
        if c.get("tag") == tag:
            return _text(c).strip()
    return ""


def _subsections(sec: dict) -> list[dict]:
    subs = []

    def walk(node):
        for c in _children(node):
            if c.get("tag") == "section" and _h_text(c, "h3"):
                subs.append(c)
            else:
                walk(c)

    walk(sec)
    return subs


def _conversations(sec: dict) -> list[list[dict]]:
    msgs = []

    def walk(node):
        for c in _children(node):
            data = c.get("data")
            if isinstance(data, dict) and data.get("type") == "conversation":
                msgs.extend(data.get("messages", []))
            walk(c)

    walk(sec)
    return msgs


def _reward_fields(sec: dict) -> dict[str, str]:
    fields = {}
    for tr in _find(sec, "tr"):
        tds = [_text(td).strip() for td in _children(tr) if td.get("tag") == "td"]
        if len(tds) >= 2:
            fields[tds[0]] = tds[1]
    return fields


def _find(node: dict, tag: str) -> list[dict]:
    found = []
    if node.get("tag") == tag:
        found.append(node)
    for c in _children(node):
        found.extend(_find(c, tag))
    return found


def _logtree_texts(tree: dict) -> list[list[dict]]:
    """Per "Group Rollout" section, one record per h3 Prompt block; Policy
    Response messages append to the open trajectory, Reward fields too."""
    groups = []
    for sec in _children(tree.get("root") or {}):
        if _h_text(sec, "h2") != "Group Rollout":
            continue
        trajs, cur = [], None
        for sub in _subsections(sec):
            title = _h_text(sub, "h3")
            if title == "Prompt":
                cur = {"prompt": [], "response": [], "reward": {}}
                trajs.append(cur)
                cur["prompt"].extend(_conversations(sub))
            elif title == "Policy Response" and cur is not None:
                cur["response"].extend(_conversations(sub))
            elif title == "Reward" and cur is not None:
                cur["reward"].update(_reward_fields(sub))
        groups.append(trajs)
    return groups


def _content_parts(content) -> tuple[str, str]:
    """(thinking, text) for a message whose content is a part list or a str."""
    if not isinstance(content, list):
        return "", content or ""
    think, text = [], []
    for p in content:
        if p.get("type") == "thinking":
            think.append(p.get("thinking") or "")
        elif p.get("type") == "text":
            text.append(p.get("text") or "")
    return "".join(think), "".join(text)


def import_rollouts(store: Store, log_dir: str | Path, study: str) -> RolloutImport:
    from tinker_cookbook.stores.storage import LocalStorage
    from tinker_cookbook.stores.training_store import TrainingRunStore

    trs = TrainingRunStore(LocalStorage(str(log_dir)))
    iterations = trs.list_iterations()
    rows, texts = [], []
    trajs_in: dict[tuple[str, int], int] = {}
    texts_in: dict[tuple[str, int], int] = {}
    splits: set[str] = set()
    for it in iterations:
        base_names = (["train"] if it.has_train_rollouts else []) + [
            f"eval_{label}" for label in it.eval_labels
        ]
        for bn in base_names:
            recs = trs.read_rollouts(it.iteration, bn)
            if not recs:
                continue
            split = "train" if bn == "train" else bn[len("eval_"):]
            splits.add(split)
            trajs_in[(split, it.iteration)] = len(recs)
            for rec in recs:
                for s in rec["steps"]:
                    rows.append({
                        "study": study, "split": split,
                        "iteration": it.iteration,
                        "policy_step": rec["sampling_client_step"],
                        "group_idx": rec["group_idx"],
                        "traj_idx": rec["traj_idx"],
                        "step_idx": s["step_idx"],
                        "n_steps": len(rec["steps"]),
                        "total_reward": rec["total_reward"],
                        "final_reward": rec["final_reward"],
                        "step_reward": s["reward"],
                        "ob_len": s["ob_len"], "ac_len": s["ac_len"],
                        "episode_done": s["episode_done"],
                        # WHY JSON strings: component names vary per env and
                        # readers always want all of a step's parts together
                        "metrics": json.dumps(s.get("metrics") or {}),
                        "trajectory_metrics": json.dumps(
                            rec.get("trajectory_metrics") or {}),
                    })
            tree = trs.read_logtree(it.iteration, bn)
            if not tree:
                continue
            n = 0
            for gp, trajs in enumerate(_logtree_texts(tree)):
                for tp, t in enumerate(trajs):
                    thinking = "".join(
                        _content_parts(m.get("content"))[0] for m in t["response"])
                    answer = "".join(
                        _content_parts(m.get("content"))[1] for m in t["response"])
                    texts.append({
                        "study": study, "split": split,
                        "iteration": it.iteration,
                        "group_pos": gp, "traj_pos": tp,
                        "prompt": json.dumps(t["prompt"]),
                        "response": json.dumps(t["response"]),
                        "thinking": thinking, "answer": answer,
                        "reward_fields": json.dumps(t["reward"]),
                    })
                    n += 1
            texts_in[(split, it.iteration)] = n
    store.put_rollouts(study, rows)
    store.put_rollout_texts(study, texts)
    coverage = {}
    for split in splits:
        covs = [
            texts_in.get((split, it), 0) / n_traj
            for (sp, it), n_traj in trajs_in.items()
            if sp == split and n_traj
        ]
        if covs:
            coverage[split] = (min(covs), max(covs))
    return RolloutImport(
        n_rollouts=sum(trajs_in.values()),
        n_steps=len(rows),
        n_texts=len(texts),
        iterations=len(iterations),
        splits=sorted(splits),
        text_coverage=coverage,
    )
