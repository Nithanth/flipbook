"""Import a cookbook log dir's metrics.jsonl into training_metrics/.
"""


from pathlib import Path

from flipbook.store import Store


def import_metrics(store: Store, log_dir: str | Path, study: str) -> int:
    from tinker_cookbook.stores.storage import LocalStorage
    from tinker_cookbook.stores.training_store import TrainingRunStore
    recs = TrainingRunStore(LocalStorage(str(log_dir))).read_metrics()
    rows = [
        {"study": study, "step": int(rec["step"]), "key": k, "value": float(v)}
        for rec in recs
        for k, v in rec.items()
        if k != "step" and isinstance(v, (int, float))
    ]
    store.put_training_metrics(study, rows)
    return len(rows)
