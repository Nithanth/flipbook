import type { Run } from "./api";

/** Display name for a run: its label, else a step tag, else the model's last path segment, else a short id. */
export function runLabel(r: Run | undefined, id?: string): string {
  if (r?.label) return r.label;
  if (r?.train_step != null) return `step${r.train_step}`;
  const model = (r?.model_id as string | undefined) ?? r?.model;
  if (model) return model.split("/").filter(Boolean).pop() ?? model;
  return (id ?? r?.run_id ?? "").slice(0, 10);
}
