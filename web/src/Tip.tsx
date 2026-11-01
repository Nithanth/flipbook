import type { ReactNode } from "react";

/** Hover tooltip for metric jargon. CSS-only - the text rides in data-tip so no JS positioning is needed. */
export default function Tip({
  text,
  right,
  children,
}: {
  text: string;
  right?: boolean;
  children: ReactNode;
}) {
  return (
    <span className={right ? "tip tip-right" : "tip"} data-tip={text}>
      {children}
    </span>
  );
}
