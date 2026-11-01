import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

/**
 * Markdown + LaTeX renderer for model text (questions and sample outputs).
 * remark-math only parses $...$ / $$...$$, so normalize() first rewrites
 * \(...\) and \[...\] to dollar delimiters (otherwise markdown eats the
 * backslashes and the LaTeX shows raw); then it wraps bare \boxed{...}
 * outside math in $...$ - models emit boxed outside math mode constantly.
 */

// Pandoc's rule for single-$ math: the opening $ can't be followed by
// whitespace and the closing $ can't be followed by a digit - so "$4 each and
// a $20 bill" is currency, not a math span
const MATH_SPAN =
  /(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|\$(?!\s)[^$]+?\$(?!\d))/g;
const BOXED = /\\boxed\{(?:[^{}]|\{[^{}]*\})*\}/g;

function normalize(text: string): string {
  const dollars = text
    // [asy]...[/asy] is Asymptote figure source embedded in problems -
    // fence it so it renders as code, not loose prose (and before the math
    // pass, which would mangle any $ inside it)
    .replace(/\[asy\]([\s\S]*?)\[\/asy\]/g, (_, m) => `\n\`\`\`asy\n${m.trim()}\n\`\`\`\n`)
    .replace(/\\\[([\s\S]+?)\\\]/g, (_, m) => `$$${m}$$`)
    .replace(/\\\(([\s\S]+?)\\\)/g, (_, m) => `$${m}$`);
  // split keeps math spans at odd indices - only rewrite the text between them
  return dollars
    .split(MATH_SPAN)
    .map((seg, i) =>
      i % 2 === 1
        ? seg
        : seg
            // every $ left in a text segment failed the math rule - escape it
            // so remark-math can't re-pair it with a later stray $
            .replace(/\$/g, "\\$")
            .replace(BOXED, (m) =>
              // an empty \boxed{} is prompt boilerplate ("answer in \boxed{}"),
              // not math - show the literal syntax instead of an empty frame
              m === "\\boxed{}" ? "`\\boxed{}`" : `$${m}$`,
            ),
    )
    .join("");
}

export default function Md({ text }: { text: string }) {
  return (
    <div className="md">
      <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
        {normalize(text)}
      </ReactMarkdown>
    </div>
  );
}
