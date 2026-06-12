// extractSources — best-effort citation/source extraction from an agent answer's TEXT ().
//
// Today the chat receives only the agent's terminal free-text answer (task_result.answer); the DocQA /
// web_answer tools DO produce structured `citations`, but those live in the tool RESULT the model
// consumes — a structured citation channel to the UI is a concern. So
// surfaces whatever the answer text itself carries: http(s) URLs and `[source…]`/`[…#N]` reference
// tokens (the DocQA citation shape). Pure + graceful: no match → []. Callers render the result as
// ESCAPED text (never as live links) since the answer is untrusted.

const URL_RE = /\bhttps?:\/\/[^\s<>"')\]]+/gi;
const REF_RE = /\[[^\]\n]*(?:source|#\d+)[^\]\n]*\]/gi;

export function extractSources(text: string): string[] {
 if (!text || typeof text !== "string") return [];
 const out = new Set<string>();
 for (const m of text.match(URL_RE) ?? []) out.add(m.replace(/[.,;:]+$/, "")); // trim trailing punctuation
 for (const m of text.match(REF_RE) ?? []) out.add(m);
 return [...out];
}
