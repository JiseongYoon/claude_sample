// ChatTabs — the chat workspace (+ ). Two DISTINCT tabs (DE4): the gated
// **Agent** chat (run_task → tool loop → HITL) and the toolless **Direct** chat (/chat, no gate). A single
// shared **system prompt** (persona) AND a shared set of **attachments** (uploaded docs) apply to both.
// Upload is ZERO authority: the UI POSTs the file and relays the opaque id the server returns; it holds
// no path and performs no authoritative validation — the server gate/policy is the sole authority. A 403
// means the token lacks the `ingest` scope (mint it in Auth). Keeping the tabs separate makes it
// unambiguous which path runs (gated/tool-using vs direct).
import { useEffect, useRef, useState } from "react";
import type { ApiClient } from "../api/client";
import { CapabilitiesController } from "../state/capabilities";
import { Chat } from "./Chat";
import { DirectChat } from "./DirectChat";
import styles from "./ChatTabs.module.css";

interface Attachment {
 id: string;
 filename: string;
}

// the file types the picker SUGGESTS (a UX `accept` hint only — the server allowlist is authoritative).
const DOC_ACCEPT = ".txt,.md,.markdown,.html,.htm,.pdf,.docx,.hwpx";
const IMAGE_ACCEPT = ".png,.jpg,.jpeg,.webp";

// map the upload endpoint's status (status-only error message) → a friendly, non-leaking notice.
function uploadNotice(err: unknown): string {
 const msg = err instanceof Error ? err.message : "";
 const m = /upload failed: (\d+)/.exec(msg);
 const code = m ? Number(m[1]) : 0;
 if (code === 413) return "file too large";
 if (code === 415) return "unsupported file type";
 if (code === 400) return "invalid file";
 if (code === 429) return "too many files";
 if (code === 403) return "requires the ingest scope (re-mint with ingest)";
 if (code === 503) return "ingestion unavailable on the server";
 return "upload failed";
}

export function ChatTabs({ client }: { client: ApiClient }) {
 const [tab, setTab] = useState<"agent" | "direct">("agent");
 const [system, setSystem] = useState("");
 const [attachments, setAttachments] = useState<Attachment[]>([]);
 const [uploading, setUploading] = useState(false);
 const [uploadErr, setUploadErr] = useState<string | null>(null);
 const [multimodal, setMultimodal] = useState(false);
 const fileRef = useRef<HTMLInputElement | null>(null);

 // reflect whether the server's `multimodal` capability is available (vision projector
 // present + serving up). Pure UX — gates NOTHING (the server validates the upload type + the tool is
 // capability-gated server-side); we only suggest image types + show a hint when images are accepted.
 useEffect(() => {
 const caps = new CapabilitiesController(client);
 const off = caps.subscribe((s) =>
 setMultimodal(s.caps.some((c) => c.name === "multimodal" && c.available)),
 );
 void caps.refresh();
 return () => {
 off();
 caps.dispose();
 };
 }, [client]);

 async function onPick(e: React.ChangeEvent<HTMLInputElement>) {
 const file = e.target.files?.[0];
 if (fileRef.current) fileRef.current.value = ""; // reset so the same file can be re-picked
 if (!file || uploading) return;
 setUploading(true);
 setUploadErr(null);
 try {
 const r = await client.uploadFile(file);
 setAttachments((a) => [...a, { id: r.id, filename: r.filename }]);
 } catch (err) {
 setUploadErr(uploadNotice(err));
 } finally {
 setUploading(false);
 }
 }
 function remove(id: string) {
 setAttachments((a) => a.filter((x) => x.id !== id));
 }
 const ids = attachments.map((a) => a.id);

 return (
 <section aria-label="chat-tabs" className={styles.wrap}>
 <div className={styles.tabs} role="tablist">
 <button
 type="button"
 role="tab"
 aria-selected={tab === "agent"}
 className={tab === "agent" ? styles.active : undefined}
 onClick={() => setTab("agent")}
 >
 Agent
 </button>
 <button
 type="button"
 role="tab"
 aria-selected={tab === "direct"}
 className={tab === "direct" ? styles.active : undefined}
 onClick={() => setTab("direct")}
 >
 Direct
 </button>
 </div>

 <label className={styles.system}>
 System prompt (persona — applies to both)
 <textarea
 aria-label="system-prompt"
 value={system}
 onChange={(e) => setSystem(e.target.value)}
 rows={2}
 placeholder="optional — e.g. 'You are a concise assistant.'"
 />
 </label>

 <div className={styles.attach} aria-label="attachments">
 <div className={styles.uploadRow}>
 <input
 ref={fileRef}
 type="file"
 aria-label="attach-file"
 accept={multimodal ? `${DOC_ACCEPT},${IMAGE_ACCEPT}` : DOC_ACCEPT}
 onChange={onPick}
 disabled={uploading}
 />
 {uploading && <span className={styles.uploadStatus} data-testid="upload-status">uploading…</span>}
 <span className={styles.uploadStatus} data-testid="attach-hint">
 {multimodal ? "documents · images · scans" : "text documents (images need a multimodal model)"}
 </span>
 </div>
 {uploadErr && (
 <p className={styles.uploadErr} data-testid="upload-error">
 {uploadErr}
 </p>
 )}
 {attachments.length > 0 && (
 <ul className={styles.chips} data-testid="attachment-chips">
 {attachments.map((a) => (
 <li key={a.id} className={styles.chip}>
 {/* filename rendered as escaped React text — a hostile name cannot inject markup */}
 <span>{a.filename}</span>
 <button
 type="button"
 aria-label={`remove-${a.filename}`}
 className={styles.chipRemove}
 onClick={() => remove(a.id)}
 >
 ×
 </button>
 </li>
 ))}
 </ul>
 )}
 </div>

 {/* / ISSUE-001 fix (DH6): keep BOTH tabs MOUNTED and hide the inactive one with
 CSS, so switching tabs no longer unmounts the active view and WIPES its chat history /
 in-flight run. Each tab keeps its own state (incl. the Agent's multi-turn transcript, the
 source for run_task.history). The Direct tab is REST (/chat); the Agent tab is the gated WS
 run — both mounted is safe (no shared WS-event handling conflict). */}
 <div hidden={tab !== "agent"} data-testid="agent-panel">
 <Chat client={client} system={system} attachments={ids} />
 </div>
 <div hidden={tab !== "direct"} data-testid="direct-panel">
 <DirectChat client={client} system={system} attachments={ids} />
 </div>
 </section>
 );
}
