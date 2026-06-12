// Chat view — send a task, render the run with LIVE streamed tokens + tool activity (
// /), show the terminal result, and replay prior turns so the agent remembers. All
// server-provided text (streamed tokens, the answer, tool names) is rendered as React children →
// ESCAPED by default (no dangerouslySetInnerHTML), so content containing HTML/script is shown
// literally, never executed. ZERO authority: streaming/history are display + replay only; the
// server gate/approval path is unchanged.
import { useEffect, useRef, useState } from "react";
import type { ApiClient } from "../api/client";
import type { HistoryMessage } from "../api/types";
import { RunController, type RunState, type ToolActivity } from "../state/run";
import { Citations } from "./Citations";
import styles from "./Chat.module.css";

interface Msg {
 id: number;
 role: "user" | "agent" | "system";
 text: string;
 meta?: string;
}

interface Live {
 streamedText: string;
 activity: ToolActivity[];
}

const EMPTY_LIVE: Live = { streamedText: "", activity: [] };

// Prior turns → the bounded history the server replays into the agent's context. Only real
// user/agent turns (system notices like errors are excluded); the server further bounds + validates.
function toHistory(messages: Msg[]): HistoryMessage[] {
 const out: HistoryMessage[] = [];
 for (const m of messages) {
 if (m.role === "user") out.push({ role: "user", content: m.text });
 else if (m.role === "agent") out.push({ role: "assistant", content: m.text });
 }
 return out;
}

export function Chat({
 client,
 timeoutMs,
 system,
 attachments,
}: {
 client: ApiClient;
 timeoutMs?: number;
 system?: string;
 attachments?: string[];
}) {
 const [messages, setMessages] = useState<Msg[]>([]);
 const [runStatus, setRunStatus] = useState<RunState["status"]>("idle");
 const [live, setLive] = useState<Live>(EMPTY_LIVE);
 const [input, setInput] = useState("");
 const ctrlRef = useRef<RunController | null>(null);
 const idRef = useRef(0);
 const messagesRef = useRef<Msg[]>([]);
 messagesRef.current = messages;

 function push(role: Msg["role"], text: string, meta?: string) {
 idRef.current += 1;
 const id = idRef.current;
 setMessages((m) => [...m, { id, role, text, meta }]);
 }

 useEffect(() => {
 const ctrl = new RunController(client, timeoutMs !== undefined ? { timeoutMs } : {});
 ctrlRef.current = ctrl;
 const off = ctrl.subscribe((s) => {
 setRunStatus(s.status);
 if (s.status === "running") {
 setLive({ streamedText: s.streamedText, activity: s.activity });
 } else if (s.status === "done") {
 const r = s.result;
 const activity =
 s.activity.length > 0
 ? ` · tools: ${s.activity.map((a) => `${a.tool}${a.outcome ? `(${a.outcome})` : ""}`).join(", ")}`
 : "";
 push(
 "agent",
 r.answer ?? "(no answer)",
 `status: ${r.status} · steps: ${r.steps} · tools: ${r.tool_calls_made}${activity}`,
 );
 setLive(EMPTY_LIVE);
 ctrl.reset();
 } else if (s.status === "error") {
 push("system", `error: ${s.reason}`);
 setLive(EMPTY_LIVE);
 ctrl.reset();
 } else if (s.status === "timeout") {
 push("system", "timed out — no response from the agent");
 setLive(EMPTY_LIVE);
 ctrl.reset();
 }
 });
 return () => {
 off();
 ctrl.dispose();
 };
 }, [client, timeoutMs]);

 const running = runStatus === "running";

 function submit(e: React.FormEvent) {
 e.preventDefault();
 const t = input.trim();
 if (!t || running) return; // single-flight guard (button is also disabled)
 const history = toHistory(messagesRef.current); // prior turns BEFORE this new one
 push("user", t);
 ctrlRef.current?.start(t, system, attachments, history);
 setInput("");
 }

 return (
 <section aria-label="chat" className={styles.panel}>
 <h2 className={styles.heading}>Agent</h2>
 <ul className={styles.messages} data-testid="messages">
 {messages.map((m) => (
 <li key={m.id} className={styles.msg} data-role={m.role}>
 <span>{m.text}</span>
 {m.meta && (
 <small className={styles.meta} data-testid="tool-activity">
 {" "}
 — {m.meta}
 </small>
 )}
 {m.role === "agent" && <Citations text={m.text} />}
 </li>
 ))}
 </ul>
 {running && (
 <div className={styles.running} data-testid="running">
 {/* live tool activity */}
 {live.activity.length > 0 && (
 <ul className={styles.activity} data-testid="live-activity">
 {live.activity.map((a) => (
 <li key={a.id}>
 {a.tool}
 {a.outcome ? ` → ${a.outcome}` : " …"}
 </li>
 ))}
 </ul>
 )}
 {/* live streamed answer — escaped React text */}
 {live.streamedText ? (
 <p className={styles.streaming} data-testid="streaming-text">
 {live.streamedText}
 </p>
 ) : (
 <p>running…</p>
 )}
 </div>
 )}
 <form className={styles.form} onSubmit={submit}>
 <input
 aria-label="task"
 value={input}
 onChange={(e) => setInput(e.target.value)}
 disabled={running}
 placeholder="Ask the agent to do something…"
 />
 <button type="submit" disabled={running || !input.trim()}>
 Send
 </button>
 </form>
 </section>
 );
}
