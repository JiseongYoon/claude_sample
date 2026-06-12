// DirectChat — a TOOLLESS, multi-turn chat over REST /chat (). Visually + behaviourally
// distinct from the gated Agent tab: it states "no tools / not gated" so the user knows no tool loop or
// HITL gate is involved. Server text rendered as escaped React children (no injection). Reuses the chat
// bubble styles. The shared system prompt (from ChatTabs) is applied per send.
import { useEffect, useRef, useState } from "react";
import type { ApiClient } from "../api/client";
import { DirectChatController, type DirectState } from "../state/directChat";
import styles from "./Chat.module.css";

export function DirectChat({
 client,
 system,
 attachments,
}: {
 client: ApiClient;
 system?: string;
 attachments?: string[];
}) {
 const [state, setState] = useState<DirectState>({ messages: [], status: "idle", notice: null });
 const [input, setInput] = useState("");
 const ctrlRef = useRef<DirectChatController | null>(null);

 useEffect(() => {
 const ctrl = new DirectChatController(client);
 ctrlRef.current = ctrl;
 const off = ctrl.subscribe(setState);
 return () => {
 off();
 ctrl.dispose();
 ctrlRef.current = null;
 };
 }, [client]);

 const sending = state.status === "sending";

 function submit(e: React.FormEvent) {
 e.preventDefault();
 const t = input.trim();
 if (!t || sending) return;
 void ctrlRef.current?.send(t, system, attachments);
 setInput("");
 }

 return (
 <section aria-label="direct-chat" className={styles.panel}>
 <h2 className={styles.heading}>Direct chat</h2>
 <small className={styles.meta} data-testid="direct-toolless-note">
 no tools · not gated · direct model completion
 </small>
 <ul className={styles.messages} data-testid="direct-messages">
 {state.messages.map((m, i) => (
 <li key={i} className={styles.msg} data-role={m.role === "assistant" ? "agent" : "user"}>
 <span>{m.content}</span>
 </li>
 ))}
 </ul>
 {sending && (
 <p className={styles.running} data-testid="direct-sending">
 sending…
 </p>
 )}
 {state.notice && (
 <p className={styles.running} data-testid="direct-notice">
 {state.notice}
 </p>
 )}
 <form className={styles.form} onSubmit={submit}>
 <input
 aria-label="direct-task"
 value={input}
 onChange={(e) => setInput(e.target.value)}
 disabled={sending}
 placeholder="Chat directly with the model…"
 />
 <button type="submit" disabled={sending || !input.trim()}>
 Send
 </button>
 </form>
 </section>
 );
}
